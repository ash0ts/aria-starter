# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "marimo==0.25.2",
#   "coreweave==0.1.0b0",
#   "openai==2.54.0",
#   "wandb==0.31.0",
#   "scikit-learn==1.9.1",
#   "python-dotenv==1.2.4",
#   "openpipe-art==0.5.21",
#   "joblib==1.6.0",
#   "pytest==9.1.1",
# ]
# ///
"""Offline contracts for the independently runnable support-desk notebooks."""

import asyncio
import copy
import io
import json
import os
import runpy
import socket
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

STARTER = Path(__file__).resolve().parent
NOTEBOOKS = ("train_classifier", "build_agent", "finetune_classifier")


@pytest.fixture(scope="module")
def notebooks(tmp_path_factory):
    """Opening every notebook, even twice, must not contact external services."""
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("starter-opening"))
        for key in tuple(os.environ):
            if key.startswith(("WANDB_", "OPENAI_", "FORGE_", "COREWEAVE_")):
                patch.delenv(key)
        patch.setenv("PYTHON_DOTENV_DISABLED", "1")
        attempts = []

        def forbid_service(*args, **kwargs):
            attempts.append((args, kwargs))
            raise AssertionError("Notebook attempted a service call without submission")

        patch.setattr(socket.socket, "connect", forbid_service)
        patch.setattr(socket.socket, "connect_ex", forbid_service)
        patch.setattr(socket, "create_connection", forbid_service)

        import dotenv
        import openai
        import wandb
        from coreweave.forge.agentlens import tracing

        patch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
        patch.setattr(openai.OpenAI, "__init__", forbid_service)
        patch.setattr(openai.AsyncOpenAI, "__init__", forbid_service)
        for method in ("init", "login", "Api"):
            patch.setattr(wandb, method, forbid_service)
        patch.setattr(tracing, "init", forbid_service)
        patch.setattr(tracing, "start_conversation", forbid_service)

        loaded = {}
        for name in NOTEBOOKS:
            module = runpy.run_path(str(STARTER / f"{name}.py"))
            for _ in range(2):
                _, definitions = module["app"].run()
            loaded[name] = definitions
        assert not attempts, "A notebook tried to contact a service while opening"
        yield loaded


def test_opening_notebooks_does_not_submit_or_evaluate_test_set(notebooks):
    cpu, agent, sft = (notebooks[name] for name in NOTEBOOKS)
    assert cpu["upload_form"].value is None
    assert cpu["get_upload_result"]() is None
    assert cpu["get_final_result"]() is None
    assert agent["agent_form"].value is None
    assert agent["get_experiments"]() == []
    for form in ("training_form", "evaluation_form", "recovery_form"):
        assert sft[form].value is None
    assert "training_result" not in sft
    assert "comparison" not in sft


def test_independent_notebooks_share_identical_fixtures(notebooks):
    canonical = notebooks["train_classifier"]
    for name in NOTEBOOKS[1:]:
        for key in (
            "DATASET_VERSION",
            "DATASET_SHA256",
            "LABELS",
            "TICKETS",
            "STORE_POLICY",
            "ORDERS",
            "PRODUCTS",
            "AGENT_SCENARIOS",
        ):
            assert notebooks[name][key] == canonical[key], f"{name}: {key} drifted"


def test_fixture_splits_are_unique_disjoint_and_balanced(notebooks):
    data = notebooks["train_classifier"]
    tickets = data["TICKETS"]
    assert len({row["id"] for row in tickets}) == len(tickets)
    assert len({row["text"].strip().casefold() for row in tickets}) == len(tickets)
    families = defaultdict(set)
    by_split = defaultdict(Counter)
    for row in tickets:
        families[row["family"]].add(row["split"])
        by_split[row["split"]][row["label"]] += 1
    assert all(len(splits) == 1 for splits in families.values())
    assert set(by_split) == {"train", "dev", "test"}
    for counts in by_split.values():
        assert set(counts) == set(data["LABELS"])
        assert len(set(counts.values())) == 1


@pytest.mark.parametrize(
    "corruption", ["duplicate_id", "family_leak", "duplicate_text"]
)
def test_fixture_validation_rejects_leakage(notebooks, corruption):
    data = notebooks["train_classifier"]
    tickets = copy.deepcopy(data["TICKETS"])
    if corruption == "duplicate_id":
        tickets[1]["id"] = tickets[0]["id"]
    elif corruption == "family_leak":
        next(row for row in tickets if row["split"] == "dev")["family"] = tickets[0][
            "family"
        ]
    else:
        tickets[-1]["text"] = "  " + tickets[0]["text"].upper() + "  "
    with pytest.raises(ValueError):
        data["validate_tickets"](tickets)


def test_cpu_baseline_is_reproducible_and_matches_sft_comparison(notebooks):
    cpu = notebooks["train_classifier"]
    texts = [row["text"] for row in cpu["train_tickets"]]
    labels = [row["label"] for row in cpu["train_tickets"]]
    first = cpu["make_classifier"](cpu["BASELINE_CONFIG"]).fit(texts, labels)
    second = cpu["make_classifier"](cpu["BASELINE_CONFIG"]).fit(texts, labels)
    assert cpu["evaluate_classifier"](first, cpu["dev_tickets"]) == cpu["baseline_dev"]
    assert cpu["evaluate_classifier"](first, cpu["test_tickets"]) == cpu[
        "evaluate_classifier"
    ](second, cpu["test_tickets"])
    sft_cpu = notebooks["finetune_classifier"]["fit_cpu_classifier"](
        cpu["train_tickets"]
    )
    held_out = [row["text"] for row in cpu["dev_tickets"] + cpu["test_tickets"]]
    np.testing.assert_array_equal(first.predict(held_out), sft_cpu.predict(held_out))


def test_cpu_vocabulary_excludes_words_seen_only_in_held_out_data(notebooks):
    cpu = notebooks["train_classifier"]
    vectorizer = cpu["baseline_model"].steps[0][1]
    tokenize = vectorizer.build_analyzer()
    training_tokens = {
        token for row in cpu["train_tickets"] for token in tokenize(row["text"])
    }
    held_out_tokens = {
        token
        for row in cpu["dev_tickets"] + cpu["test_tickets"]
        for token in tokenize(row["text"])
    }
    held_out_only = held_out_tokens - training_tokens
    assert held_out_only, (
        "Fixtures must distinguish training-only from leaked vocabulary"
    )
    assert set(vectorizer.vocabulary_) <= training_tokens
    assert not set(vectorizer.vocabulary_) & held_out_only


def test_classifier_confusion_matrix_uses_actual_rows_and_predicted_columns(notebooks):
    cpu = notebooks["train_classifier"]
    labels = cpu["LABELS"]
    rows = [
        {"id": str(i), "text": str(i), "label": label} for i, label in enumerate(labels)
    ]
    model = SimpleNamespace(
        predict=lambda texts: np.array([labels[0], labels[0], labels[1], labels[3]])
    )
    scored = cpu["evaluate_classifier"](model, rows)
    assert scored["metrics"]["accuracy"] == 0.5
    assert scored["metrics"]["macro_f1"] == pytest.approx(5 / 12)
    matrix = scored["confusion_matrix"]
    assert matrix[1]["actual"] == labels[1]
    assert matrix[1][labels[0]] == 1
    assert matrix[2][labels[1]] == 1
    assert sum(sum(row[label] for label in labels) for row in matrix) == len(rows)
    assert [row["correct"] for row in scored["predictions"]] == [
        True,
        False,
        False,
        True,
    ]


def test_sft_invalid_labels_count_as_failures_and_remain_in_confusion_matrix(notebooks):
    sft = notebooks["finetune_classifier"]
    rows = [
        {"id": str(i), "text": str(i), "label": label}
        for i, label in enumerate(sft["LABELS"])
    ]
    scored = sft["score_predictions"](
        rows, [" order_status ", "returns, probably", "product_question", ""]
    )
    assert scored["accuracy"] == scored["macro_f1"] == 0.5
    assert scored["invalid_labels"] == 2
    invalid = scored["confusion_labels"].index("INVALID")
    assert scored["confusion_matrix"][1][invalid] == 1
    assert scored["confusion_matrix"][3][invalid] == 1
    assert sum(map(sum, scored["confusion_matrix"])) == len(rows)
    for predictions in ([], ["returns"]):
        with pytest.raises(ValueError):
            sft["score_predictions"](rows, predictions)


def _result(outcome="order_status", tools=(), answer=None, error=""):
    return {
        "answer": json.dumps(
            {"outcome": outcome, "answer": "Your parcel is on the way."}
        )
        if answer is None
        else answer,
        "tools": list(tools),
        "error": error,
        "conversation_id": "offline-conversation",
    }


def _tool(name, **arguments):
    return {"name": name, "arguments": arguments, "result": {}}


@pytest.mark.parametrize(
    "answer",
    [
        "not json",
        "[]",
        '"answer"',
        '{"outcome": "order_status"}',
        '{"outcome": "order_status", "answer": " "}',
    ],
)
def test_agent_malformed_answers_fail_scoring(notebooks, answer):
    agent = notebooks["build_agent"]
    scored = agent["score_agent"](
        agent["AGENT_SCENARIOS"][0],
        _result(answer=answer, tools=[_tool("lookup_order", order_id="TRAIL-1001")]),
    )
    assert not scored["valid_output"]
    assert not scored["passed"]


@pytest.mark.parametrize(
    "tools",
    [
        [],
        [_tool("lookup_order", order_id="TRAIL-9999")],
        [
            _tool("lookup_order", order_id="TRAIL-1001"),
            _tool("escalate_to_human", reason="unnecessary"),
        ],
    ],
)
def test_agent_requires_the_right_order_lookup_without_unnecessary_escalation(
    notebooks, tools
):
    agent = notebooks["build_agent"]
    assert not agent["score_agent"](agent["AGENT_SCENARIOS"][0], _result(tools=tools))[
        "passed"
    ]


def test_agent_valid_contract_and_required_policy(notebooks):
    agent = notebooks["build_agent"]
    scenarios = {row["id"]: row for row in agent["AGENT_SCENARIOS"]}
    assert agent["score_agent"](
        scenarios["track_known"],
        _result(tools=[_tool("lookup_order", order_id="TRAIL-1001")]),
    )["passed"]
    assert agent["score_agent"](
        scenarios["missing_id"], _result(outcome="ask_order_id")
    )["passed"]
    assert not agent["score_agent"](
        scenarios["missing_id"],
        _result(
            outcome="ask_order_id", tools=[_tool("lookup_order", order_id="TRAIL-1001")]
        ),
    )["passed"]
    for policy, passed in (("returns", True), ("safety", False)):
        tools = [
            _tool("lookup_order", order_id="TRAIL-1002"),
            _tool("lookup_policy", topic=policy),
        ]
        assert (
            agent["score_agent"](
                scenarios["eligible_return"],
                _result(outcome="return_eligible", tools=tools),
            )["passed"]
            is passed
        )
    assert not agent["score_agent"](
        scenarios["track_known"],
        _result(
            tools=[_tool("lookup_order", order_id="TRAIL-1001")], error="step_limit"
        ),
    )["passed"]


class _Trace:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def start_turn(self, **kwargs):
        return self

    def start_llm(self, **kwargs):
        return self

    def start_tool(self, **kwargs):
        return self

    def output(self, value):
        pass

    def record(self, **kwargs):
        pass


def _completion(content=None, calls=()):
    tool_calls = []
    for index, (name, arguments) in enumerate(calls):
        payload = {
            "id": str(index),
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        tool_calls.append(
            SimpleNamespace(
                id=str(index),
                function=SimpleNamespace(name=name, arguments=arguments),
                model_dump=lambda payload=payload: payload,
            )
        )
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=lambda **kwargs: {
            "role": "assistant",
            "content": content,
            "tool_calls": [call.model_dump() for call in tool_calls],
        },
    )
    return SimpleNamespace(
        id="fixture-completion",
        model="fixture-model",
        choices=[
            SimpleNamespace(
                message=message, finish_reason="tool_calls" if calls else "stop"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
    )


def _agent_client(responses):
    calls = []
    iterator = iter(responses)

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        return next(iterator)

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ), calls


def test_agent_step_limit_bounds_requests_tokens_and_tool_dispatch(
    notebooks, monkeypatch
):
    agent = notebooks["build_agent"]
    monkeypatch.setattr(
        agent["tracing"], "start_conversation", lambda **kwargs: _Trace()
    )
    client, requests = _agent_client(
        [_completion(calls=[("lookup_order", '{"order_id":"TRAIL-1001"}')])] * 10
    )
    dispatched = []
    result = agent["run_agent"](
        "track order",
        client,
        "fixture-model",
        "fixture prompt",
        {"lookup_order": lambda **kwargs: dispatched.append(kwargs) or {}},
        max_steps=2,
        max_tokens=11,
    )
    assert result["error"] == "step_limit"
    assert len(requests) == len(dispatched) == len(result["tools"]) == 2
    assert all(request["max_tokens"] == 11 for request in requests)
    assert result["input_tokens"] == 6
    assert result["output_tokens"] == 4


def test_agent_rejects_too_many_tools_before_dispatch(notebooks, monkeypatch):
    agent = notebooks["build_agent"]
    monkeypatch.setattr(
        agent["tracing"], "start_conversation", lambda **kwargs: _Trace()
    )
    client, requests = _agent_client([_completion(calls=[("lookup_order", "{}")] * 4)])
    dispatched = []
    result = agent["run_agent"](
        "track order",
        client,
        "fixture-model",
        "prompt",
        {"lookup_order": lambda **kwargs: dispatched.append(kwargs) or {}},
    )
    assert result["error"] == "too_many_tool_calls"
    assert len(requests) == 1
    assert not dispatched
    assert not result["tools"]


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("unknown_tool", "{}"),
        ("lookup_order", "not json"),
        ("lookup_order", "[]"),
        ("lookup_order", '{"order_id":42}'),
    ],
)
def test_agent_invalid_tool_calls_do_not_dispatch(
    notebooks, monkeypatch, name, arguments
):
    agent = notebooks["build_agent"]
    monkeypatch.setattr(
        agent["tracing"], "start_conversation", lambda **kwargs: _Trace()
    )
    client, requests = _agent_client(
        [
            _completion(calls=[(name, arguments)]),
            _completion(content='{"outcome":"ask_order_id","answer":"Which order?"}'),
        ]
    )
    dispatched = []
    result = agent["run_agent"](
        "track order",
        client,
        "fixture-model",
        "prompt",
        {"lookup_order": lambda **kwargs: dispatched.append(kwargs) or {}},
    )
    assert result["error"] == "invalid_tool_call"
    assert len(requests) == 2
    assert not dispatched
    assert result["tools"][0]["result"] == {"error": "invalid_tool_call"}


def _experiment_services(
    monkeypatch, agent, tmp_path, responses, *, available=True, artifact_error=None
):
    """Replace service boundaries while exercising the real agent and ledger."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture-secret")
    state = SimpleNamespace(
        client_settings=[],
        requests=[],
        before_request=[],
        run_settings=[],
        summary={},
        artifacts=[],
        tracing_projects=[],
    )
    remaining = iter(responses)

    def create(**kwargs):
        state.requests.append(copy.deepcopy(kwargs))
        records = list((tmp_path / ".aria-starter" / "agent").glob("*.json"))
        assert len(records) == 1
        state.before_request.append(json.loads(records[0].read_text()))
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        return response

    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[
                    SimpleNamespace(id="fixture-model" if available else "other-model")
                ]
            )
        ),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )

    def openai_client(**kwargs):
        state.client_settings.append(kwargs)
        return nullcontext(client)

    def log_artifact(artifact):
        state.artifacts.append(artifact)

        def wait():
            if artifact_error is not None:
                raise artifact_error
            return artifact

        return SimpleNamespace(wait=wait)

    def start_run(**kwargs):
        state.run_settings.append(kwargs)
        return nullcontext(
            SimpleNamespace(
                url=f"https://example.test/runs/{kwargs['id']}",
                summary=state.summary,
                log_artifact=log_artifact,
            )
        )

    def start_tracing(project, **kwargs):
        state.tracing_projects.append(project)
        return SimpleNamespace(force_flush=lambda: True)

    monkeypatch.setitem(agent["run_experiment"].__globals__, "OpenAI", openai_client)
    monkeypatch.setattr(agent["wandb"], "init", start_run)
    monkeypatch.setattr(agent["tracing"], "init", start_tracing)
    monkeypatch.setattr(agent["tracing"], "shutdown", lambda: None)
    monkeypatch.setattr(
        agent["tracing"], "start_conversation", lambda **kwargs: _Trace()
    )
    return state


def _agent_settings(examples):
    return {
        "entity": "fixture-team",
        "project": "fixture-student",
        "model": "fixture-model",
        "variant": "baseline",
        "examples": examples,
    }


def _successful_order_responses():
    return [
        _completion(calls=[("lookup_order", '{"order_id":"TRAIL-1001"}')]),
        _completion(
            content=json.dumps(
                {"outcome": "order_status", "answer": "Your daypack has shipped."}
            )
        ),
    ]


def test_unavailable_agent_model_is_blocked_before_paid_calls(
    notebooks, monkeypatch, tmp_path
):
    agent = notebooks["build_agent"]
    state = _experiment_services(monkeypatch, agent, tmp_path, [], available=False)
    result = agent["run_experiment"](
        _agent_settings(2), agent["AGENT_PROMPT"], agent["support_tools"]
    )
    assert result["status"] == "blocked"
    assert "unavailable" in result["message"]
    assert not state.requests
    assert not state.run_settings
    assert not state.tracing_projects
    assert not state.artifacts
    persisted = json.loads(Path(result["local_record"]).read_text())
    assert persisted["run_id"] == result["run_id"]
    assert persisted["status"] == "blocked"
    assert persisted["rows"] == []


@pytest.mark.parametrize(
    "status,error_class",
    [
        (401, "AuthenticationError"),
        (403, "PermissionDeniedError"),
        (429, "RateLimitError"),
    ],
)
def test_inference_failures_preserve_partial_results_and_ids_without_retry(
    notebooks, monkeypatch, tmp_path, status, error_class
):
    import httpx
    import openai

    agent = notebooks["build_agent"]
    error = getattr(openai, error_class)(
        "offline-fixture-secret must never appear in the saved report",
        response=httpx.Response(
            status, request=httpx.Request("POST", "https://example.test/completions")
        ),
        body={"message": "fixture service failure"},
    )
    state = _experiment_services(
        monkeypatch, agent, tmp_path, [*_successful_order_responses(), error]
    )
    result = agent["run_experiment"](
        _agent_settings(3), agent["AGENT_PROMPT"], agent["support_tools"]
    )
    assert result["status"] == "needs_attention"
    assert result["errors"] == [{"stage": "inference", "type": error_class}]
    assert len(result["rows"]) == 1
    assert result["rows"][0]["passed"]
    active = result["active_scenario"]
    assert active["id"] == agent["AGENT_SCENARIOS"][1]["id"]
    assert active["conversation_id"] != result["rows"][0]["conversation_id"]
    assert state.before_request[-1]["active_scenario"] == active
    assert all(record["run_id"] == result["run_id"] for record in state.before_request)
    assert len(state.requests) == 3  # Two successful requests, then one failed request.
    assert len(state.client_settings) == len(state.run_settings) == 1
    assert state.client_settings[0]["max_retries"] == 0
    assert state.summary["completed_examples"] == 1
    assert state.summary["contract_pass_rate"] == pytest.approx(1 / 3)
    persisted_text = Path(result["local_record"]).read_text()
    persisted = json.loads(persisted_text)
    assert persisted["rows"] == result["rows"]
    assert persisted["active_scenario"] == active
    assert persisted["run_id"] == result["run_id"]
    assert persisted["evidence_uploaded"] is True
    assert "offline-fixture-secret" not in persisted_text


def test_artifact_failure_retains_completed_agent_results(
    notebooks, monkeypatch, tmp_path
):
    agent = notebooks["build_agent"]
    state = _experiment_services(
        monkeypatch,
        agent,
        tmp_path,
        _successful_order_responses(),
        artifact_error=ConnectionError("fixture artifact upload interrupted"),
    )
    result = agent["run_experiment"](
        _agent_settings(1), agent["AGENT_PROMPT"], agent["support_tools"]
    )
    assert result["status"] == "needs_attention"
    assert result["errors"] == [{"stage": "evidence_upload", "type": "ConnectionError"}]
    assert result["evidence_uploaded"] is False
    assert result["traces_flushed"] is True
    assert len(result["rows"]) == 1
    assert result["rows"][0]["passed"]
    assert "active_scenario" not in result
    assert state.summary["completed_examples"] == 1
    assert state.summary["contract_pass_rate"] == 1
    assert len(state.requests) == 2
    assert len(state.artifacts) == 1
    persisted = json.loads(Path(result["local_record"]).read_text())
    assert persisted == result


def test_missing_credentials_fail_before_external_calls(notebooks):
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        notebooks["build_agent"]["read_destination"]("team", "student")
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        notebooks["finetune_classifier"]["connection_settings"]()
    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        notebooks["train_classifier"]["upload_comparison"](
            {}, {}, "team", "student", lambda result: None
        )


def test_cpu_upload_creates_distinct_runs_with_evidence_and_models(
    notebooks, monkeypatch
):
    import wandb

    cpu = notebooks["train_classifier"]
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    runs, artifacts, logs, statuses = [], [], [], []

    def init(**kwargs):
        runs.append(kwargs)
        return nullcontext(
            SimpleNamespace(
                url=f"https://example.test/runs/{kwargs['id']}",
                log=logs.append,
                log_artifact=artifacts.append,
            )
        )

    # Leave the actual SDK's ID utilities, Tables, Artifacts, and Settings intact.
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.setattr(wandb, "login", lambda **kwargs: True)
    snapshot = {
        "experiment_id": cpu["experiment_id"],
        "dataset_version": cpu["DATASET_VERSION"],
        "dataset_sha256": cpu["DATASET_SHA256"],
        "dataset": cpu["TICKETS"],
        "source": "# Selected notebook source, without credentials.\n",
        "configs": {
            "baseline": cpu["BASELINE_CONFIG"],
            "candidate": cpu["CANDIDATE_CONFIG"],
        },
        "dev": {"baseline": cpu["baseline_dev"], "candidate": cpu["candidate_dev"]},
    }
    cpu["upload_comparison"](
        snapshot,
        {"baseline": cpu["baseline_model"], "candidate": cpu["candidate_model"]},
        "fixture-team",
        "fixture-student",
        lambda status: statuses.append(copy.deepcopy(status)),
    )
    assert statuses[-1]["state"] == "complete", statuses[-1]
    assert len(runs) == len({run["id"] for run in runs}) == 2
    assert set(statuses[-1]["urls"]) == {"baseline", "candidate"}
    assert Counter(artifact.type for artifact in artifacts) == {
        "experiment": 2,
        "model": 2,
    }
    for artifact in artifacts:
        if artifact.type == "experiment":
            assert set(artifact.manifest.entries) == {
                "dataset.json",
                "experiment.json",
                "train_classifier.py",
            }
    assert sum("dev/predictions" in entry for entry in logs) == 2


@pytest.mark.parametrize("examples", [2, 32])
def test_sft_training_artifact_declares_jsonl_format_and_trajectory_count(
    notebooks, monkeypatch, examples
):
    import wandb

    artifacts = []

    def make_artifact(name, **kwargs):
        artifacts.append({"name": name, **kwargs})
        return SimpleNamespace(new_file=lambda *args, **kwargs: io.StringIO())

    run = SimpleNamespace(
        url="https://example.test/fixture-run",
        summary={},
        log_artifact=lambda artifact: SimpleNamespace(
            wait=lambda: SimpleNamespace(name="sft-inputs:v0")
        ),
    )
    monkeypatch.setattr(wandb, "init", lambda **kwargs: nullcontext(run))
    monkeypatch.setattr(wandb, "Artifact", make_artifact)
    rows = [{"messages": [{"role": "assistant", "content": "returns"}]}] * examples
    record = {"status": "reserved", "settings": {"entity": "fixture-team"}}
    reference = notebooks["finetune_classifier"]["reserve_remote"](
        {"entity": "fixture-team", "project": "fixture-student"},
        "sft-inputs",
        record,
        {"train.jsonl": "\n".join(json.dumps(row) for row in rows) + "\n"},
    )
    assert artifacts[0]["metadata"] == {
        "format": "jsonl",
        "num_trajectories": examples,
    }
    assert reference == "wandb-artifact:///fixture-team/fixture-student/sft-inputs:v0"
    assert isinstance(run.summary["recovery_record_json"], str)
    assert json.loads(run.summary["recovery_record_json"]) == record


def test_sft_recovery_snapshot_round_trips_public_summary_and_replaces_old_status(
    notebooks, monkeypatch, tmp_path
):
    import wandb
    from wandb.apis.public.summary import HTTPSummary

    sft = notebooks["finetune_classifier"]
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    settings = {"entity": "fixture-team", "project": "fixture-student"}
    # Reproduce the service's dotted summaries, including obsolete fields that
    # previously survived a later write of a nested recovery dictionary.
    stored = {"record.status": "reserved", "record.job_id": "obsolete-job"}
    commits = []

    def flatten(values, prefix=""):
        flattened = {}
        for key, value in values.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(flatten(value, path))
            else:
                flattened[path] = copy.deepcopy(value)
        return flattened

    def execute_graphql(query, *, variables):
        # Exercise the pinned SDK's actual JSON serialization before applying
        # the server's flattening behavior at the network boundary.
        summary = json.loads(variables["summaryMetrics"])
        commits.append(summary)
        stored.update(flatten(summary))
        return {"upsertBucket": {"bucket": {"id": "fixture-storage-id"}}}

    def make_run():
        run = SimpleNamespace(
            dir=str(tmp_path),
            storage_id="fixture-storage-id",
            log=lambda values: None,
            log_artifact=lambda artifact: SimpleNamespace(wait=lambda: artifact),
        )
        run.summary = HTTPSummary(
            run, SimpleNamespace(execute_graphql=execute_graphql), copy.deepcopy(stored)
        )
        return run

    monkeypatch.setattr(
        wandb, "Api", lambda **kwargs: SimpleNamespace(run=lambda path: make_run())
    )

    @contextmanager
    def start_run(**kwargs):
        run = make_run()
        yield run
        run.summary.update()

    monkeypatch.setattr(wandb, "init", start_run)
    monkeypatch.setattr(
        wandb,
        "Artifact",
        lambda *args, **kwargs: SimpleNamespace(
            new_file=lambda *args, **kwargs: io.StringIO()
        ),
    )
    training = {
        "experiment_id": "sft-roundtrip",
        "status": "training",
        "job_id": "current-job",
        "model_id": "current-model",
        "settings": {**settings, "training_url": "https://example.test/v1"},
        "partial_predictions": {"checkpoint": "step0", "predictions": ["returns"]},
    }
    sft["publish_record"](settings, "sft-roundtrip", training)
    assert sft["read_remote_record"](settings, "sft-roundtrip") == training
    complete = {
        key: value for key, value in training.items() if key != "partial_predictions"
    }
    complete.update(
        {"status": "complete", "trained_step": 1, "trained_name": "current-model:step1"}
    )
    sft["publish_record"](settings, "sft-roundtrip", complete)
    assert sft["read_remote_record"](settings, "sft-roundtrip") == complete
    complete["results"] = {
        "trained": {
            "accuracy": 1.0,
            "macro_f1": 0.25,
            "invalid_labels": 0,
            "confusion_matrix": [[0, 0], [0, 1]],
            "predictions": [
                {
                    "id": "ticket-1",
                    "text": "Return this item.",
                    "label": "returns",
                    "prediction": "returns",
                    "correct": True,
                }
            ],
        }
    }
    sft["publish_results"](settings, "sft-roundtrip", complete)
    assert sft["read_remote_record"](settings, "sft-roundtrip") == complete
    assert stored["record.status"] == "reserved"  # Legacy fields must not win.
    assert len(commits) == 3
    assert all(isinstance(commit["recovery_record_json"], str) for commit in commits)
    assert "partial_predictions" not in json.loads(stored["recovery_record_json"])


def test_sft_recovery_retries_fresh_reads_when_snapshot_is_temporarily_absent(
    notebooks, monkeypatch
):
    import wandb

    sft = notebooks["finetune_classifier"]
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    record = {
        "status": "complete",
        "job_id": "existing-job",
        "trained_step": 1,
        "settings": {"entity": "fixture-team", "project": "fixture-student"},
    }
    snapshots = iter(
        [
            {},
            {"record.status": "reserved"},
            {"recovery_record_json": json.dumps(record)},
        ]
    )
    clients, reads, sleeps = [], [], []

    def api(**kwargs):
        summary = next(snapshots)

        def run(path):
            reads.append(path)
            return SimpleNamespace(summary=summary)

        client = SimpleNamespace(run=run)
        clients.append(client)
        return client

    monkeypatch.setattr(wandb, "Api", api)
    monkeypatch.setattr(sft["time"], "sleep", sleeps.append)
    assert sft["read_remote_record"](record["settings"], "sft-existing") == record
    assert len(clients) == len({id(client) for client in clients}) == 3
    assert reads == ["fixture-team/fixture-student/sft-existing"] * 3
    assert sleeps == [1, 1]


def test_sft_missing_recovery_snapshot_stops_after_bounded_read_attempts(
    notebooks, monkeypatch
):
    import wandb

    sft = notebooks["finetune_classifier"]
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    clients, sleeps = [], []

    def api(**kwargs):
        # A stale legacy status is insufficient evidence to recover a paid job.
        client = SimpleNamespace(
            run=lambda path: SimpleNamespace(summary={"record.status": "complete"})
        )
        clients.append(client)
        return client

    monkeypatch.setattr(wandb, "Api", api)
    monkeypatch.setattr(sft["time"], "sleep", sleeps.append)
    with pytest.raises(ValueError, match="do not submit another job"):
        sft["read_remote_record"](
            {"entity": "fixture-team", "project": "fixture-student"}, "sft-existing"
        )
    assert len(clients) == 5
    assert sleeps == [1] * 4


def test_sft_recovery_does_not_retry_service_errors_as_missing_snapshots(
    notebooks, monkeypatch
):
    import wandb

    sft = notebooks["finetune_classifier"]
    monkeypatch.setenv("WANDB_API_KEY", "not-a-real-key")
    attempts, sleeps = [], []

    def api(**kwargs):
        attempts.append(True)
        raise ConnectionError("fixture connection failed")

    monkeypatch.setattr(wandb, "Api", api)
    monkeypatch.setattr(sft["time"], "sleep", sleeps.append)
    with pytest.raises(ConnectionError, match="fixture connection failed"):
        sft["read_remote_record"](
            {"entity": "fixture-team", "project": "fixture-student"}, "sft-existing"
        )
    assert len(attempts) == 1
    assert not sleeps


def test_interrupted_sft_preserves_job_and_cannot_resubmit_same_id(
    notebooks, monkeypatch, tmp_path
):
    import art
    import art.serverless.backend
    import art.serverless.client

    sft = notebooks["finetune_classifier"]
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "WANDB_API_KEY": "not-a-real-key",
        "WANDB_ENTITY": "fixture-team",
        "WANDB_PROJECT": "fixture-student",
    }.items():
        monkeypatch.setenv(key, value)
    create_job = AsyncMock(return_value=SimpleNamespace(id="received-job-id"))
    client = SimpleNamespace(
        sft_training_jobs=SimpleNamespace(create=create_job), close=AsyncMock()
    )
    backend = SimpleNamespace(close=AsyncMock())
    model = SimpleNamespace(
        id="model-id",
        run_id="model-run-id",
        register=AsyncMock(),
        _wandb_run=None,
        get_inference_name=lambda step: f"fixture-model:step{step}",
    )
    monkeypatch.setattr(art, "TrainableModel", lambda **kwargs: model)
    monkeypatch.setattr(
        art.serverless.backend, "ServerlessBackend", lambda **kwargs: backend
    )
    monkeypatch.setattr(art.serverless.client, "Client", lambda **kwargs: client)
    remote_ids, uploaded_inputs, published_records = set(), [], []

    def reserve(settings, identity, record, files):
        if identity in remote_ids:
            raise ValueError("Remote run already exists; resume=never refuses reuse")
        remote_ids.add(identity)
        uploaded_inputs.append(files)
        record["run_url"] = "https://example.test/fixture-run"
        return "wandb-artifact:///fixture-team/fixture-student/inputs:v0"

    training_globals = sft["train_experiment"].__globals__
    monkeypatch.setitem(training_globals, "reserve_remote", reserve)
    monkeypatch.setitem(
        training_globals,
        "publish_record",
        lambda settings, identity, record: published_records.append(
            copy.deepcopy(record)
        ),
    )
    monkeypatch.setitem(
        training_globals,
        "wait_for_training",
        AsyncMock(side_effect=ConnectionError("interrupted polling")),
    )
    submission = {
        "experiment_id": "sft-interrupted",
        "mode": "Smoke: 2 tickets",
        "base_model": "fixture-model",
    }
    args = (
        submission,
        sft["TICKETS"],
        sft["DATASET_VERSION"],
        sft["DATASET_SHA256"],
        sft["CLASSIFIER_PROMPT"],
        "source snapshot",
    )
    first = asyncio.run(sft["train_experiment"](*args))
    assert first["status"] == "needs_recovery"
    assert first["job_id"] == "received-job-id"
    assert any(
        record.get("job_id") == "received-job-id" for record in published_records
    )
    assert len(first["training_rows"]) == 2
    assert all(row["split"] == "train" for row in first["training_rows"])
    training_lines = uploaded_inputs[0]["train.jsonl"].splitlines()
    assert len(training_lines) == 2
    assert all(
        json.loads(line)["messages"][-1]["role"] == "assistant"
        for line in training_lines
    )
    second = asyncio.run(sft["train_experiment"](*args))
    assert second["job_id"] == first["job_id"]
    assert create_job.await_count == 1
    # A fresh molab workspace loses its local claim, but the remote reservation
    # must still be consulted before a new training request is issued.
    path = sft["record_path"](first["settings"], submission["experiment_id"])
    path.unlink()
    third = asyncio.run(sft["train_experiment"](*args))
    assert third["status"] == "needs_recovery"
    assert create_job.await_count == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
