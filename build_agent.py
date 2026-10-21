# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo==0.25.2",
#   "coreweave==0.1.0b0",
#   "openai==2.54.0",
#   "wandb==0.31.0",
#   "python-dotenv==1.2.4",
# ]
# ///
# Support-desk teaching flow adapted from W&B examples; see sources.md.
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import inspect
    import json
    import os
    import re
    import textwrap
    import uuid
    from pathlib import Path

    import marimo as mo
    import wandb
    from coreweave.forge.agentlens import tracing
    from dotenv import load_dotenv
    from openai import OpenAI

    _environment_loaded = load_dotenv(Path.cwd() / ".env", override=False)
    return OpenAI, Path, inspect, json, mo, os, re, textwrap, tracing, uuid, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Build the outdoor shop's support assistant

    You are helping **Trail Supply, a fictional outdoor shop**, answer customer messages. A customer might ask
    “Where is order TRAIL-1001?” The assistant should look up that exact order and answer from the record.
    If the customer says only “Where is my order?”, it should ask for the order ID before claiming to know its status.
    Your first experiment is to make that missing-ID behavior more reliable.

    This is an **agent**: a language model can choose a supplied Python function, read its result, and then answer.
    Those functions are called **tools**. The model already knows how to generate text; this notebook does not train
    its weights. You change the **prompt** (instructions sent to the model) or a tool's Python behavior instead.

    **What you will make:** a baseline run, a changed-prompt run on the same questions, and an explanation of one
    behavioral difference. The supplied orders, policies, questions, and expected outcomes make this runnable
    without a business, dataset, or existing agent project. All actions are simulated.

    **Before starting:** you need basic Python, a W&B team/project, your API key, Forge Inference access and credits,
    and ARIA access. Each **Run agent experiment** submission spends inference credits; opening or editing does not.

    ### Open and run this notebook

    If you are in molab, fork this notebook into your own workspace before editing.
    If you downloaded this `.py` file, open a terminal in its folder and run the command below.
    [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
    are available if your terminal says `uvx` is not found. The first launch installs the pinned packages.

    ```sh
    uvx marimo edit --sandbox build_agent.py
    ```

    This file includes all its data and helpers; you do not need another notebook or a data download.
    If results or forms are missing, open marimo's command palette and choose **Run all stale cells**
    (or **Re-run all cells**). That initializes local Python and the forms. Separate submit buttons control uploads and paid work.

    A **cell** is one editable block in the notebook. **Edit here** identifies the blocks for experiments.
    Click the code, make your change, and run that cell with its play button; dependent results update.
    **Run this** identifies an action or its output. **Supplied implementation** contains provided helpers:
    expand a collapsed cell with **Show code** in the cell menu to read the Python.
    You do not need to edit those helpers for the first exercise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Connect your own project

    A **W&B project** is the folder on the website that holds your experiments. A **run** is one saved experiment
    inside it. An **artifact** is a versioned file bundle, such as the data, predictions, and Python used for that run.
    ARIA can investigate evidence saved in the project; it does not read your unsaved molab edits.

    Ask your facilitator for a W&B **team name**, a **project name just for you**, and access to ARIA.
    Use your own W&B API key from [W&B authorization](https://wandb.ai/authorize).
    In molab, open the notebook's **Secrets** panel and add these three names:

    | Secret name | What to put in it |
    | --- | --- |
    | `WANDB_ENTITY` | Your facilitator's team name |
    | `WANDB_PROJECT` | Your assigned project name |
    | `WANDB_API_KEY` | Your own API key |

    Locally, create a file named `.env` beside the notebook with `NAME=value` on each line for those three names.
    Keep that file out of Git and never paste its contents into a notebook cell, ARIA prompt, or screenshot.
    Restart the notebook after adding credentials. The project must allow your account to log runs;
    ARIA access also needs to be enabled by your organization. If a page says you lack access, ask your facilitator
    about that project rather than creating a different personal project.

    [ARIA setup requirements](https://docs.coreweave.com/products/aria) explain the account prerequisites.

    ## Follow one customer question through the system

    For the supplied “Where is order TRAIL-1001?” question, the expected path is:

    1. Python sends the prompt and customer question to **Forge Inference**, which runs the language model.
    2. The model requests `lookup_order(order_id="TRAIL-1001")`.
    3. Python runs that tool against the supplied dictionary. It returns the shipped Ridge daypack and its ETA.
    4. The model sees the tool result and writes an answer. Python checks its structured outcome and tool behavior.
    5. W&B stores the experiment's settings and scores. **Agent Lens** stores the conversation and its model/tool steps.

    Calling the model to get an answer is **inference**. A **trace** is the recorded sequence of steps for a question;
    it helps you see why an answer happened. ARIA investigates that recorded evidence after the run.

    ### What the assistant is allowed to know and do

    | Supplied tool | Input | Result |
    | --- | --- | --- |
    | `lookup_order` | An order ID supplied by the customer | That order's record, or an unknown-order error |
    | `lookup_policy` | A policy topic such as `returns` | The shop's written policy |
    | `escalate_to_human` | A reason for the handoff | A simulated escalation receipt; no person is contacted |

    Order lookup is not a search for any plausible order: it must use the customer's ID.
    Return decisions need the order and return policy. Safety or explicit requests for human help should not be
    answered with an invented resolution. For the first experiment, concentrate on missing IDs; keep other behavior fixed.

    The first two scenarios are a known order and a missing order ID. **Questions = 2** runs those two in that order.
    Using 6 runs all supplied scenarios, including returns and safety checks. Choose the same count for baseline and candidate.
    A **scenario** is a prepared customer question plus an answer key describing the behavior to check.
    """)
    return


@app.cell(hide_code=True)
def _(AGENT_SCENARIOS, ORDERS, STORE_POLICY, mo):
    mo.vstack(
        [
            mo.md(
                "### Inspect what is supplied before calling a model\nThese records are already in this file. Read the first two questions and open the orders/policy if you want to check what a correct answer can claim."
            ),
            mo.ui.table(
                [
                    {
                        "scenario": row["id"],
                        "customer_question": row["message"],
                        "expected_outcome": row["expected_outcome"],
                        "required_tools": row["expected_tools"],
                    }
                    for row in AGENT_SCENARIOS[:2]
                ],
                selection=None,
            ),
            mo.accordion(
                {
                    "Fictional order records": mo.ui.table(
                        [{"order_id": key, **value} for key, value in ORDERS.items()],
                        selection=None,
                    ),
                    "Fictional shop policies": mo.json(STORE_POLICY),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Edit here: assistant prompt

    Run the supplied prompt as `baseline` first. For your first edit, focus on **missing-order handling**:
    when no order ID is supplied, should the assistant ask for it before trying a lookup?
    Keep inference settings and the number of scenarios fixed while testing that prompt change.

    The response must be a JSON object with `outcome` and `answer`. `outcome` is one of the four values below;
    `answer` is the customer-facing reply. This **response contract** lets the scorer compare structured behavior.
    For example, `{"outcome": "ask_order_id", "answer": "What is your order ID?"}`
    says both what happened and what the customer sees. JSON is just a structured way to store those two fields.
    `order_status` means a tracking answer, `ask_order_id` asks for the missing identifier,
    `return_eligible` says the return meets the policy, and `human_escalation` hands off to a person.
    A valid JSON answer can still mislead a customer, so read the answer as well as its score.
    """)
    return


@app.cell
def _():
    # Edit this cell for your next experiment. Keep the output contract intact.
    AGENT_PROMPT = """You help customers of Trail Supply, a fictional outdoor shop.
    Use the available tools to answer questions. Explain the next step concisely.
    Your final response must be a JSON object with an `outcome` and an `answer`.
    Outcomes: order_status, ask_order_id, return_eligible, human_escalation.
    `answer` is the message to the customer. Do not wrap JSON in a code fence.
    """
    return (AGENT_PROMPT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Edit here: inference settings

    `model` selects which hosted language model answers; it must be available to your Forge account.
    `temperature` (0–1) controls how much the model varies its word choices;
    keep it at 0 for the first comparison. `max_steps` (1–4) bounds model calls per question, including tool requests.
    `max_tokens` (64–512) caps each response. A token is a small piece of text, often part of a word.
    Too small a limit can cut off the JSON answer before it finishes. A model/tool round trip needs another step
    for the model to read the tool result and answer; `max_steps=1` often stops before that answer.
    These limits bound one submission, not repeated experiments. Keep them fixed for the first prompt experiment.
    """)
    return


@app.cell
def _():
    # Edit here: these values are frozen into each explicitly submitted experiment.
    AGENT_CONFIG = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "temperature": 0,
        "max_steps": 4,
        "max_tokens": 512,
    }
    return (AGENT_CONFIG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Your first baseline run

    Leave `AGENT_PROMPT` and `AGENT_CONFIG` as supplied for now. In the form below:

    1. Enter your team and assigned project. These must match the project your facilitator enabled.
    2. Keep **Experiment name** as `baseline` and **Questions** as `2` for the first small run.
    3. Click **Run agent experiment** once. Wait for its status and two result rows.
    4. Keep the W&B run link. A status of `finished` means the requests completed, not that the answers passed.

    Every submission is a new paid experiment. If a request fails, inspect the saved run ID and error first;
    clicking the button again is a deliberate new attempt. Missing model access or credits needs your facilitator's help.
    """)
    return


@app.cell(hide_code=True)
def _(
    AGENT_CONFIG,
    AGENT_PROMPT,
    AGENT_SCENARIOS,
    mo,
    os,
    run_experiment,
    set_experiments,
    set_status,
    support_tools,
):
    def submit_agent_experiment(values):
        if values is None:
            return
        # A callback snapshots the current code/config. Reactive edits never call it.
        try:
            report = run_experiment(
                dict(values), AGENT_PROMPT, dict(support_tools), dict(AGENT_CONFIG)
            )
            set_experiments(lambda previous: [*previous, report])
            evidence = (
                "Evidence uploaded to W&B."
                if report["evidence_uploaded"]
                else "Evidence upload is incomplete. Download the results and inspect the run before submitting again."
            )
            set_status(f"{report['status']}: {report['run_id']}. {evidence}")
        except ValueError as error:
            set_status(str(error))
        except Exception as error:
            set_status(
                f"{type(error).__name__}: request failed. Check access and credits in Forge before submitting again."
            )

    agent_form = (
        mo.md("""
    ## 3. Run this: baseline or candidate

    {entity} {project}

    {variant} {examples}

    Each question uses the step and token limits in `AGENT_CONFIG` above. Give the first run the name `baseline`, then edit code and name the
    next run `candidate`. Submitting again intentionally starts a new experiment.
    """)
        .batch(
            entity=mo.ui.text(value=os.getenv("WANDB_ENTITY", ""), label="W&B team"),
            project=mo.ui.text(
                value=os.getenv("WANDB_PROJECT", "aria-support-desk"), label="Project"
            ),
            variant=mo.ui.text(value="baseline", label="Experiment name"),
            examples=mo.ui.slider(1, len(AGENT_SCENARIOS), value=2, label="Questions"),
        )
        .form(
            submit_button_label="Run agent experiment",
            on_change=submit_agent_experiment,
        )
    )
    mo.vstack([mo.md(f"**Selected inference settings:** `{AGENT_CONFIG}`"), agent_form])
    return agent_form, submit_agent_experiment


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Inspect and ask ARIA

    A scenario specifies required tools and an expected outcome. A **tool contract** checks that the assistant
    used the right arguments and avoided unsupported actions, such as inventing an order ID or escalating unnecessarily.
    A pass requires all checks for that scenario. Read each tool's arguments and result in **Agent Lens**, then
    read the customer answer below. The automated score cannot verify every factual claim.

    If a run is incomplete, keep its ID, inspect access/credits and saved errors, and resolve the failure with your
    facilitator before discussing model quality. Re-submitting the form intentionally spends credits again.
    """)
    return


@app.cell(hide_code=True)
def _(get_experiments, get_status, json, mo):
    _runs = get_experiments()
    _sections = [mo.md(get_status())]
    for _run in _runs:
        _sections.append(
            mo.md(
                f"### {_run['variant']}\n[W&B run]({_run['run_url']}) · [Agent Lens]({_run['agent_lens_url']}) · [Weave Agents]({_run['project_url']}/weave/agents)"
            )
        )
        if _run.get("message"):
            _sections.append(mo.callout(_run["message"], kind="warn"))
        if _run.get("errors"):
            _sections.append(
                mo.callout(
                    f"Keep run ID `{_run['run_id']}`. Status: `{json.dumps(_run['errors'])}`. Local record: `{_run['local_record']}`. No automatic retry was submitted.",
                    kind="warn",
                )
            )
        _sections.append(
            mo.md(
                f"**Completed questions:** {len(_run['rows'])}/{_run['config']['examples']}. "
                f"**Passed contracts:** {sum(row['passed'] for row in _run['rows'])}/{_run['config']['examples']}. "
                "Check the table for which questions failed; unfinished questions are not successful answers."
            )
        )
        _sections.append(mo.ui.table(_run["rows"], selection=None))
    if _runs:
        _latest = _runs[-1]
        _baseline = _runs[0]
        _prompt = (
            f"I am building the fictional Trail Supply support assistant. Compare {_baseline['run_url']} "
            f"with {_latest['run_url']}. Read their evaluation artifacts, config, and experiment.py. "
            f"Saved baseline settings: {json.dumps(_baseline['config'])}. "
            f"Saved variant settings: {json.dumps(_latest['config'])}. "
            "Investigate one failed tool/outcome contract. For two runs, compare that same scenario and regressions. "
            "Explain it using the logged evidence. Suggest one small Python or prompt edit and a test. "
            "The automated score does not judge every factual claim; also inspect the customer answers. "
            "Help me state a one-sentence hypothesis for an edit to AGENT_PROMPT, holding AGENT_CONFIG fixed. "
            "If this is my comparison, does the evidence support that hypothesis? "
            "Do not launch another paid experiment. I will edit AGENT_PROMPT and explicitly submit a candidate in marimo."
        )
        if _latest["status"] != "finished" or not _latest["evidence_uploaded"]:
            _prompt = (
                f"Help me diagnose incomplete experiment {_latest['run_url']} using its saved record. "
                f"Status: {_latest['status']}; errors: {json.dumps(_latest['errors'])}. "
                "Check access, credits, model availability, and artifact upload before discussing model quality. "
                "Do not retry paid work. I will inspect the existing run and saved identifiers with my facilitator."
            )
        _sections += [
            mo.md(
                f"### Copy the ARIA handoff\nOpen [this project]({_latest['project_url']}), click **Ask ARIA**, and paste:"
            ),
            mo.ui.text_area(value=_prompt, full_width=True),
            mo.download(
                json.dumps(_runs, indent=2),
                filename="agent-experiments.json",
                label="Download results",
            ),
        ]
    mo.vstack(_sections)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Read the result and open the dashboards

    Each row below the run link belongs to one question. `scenario` identifies the test, `expected` is the
    answer key, `outcome` is what the model returned, and `passed` says whether all automated checks passed.
    Open the row's `tools` data to inspect names, arguments, and returned records; read `answer` for the customer-facing text.
    An `error` such as `step_limit` means the agent did not complete normally within its bounds.

    Use the **W&B run** link for saved settings and the evidence artifact. Its `results.json` contains all scored rows;
    `config.json` and `experiment.py` show the exact choices for that run. `contract_pass_rate` is the fraction of
    submitted scenarios that passed every check, not a rating of the prose quality.

    Use the **Agent Lens** link for the actual conversations. Match a row's `conversation_id` with the conversation
    in the dashboard's **Conversations** view. Open it and read the **Thread** for the conversation,
    then **Spans** for individual model/tool operations. A span is one timed operation in the trace.
    [Agent Lens navigation guide](https://docs.coreweave.com/products/agent-lens/conversations/view-activity)
    describes these views. Read the customer message, tool arguments/results, and final answer.
    For a missing ID, a lookup using an invented ID is the specific behavior to investigate.

    The notebook uses the Forge SDK's Agent Lens tracing. These conversations are not ordinary Weave Calls or
    `weave.Evaluation` runs; an empty Weave Calls/Evaluations page does not mean this agent failed to log.
    The **Weave Agents** link opens these conversations within the W&B project. For this lab, compare scores in W&B and inspect behavior in Agent Lens or that Agents view. A separate Weave integration is not required.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Ask ARIA, make a prompt edit, and compare

    **Investigate the baseline.** Read the missing-ID row and its Agent Lens conversation. Was an order lookup attempted?
    Did the assistant invent an ID, ask the customer for it, or escalate? Open **Ask ARIA** from the linked W&B project
    and paste the handoff above. Add: “Focus on the missing-ID scenario. Show the tool arguments that explain its result.
    Explain the failure in plain language before proposing a prompt change.” If it already passed, say so: test whether
    an explicit instruction preserves that behavior and avoids regressions elsewhere.

    **Make a prediction before editing:** “Explicitly asking for a missing order ID will prevent invented order lookups.”
    Return to **AGENT_PROMPT** in section 1 and add this sentence inside its triple-quoted text:

    ```text
    For order tracking, if the customer has not supplied an order ID, do not call tools or guess an ID. Ask for the order ID and use the ask_order_id outcome.
    ```

    Keep the existing JSON instructions. Run the edited cell. This changes the instructions for the next submission;
    it does not call Forge by itself. Keep `AGENT_CONFIG` and **Questions** unchanged.
    Return to section 3, name the next experiment `candidate-missing-id`, and click **Run agent experiment** once.

    **Compare the two runs.** Find the missing-ID scenario in both result tables. Check the outcome and tool calls,
    then check the known-order question still works. If you used all six scenarios, inspect returns and safety too.
    A better result on one question can cause a worse result on another; that is a **regression**.

    Use the updated handoff in ARIA: “My hypothesis was __. Compare the baseline and candidate on the same questions.
    Did the missing-ID behavior change? What regressed? Cite the conversations and saved prompt, and tell me what
    this small experiment can and cannot establish.” Read the customer answers yourself as well.

    You are done with the first exercise when you can explain one behavior, the instruction you changed,
    and the result. Save the edited notebook and both run links. Only then try the optional tools below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Extend the exercise

    - Ask ARIA why an unknown order should trigger a handoff rather than a made-up ETA.
    - Add a tool over `PRODUCTS`, then add a scenario and its expected behavior.
    - Change the prompt to check return eligibility before promising a refund.
    - Compare failures on the same scenarios; adding easier questions can inflate the average.

    Record what you changed, what ARIA helped explain, and what remained confusing.
    Save or download your edited notebook. After a restart, use your W&B run links
    to recover the durable evidence; the comparison list above belongs to this session.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Edit here: optional tools

    Try this only after a prompt comparison. Each tool has three connected parts:
    its **Python implementation** returns data, its **schema** tells the model the name and arguments,
    and an **evaluation scenario** specifies the behavior you expect. The registry maps names to implementations.
    If you add a product lookup, update all three and the registry. The supplied scenarios are in the collapsed fixtures;
    keep original scenarios for regression checks and distinguish new ones in your report.

    The functions below simulate actions. An unknown order returns an error the assistant must handle;
    escalation only returns a fictional receipt. Edit the implementation and schema cells together when changing an interface.
    """)
    return


@app.cell
def _(ORDERS, STORE_POLICY):
    def lookup_order(order_id: str) -> dict:
        """Find a supplied order, never invent one."""
        key = order_id.strip().upper()
        if key not in ORDERS:
            return {"error": "order_not_found", "order_id": key}
        return {"order_id": key, **ORDERS[key]}

    def lookup_policy(topic: str) -> dict:
        """Read returns, missing_order, or safety policy."""
        if topic not in STORE_POLICY:
            return {"error": "unknown_policy", "available": list(STORE_POLICY)}
        return {"topic": topic, "policy": STORE_POLICY[topic]}

    def escalate_to_human(reason: str) -> dict:
        """Simulate a handoff. This does not send a message or open a real ticket."""
        return {"simulated": True, "outcome": "human_escalation", "reason": reason}

    support_tools = {
        "lookup_order": lookup_order,
        "lookup_policy": lookup_policy,
        "escalate_to_human": escalate_to_human,
    }
    return escalate_to_human, lookup_order, lookup_policy, support_tools


@app.cell
def _():
    TOOL_SCHEMAS = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {argument: {"type": "string"}},
                    "required": [argument],
                    "additionalProperties": False,
                },
            },
        }
        for name, argument, description in [
            ("lookup_order", "order_id", "Look up an order such as TRAIL-1001."),
            (
                "lookup_policy",
                "topic",
                "Read returns, missing_order, or safety policy.",
            ),
            ("escalate_to_human", "reason", "Simulate handing a problem to a human."),
        ]
    ]
    return (TOOL_SCHEMAS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Supplied implementation

    Expand a collapsed cell using its **Show code** control (the code icon or cell menu in the editor).
    These cells contain the fixtures, scoring, uploads, and recovery helpers used above.
    You can inspect every function. Keep fixtures and scoring fixed during your first comparison.
    """)
    return


@app.cell(hide_code=True)
def _():
    # START SHARED FIXTURES
    # All people, tickets, products, orders, and policies below are fictional.
    # Split scenario families first; keep each family's paraphrases in one split.
    DATASET_VERSION = "outdoor-support-v1"
    LABELS = ["order_status", "returns", "product_question", "human_escalation"]
    TICKET_FAMILIES = [
        (
            "train",
            "tracking_link",
            "order_status",
            [
                "Where is my order? Please send a tracking link.",
                "Can you track my parcel and tell me where it is?",
            ],
        ),
        (
            "train",
            "shipping_delay",
            "order_status",
            [
                "My delivery is late. Has the package shipped?",
                "The parcel should have arrived yesterday. What is its shipping status?",
            ],
        ),
        (
            "train",
            "dispatch_date",
            "order_status",
            [
                "When will my tent order leave your warehouse?",
                "Has my purchase been dispatched yet?",
            ],
        ),
        (
            "train",
            "delivery_estimate",
            "order_status",
            [
                "What is the estimated delivery date for my backpack?",
                "When should my order arrive at my home?",
            ],
        ),
        (
            "train",
            "wrong_size",
            "returns",
            [
                "These boots are too small. How do I return them?",
                "The jacket does not fit. Can I exchange it for a larger size?",
            ],
        ),
        (
            "train",
            "unwanted_item",
            "returns",
            [
                "I changed my mind about the tent. Can I get a refund?",
                "I no longer need this backpack. How can I send it back?",
            ],
        ),
        (
            "train",
            "return_label",
            "returns",
            [
                "Please send a prepaid label for my return.",
                "Where do I print the shipping label to return my order?",
            ],
        ),
        (
            "train",
            "refund_status",
            "returns",
            [
                "You received my returned boots. When will the refund arrive?",
                "Has the refund for the jacket I sent back been processed?",
            ],
        ),
        (
            "train",
            "waterproof_material",
            "product_question",
            [
                "Is the rain jacket waterproof or just water resistant?",
                "What material keeps water out of this shell jacket?",
            ],
        ),
        (
            "train",
            "pack_capacity",
            "product_question",
            [
                "How many liters does the daypack hold?",
                "What is the storage capacity of the hiking backpack?",
            ],
        ),
        (
            "train",
            "tent_weight",
            "product_question",
            [
                "How much does the two person tent weigh?",
                "What is the packed weight of your trail tent?",
            ],
        ),
        (
            "train",
            "boot_sizing",
            "product_question",
            [
                "Do your hiking boots run true to size?",
                "Where is the size guide for boots before I buy?",
            ],
        ),
        (
            "train",
            "double_charge",
            "human_escalation",
            [
                "My card was charged twice for one order. I need someone to fix it.",
                "There are duplicate charges on my bank statement for this purchase.",
            ],
        ),
        (
            "train",
            "unsafe_equipment",
            "human_escalation",
            [
                "A stove caught fire and injured me. Please connect me with a person.",
                "Your climbing gear broke during use. I need a safety specialist.",
            ],
        ),
        (
            "train",
            "account_takeover",
            "human_escalation",
            [
                "Someone accessed my account and placed an order I did not make.",
                "My account was hacked. Please help me secure it.",
            ],
        ),
        (
            "train",
            "manager_request",
            "human_escalation",
            [
                "I want to speak to a human manager about a complaint.",
                "Please connect me to a person. The automatic replies have not helped.",
            ],
        ),
        (
            "dev",
            "split_shipment",
            "order_status",
            [
                "One box arrived but the rest of my order is missing. Is another parcel coming?",
                "My order has two packages. Where is the second delivery?",
            ],
        ),
        (
            "dev",
            "pickup_progress",
            "order_status",
            [
                "Is my order ready for collection at the shop?",
                "Has my click and collect purchase reached the pickup counter?",
            ],
        ),
        (
            "dev",
            "gift_exchange",
            "returns",
            [
                "I received this flask as a gift and would like to swap it.",
                "Can I exchange a present without the original payment card?",
            ],
        ),
        (
            "dev",
            "opened_packaging",
            "returns",
            [
                "I opened the packaging but did not use the lamp. Can I return it?",
                "Can I send this unused item back if the box is already open?",
            ],
        ),
        (
            "dev",
            "sleeping_temperature",
            "product_question",
            [
                "What is the sleeping bag temperature rating for cold weather?",
                "Will this sleeping bag keep me warm at freezing temperatures?",
            ],
        ),
        (
            "dev",
            "bottle_cleaning",
            "product_question",
            [
                "Can the insulated bottle go in a dishwasher?",
                "What is the recommended way to clean the water flask?",
            ],
        ),
        (
            "dev",
            "private_data",
            "human_escalation",
            [
                "You emailed my address to a stranger. I need your privacy team.",
                "My personal details appeared in another customer's receipt. Please investigate.",
            ],
        ),
        (
            "dev",
            "accessibility_help",
            "human_escalation",
            [
                "The checkout does not work with my screen reader. I need a person to assist.",
                "I cannot use the payment form with assistive technology. Can someone help me order?",
            ],
        ),
        (
            "test",
            "courier_exception",
            "order_status",
            [
                "The courier says delivery attempted but nobody knocked. Where is my parcel now?",
                "Tracking shows a failed delivery attempt. What happens to my shipment next?",
            ],
        ),
        (
            "test",
            "preorder_progress",
            "order_status",
            [
                "Has my preordered lantern been dispatched now that it is available?",
                "Can you check the shipping progress of the backpack I reserved last month?",
            ],
        ),
        (
            "test",
            "missing_return_receipt",
            "returns",
            [
                "I lost my receipt. Can I still return these unused gloves?",
                "How do I get a refund without the paper proof of purchase?",
            ],
        ),
        (
            "test",
            "damaged_arrival",
            "returns",
            [
                "The mug arrived cracked. How can I send it back for a replacement?",
                "My new hiking pole was bent in the box. Can I exchange it?",
            ],
        ),
        (
            "test",
            "battery_compatibility",
            "product_question",
            [
                "Which batteries fit the headlamp?",
                "Can I use rechargeable cells in this camping light?",
            ],
        ),
        (
            "test",
            "tent_dimensions",
            "product_question",
            [
                "How long is the tent floor? I am tall and want to fit inside.",
                "What are the inside dimensions of the two person shelter?",
            ],
        ),
        (
            "test",
            "suspected_fraud",
            "human_escalation",
            [
                "I never bought from you but there is a charge on my card. Please investigate fraud.",
                "There is an unauthorized purchase from your shop on my bank account.",
            ],
        ),
        (
            "test",
            "legal_notice",
            "human_escalation",
            [
                "I need to send a formal legal notice to your company. Who handles this?",
                "Please direct my lawyer to someone who can handle a legal complaint.",
            ],
        ),
    ]
    TICKETS = [
        {
            "id": f"{family}-{index + 1}",
            "family": family,
            "split": split,
            "text": text,
            "label": label,
        }
        for split, family, label, texts in TICKET_FAMILIES
        for index, text in enumerate(texts)
    ]
    STORE_POLICY = {
        "returns": "Unused items can be returned within 30 days of delivery. Used items or older purchases require human review. Never promise a refund before eligibility is checked.",
        "missing_order": "Ask for an order ID if it is missing. If a supplied ID is not found, offer human escalation; do not invent a delivery status.",
        "safety": "Equipment injuries, suspected fraud, and privacy complaints must be escalated to a human. Escalation here is simulated and contacts nobody.",
    }
    ORDERS = {
        "TRAIL-1001": {
            "item": "Ridge daypack",
            "status": "shipped",
            "delivery_eta": "in 2 days",
            "days_since_delivery": None,
            "used": False,
        },
        "TRAIL-1002": {
            "item": "Cloud rain shell",
            "status": "delivered",
            "delivery_eta": None,
            "days_since_delivery": 10,
            "used": False,
        },
        "TRAIL-1003": {
            "item": "Trail boots",
            "status": "delivered",
            "delivery_eta": None,
            "days_since_delivery": 45,
            "used": True,
        },
    }
    PRODUCTS = {
        "Ridge daypack": {"capacity_liters": 22, "weight_grams": 650},
        "Cloud rain shell": {"waterproof": True, "care": "Cold gentle wash; air dry."},
        "Trail boots": {
            "sizes": "EU 36-46",
            "care": "Brush dirt off; air dry away from heat.",
        },
    }
    AGENT_SCENARIOS = [
        {
            "id": "track_known",
            "message": "Where is order TRAIL-1001?",
            "expected_tools": ["lookup_order"],
            "expected_outcome": "order_status",
        },
        {
            "id": "missing_id",
            "message": "Where is my package?",
            "expected_tools": [],
            "expected_outcome": "ask_order_id",
        },
        {
            "id": "unknown_order",
            "message": "Please track order TRAIL-9999.",
            "expected_tools": ["lookup_order", "escalate_to_human"],
            "expected_outcome": "human_escalation",
        },
        {
            "id": "eligible_return",
            "message": "Can I return my unused jacket from order TRAIL-1002?",
            "expected_tools": ["lookup_order", "lookup_policy"],
            "expected_outcome": "return_eligible",
        },
        {
            "id": "old_used_return",
            "message": "Refund the boots from TRAIL-1003. I used them and they arrived 45 days ago.",
            "expected_tools": ["lookup_order", "lookup_policy", "escalate_to_human"],
            "expected_outcome": "human_escalation",
        },
        {
            "id": "safety_complaint",
            "message": "A stove from your shop caught fire and injured me. Please help.",
            "expected_tools": ["escalate_to_human"],
            "expected_outcome": "human_escalation",
        },
    ]
    import hashlib as _fixture_hashlib
    import json as _fixture_json

    DATASET_SHA256 = _fixture_hashlib.sha256(
        _fixture_json.dumps(
            [TICKETS, STORE_POLICY, ORDERS, PRODUCTS, AGENT_SCENARIOS], sort_keys=True
        ).encode()
    ).hexdigest()
    # END SHARED FIXTURES
    return (
        AGENT_SCENARIOS,
        DATASET_SHA256,
        DATASET_VERSION,
        LABELS,
        ORDERS,
        PRODUCTS,
        STORE_POLICY,
        TICKET_FAMILIES,
        TICKETS,
    )


@app.cell(hide_code=True)
def _(json, re):
    def score_agent(scenario: dict, result: dict) -> dict:
        """Score observable tool/outcome contracts, not free-text answer quality."""
        try:
            final = json.loads(result["answer"])
        except (ValueError, TypeError):
            final = {}
        valid = (
            isinstance(final, dict)
            and isinstance(final.get("answer"), str)
            and bool(final["answer"].strip())
        )
        calls = result["tools"]
        names = [call["name"] for call in calls]
        required = set(scenario["expected_tools"])
        tools_ok = required.issubset(names)
        if scenario["expected_outcome"] != "human_escalation":
            tools_ok = tools_ok and "escalate_to_human" not in names
        order_ids = re.findall(r"TRAIL-\d+", scenario["message"])
        seen_calls = set()
        for call in calls:
            arguments = call.get("arguments", {})
            signature = (call["name"], json.dumps(arguments, sort_keys=True))
            if signature in seen_calls:
                tools_ok = False
            seen_calls.add(signature)
            if not isinstance(arguments, dict):
                tools_ok = False
                continue
            if call["name"] == "lookup_order":
                order_id = arguments.get("order_id")
                tools_ok = (
                    tools_ok
                    and isinstance(order_id, str)
                    and order_id.strip().upper() in order_ids
                )
            elif call["name"] == "lookup_policy":
                tools_ok = tools_ok and arguments.get("topic") in {
                    "returns",
                    "missing_order",
                    "safety",
                }
            elif call["name"] == "escalate_to_human":
                reason = arguments.get("reason")
                tools_ok = tools_ok and isinstance(reason, str) and bool(reason.strip())
                tools_ok = tools_ok and names.count("escalate_to_human") == 1
            else:
                tools_ok = False
            # An unknown supplied order is an expected lookup result. Malformed
            # calls and unknown policies must fail even after a successful retry.
            output = call.get("result", {})
            if not isinstance(output, dict) or output.get("error") not in {
                None,
                "order_not_found",
            }:
                tools_ok = False
        if "lookup_order" in required:
            tools_ok = tools_ok and any(
                call["name"] == "lookup_order"
                and isinstance(call["arguments"], dict)
                and isinstance(call["arguments"].get("order_id"), str)
                and call["arguments"]["order_id"].strip().upper() in order_ids
                for call in calls
            )
        if scenario["expected_outcome"] == "ask_order_id":
            tools_ok = tools_ok and not calls
        if "lookup_policy" in required:
            tools_ok = tools_ok and any(
                call["name"] == "lookup_policy"
                and isinstance(call["arguments"], dict)
                and call["arguments"].get("topic") == "returns"
                for call in calls
            )
        outcome = (
            final.get("outcome", "invalid") if isinstance(final, dict) else "invalid"
        )
        return {
            "scenario": scenario["id"],
            "question": scenario["message"],
            "expected": scenario["expected_outcome"],
            "outcome": outcome,
            "tools_ok": tools_ok,
            "valid_output": bool(valid),
            "passed": bool(
                valid
                and tools_ok
                and outcome == scenario["expected_outcome"]
                and not result["error"]
            ),
            "answer": result["answer"],
            "error": result["error"],
            "tools": calls,
            "conversation_id": result["conversation_id"],
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        }

    return (score_agent,)


@app.cell(hide_code=True)
def _(TOOL_SCHEMAS, json, tracing, uuid):
    def run_agent(
        message,
        client,
        model,
        prompt,
        tools,
        *,
        temperature=0,
        max_steps=4,
        max_tokens=512,
        conversation_id=None,
    ):
        """A bounded tool loop. Edit the prompt/tools without hiding agent code."""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ]
        result = {
            "answer": "",
            "tools": [],
            "error": "",
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "input_tokens": 0,
            "output_tokens": 0,
        }

        def trace_messages(history):
            return [
                tracing.Message(
                    role=item["role"],
                    content=item.get("content") or "",
                    tool_call_id=item.get("tool_call_id", ""),
                    parts=[
                        tracing.ToolCallPart(
                            id=call["id"],
                            name=call["function"]["name"],
                            arguments=call["function"]["arguments"],
                        )
                        for call in item.get("tool_calls", [])
                    ],
                )
                for item in history
            ]

        with tracing.start_conversation(
            agent_name="trail-supply-starter", conversation_id=result["conversation_id"]
        ) as conversation:
            with conversation.start_turn(user_message=message, model=model) as turn:
                for _ in range(max_steps):
                    with turn.start_llm(
                        model=model,
                        provider_name="coreweave",
                        system_instructions=[prompt],
                    ) as llm:
                        llm.input_messages = trace_messages(messages)
                        llm.request_temperature = temperature
                        llm.request_max_tokens = max_tokens
                        completion = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=TOOL_SCHEMAS,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        reply = completion.choices[0].message
                        llm.record(
                            output_messages=trace_messages(
                                [reply.model_dump(exclude_none=True)]
                            ),
                            response_id=completion.id,
                            response_model=completion.model,
                            finish_reasons=[completion.choices[0].finish_reason or ""],
                        )
                        if completion.usage:
                            result["input_tokens"] += completion.usage.prompt_tokens
                            result["output_tokens"] += (
                                completion.usage.completion_tokens
                            )
                            llm.usage = tracing.Usage(
                                input_tokens=completion.usage.prompt_tokens,
                                output_tokens=completion.usage.completion_tokens,
                            )
                    messages.append(reply.model_dump(exclude_none=True))
                    if not reply.tool_calls:
                        result["answer"] = reply.content or ""
                        turn.record(
                            output_messages=[
                                tracing.Message(
                                    role="assistant", content=result["answer"]
                                )
                            ]
                        )
                        return result
                    # Bound calls even when a model proposes many in one response.
                    if len(reply.tool_calls) > 3:
                        result["error"] = "too_many_tool_calls"
                        return result
                    for call in reply.tool_calls:
                        name = call.function.name
                        arguments = {}
                        with turn.start_tool(
                            name=name,
                            arguments=call.function.arguments,
                            tool_call_id=call.id,
                        ) as span:
                            try:
                                arguments = json.loads(call.function.arguments)
                                if not isinstance(arguments, dict) or name not in tools:
                                    raise ValueError(
                                        "unknown tool or malformed arguments"
                                    )
                                if not all(
                                    isinstance(value, str)
                                    for value in arguments.values()
                                ):
                                    raise ValueError("tool arguments must be strings")
                                output = tools[name](**arguments)
                            except (ValueError, TypeError):
                                output = {"error": "invalid_tool_call"}
                                result["error"] = "invalid_tool_call"
                            span.result = json.dumps(output)
                        result["tools"].append(
                            {
                                "name": name,
                                "arguments": arguments
                                if isinstance(arguments, dict)
                                else {},
                                "result": output,
                            }
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": json.dumps(output),
                            }
                        )
                result["error"] = "step_limit"
        return result

    return (run_agent,)


@app.cell(hide_code=True)
def _(os, re):
    def read_destination(entity: str, project: str) -> tuple[str, str]:
        if not os.getenv("WANDB_API_KEY", "").strip():
            raise ValueError("Add WANDB_API_KEY to Secrets or your local .env file.")
        entity, project = entity.strip(), project.strip()
        if (
            not entity
            or not project
            or not re.fullmatch(r"[\w.-]+", entity)
            or not re.fullmatch(r"[\w.-]+", project)
        ):
            raise ValueError(
                "Set a team and project using letters, digits, dots, hyphens or underscores."
            )
        return entity, project

    return (read_destination,)


@app.cell(hide_code=True)
def _(
    AGENT_SCENARIOS,
    DATASET_SHA256,
    DATASET_VERSION,
    ORDERS,
    STORE_POLICY,
    TOOL_SCHEMAS,
    OpenAI,
    Path,
    inspect,
    json,
    os,
    read_destination,
    run_agent,
    score_agent,
    textwrap,
    tracing,
    uuid,
    wandb,
    validate_agent_config,
):
    def run_experiment(
        settings: dict, prompt: str, tools: dict, configuration: dict
    ) -> dict:
        """Called only by the submit callback (or an explicit smoke-test script)."""
        # Freeze and validate editable choices before credentials, files, or services.
        effective = validate_agent_config(configuration)
        entity, project = read_destination(settings["entity"], settings["project"])
        model = effective["model"]
        count = int(settings["examples"])
        if not 1 <= count <= len(AGENT_SCENARIOS):
            raise ValueError("Choose a bounded number of the supplied scenarios.")
        run_id = "agent-" + uuid.uuid4().hex[:12]
        config = {
            **settings,
            **effective,
            "prompt": prompt,
            "dataset_version": DATASET_VERSION,
            "dataset_sha256": DATASET_SHA256,
        }
        source = (
            "AGENT_CONFIG = "
            + repr(effective)
            + "\nAGENT_PROMPT = "
            + repr(prompt)
            + "\nTOOL_SCHEMAS = "
            + repr(TOOL_SCHEMAS)
            + "\n\n"
            + textwrap.dedent(inspect.getsource(run_agent))
            + "\n"
            + "\n".join(
                textwrap.dedent(inspect.getsource(function))
                for function in tools.values()
            )
        )
        # This selected source snapshot is valid Python. Supporting data lives
        # in fixtures.json; the executable notebook supplies imports.
        compile(source, "experiment.py", "exec")
        report = {
            "run_id": run_id,
            "variant": settings["variant"],
            "rows": [],
            "status": "running",
            "config": config,
            "errors": [],
            "evidence_uploaded": False,
            "traces_flushed": False,
            "run_url": f"https://wandb.ai/{entity}/{project}/runs/{run_id}",
            "project_url": f"https://forge.coreweave.com/wandb/{entity}/{project}",
            "agent_lens_url": f"https://forge.coreweave.com/agent-lens/{entity}/{project}",
        }
        record_path = Path(".aria-starter") / "agent" / f"{run_id}.json"
        report["local_record"] = str(record_path)

        def save_report():
            record_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = record_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(record_path)

        def record_error(stage, error):
            report["errors"].append({"stage": stage, "type": type(error).__name__})
            report["status"] = "needs_attention"

        # Fail before paid work if identifiers cannot be recorded locally.
        save_report()
        stage = "model_discovery"
        try:
            with OpenAI(
                base_url="https://api.inference.wandb.ai/v1",
                api_key=os.environ["WANDB_API_KEY"],
                project=f"{entity}/{project}",
                timeout=60,
                max_retries=0,
            ) as client:
                if model not in {item.id for item in client.models.list().data}:
                    report["status"] = "blocked"
                    report["message"] = (
                        "This model is unavailable to your account. Choose a catalog model."
                    )
                    return report
                stage = "wandb_run"
                with wandb.init(
                    entity=entity,
                    project=project,
                    id=run_id,
                    name=settings["variant"],
                    job_type="agent-evaluation",
                    config=config,
                    settings=wandb.Settings(
                        disable_git=True, disable_code=True, console="off"
                    ),
                ) as run:
                    report["run_url"] = run.url
                    session = None
                    try:
                        stage = "tracing_init"
                        session = tracing.init(
                            f"{entity}/{project}", autopatch_integrations=False
                        )
                        for scenario in AGENT_SCENARIOS[:count]:
                            stage = "inference"
                            conversation_id = str(uuid.uuid4())
                            report["active_scenario"] = {
                                "id": scenario["id"],
                                "conversation_id": conversation_id,
                            }
                            save_report()
                            outcome = run_agent(
                                scenario["message"],
                                client,
                                model,
                                prompt,
                                tools,
                                temperature=config["temperature"],
                                max_steps=config["max_steps"],
                                max_tokens=config["max_tokens"],
                                conversation_id=conversation_id,
                            )
                            report["rows"].append(score_agent(scenario, outcome))
                            report.pop("active_scenario", None)
                            save_report()
                        report["status"] = "finished"
                    except Exception as error:
                        # Return partial outcomes and IDs even when inference fails.
                        record_error(stage, error)
                    finally:
                        if session is not None:
                            try:
                                report["traces_flushed"] = bool(session.force_flush())
                                if not report["traces_flushed"]:
                                    record_error("trace_flush", TimeoutError())
                            except Exception as error:
                                record_error("trace_flush", error)
                            try:
                                tracing.shutdown()
                            except Exception as error:
                                record_error("trace_shutdown", error)
                        try:
                            run.summary["status"] = report["status"]
                            run.summary["completed_examples"] = len(report["rows"])
                            run.summary["contract_pass_rate"] = (
                                sum(row["passed"] for row in report["rows"]) / count
                            )
                            artifact = wandb.Artifact(
                                f"{run_id}-evidence", type="evaluation"
                            )
                            for filename, payload in {
                                "results.json": report,
                                "config.json": config,
                                "fixtures.json": {
                                    "scenarios": AGENT_SCENARIOS,
                                    "orders": ORDERS,
                                    "policy": STORE_POLICY,
                                },
                            }.items():
                                with artifact.new_file(filename) as file:
                                    json.dump(payload, file, indent=2)
                            with artifact.new_file("experiment.py") as file:
                                file.write(source)
                            run.log_artifact(artifact).wait()
                            report["evidence_uploaded"] = True
                        except Exception as error:
                            record_error("evidence_upload", error)
                        save_report()
                    stage = "wandb_finish"
        except Exception as error:
            record_error(stage, error)
        finally:
            try:
                save_report()
            except OSError as error:
                record_error("local_record", error)
        return report

    return (run_experiment,)


@app.cell(hide_code=True)
def _(mo):
    get_experiments, set_experiments = mo.state([])
    get_status, set_status = mo.state("Ready. No requests have been made.")
    return get_experiments, get_status, set_experiments, set_status


@app.cell(hide_code=True)
def _():
    def validate_agent_config(configuration):
        """Copy only supported settings, rejecting invalid values before service calls."""
        model = configuration.get("model")
        temperature = configuration.get("temperature")
        steps = configuration.get("max_steps")
        tokens = configuration.get("max_tokens")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("AGENT_CONFIG.model must name a Forge Inference model.")
        if type(temperature) not in (int, float) or not 0 <= temperature <= 1:
            raise ValueError("AGENT_CONFIG.temperature must be between 0 and 1.")
        if type(steps) is not int or not 1 <= steps <= 4:
            raise ValueError("AGENT_CONFIG.max_steps must be an integer from 1 to 4.")
        if type(tokens) is not int or not 64 <= tokens <= 512:
            raise ValueError(
                "AGENT_CONFIG.max_tokens must be an integer from 64 to 512."
            )
        return {
            "model": model.strip(),
            "temperature": temperature,
            "max_steps": steps,
            "max_tokens": tokens,
        }

    return (validate_agent_config,)


if __name__ == "__main__":
    app.run()
