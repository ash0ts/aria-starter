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


@app.cell
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

    load_dotenv(Path.cwd() / ".env", override=False)
    return OpenAI, Path, inspect, json, mo, os, re, textwrap, tracing, uuid, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Build the Trail Supply support assistant

    You maintain a fictional outdoor shop's assistant. It can look up orders,
    read store policy, and simulate a handoff to a human. All records are supplied.

    **Run → inspect a mistake → ask ARIA → edit Python → compare.**

    Start with the baseline prompt below. Run the prepared questions, then edit
    `AGENT_PROMPT` or a tool and submit another experiment. The score checks tools
    and outcomes; read the answer too, because a passing score does not prove
    every sentence is correct. Escalation contacts nobody.

    Add `WANDB_API_KEY` in molab's **Developer panel → Secrets**, then set your
    team and a project below. Inference spends Forge credits only when you submit
    **Run agent experiment**. Opening this notebook makes no service calls.
    """)
    return


@app.cell
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


@app.cell
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


@app.cell
def _(TOOL_SCHEMAS, json, tracing, uuid):
    def run_agent(
        message,
        client,
        model,
        prompt,
        tools,
        *,
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
                        llm.request_temperature = 0
                        llm.request_max_tokens = max_tokens
                        completion = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=TOOL_SCHEMAS,
                            temperature=0,
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


@app.cell
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


@app.cell
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
):
    def run_experiment(settings: dict, prompt: str, tools: dict) -> dict:
        """Called only by the submit callback (or an explicit smoke-test script)."""
        entity, project = read_destination(settings["entity"], settings["project"])
        model = settings["model"].strip()
        if not model:
            raise ValueError("Choose a model from the Forge Inference catalog.")
        count = int(settings["examples"])
        if not 1 <= count <= len(AGENT_SCENARIOS):
            raise ValueError("Choose a bounded number of the supplied scenarios.")
        run_id = "agent-" + uuid.uuid4().hex[:12]
        config = {
            **settings,
            "prompt": prompt,
            "dataset_version": DATASET_VERSION,
            "dataset_sha256": DATASET_SHA256,
            "max_steps": 4,
            "max_tokens": 512,
        }
        source = (
            "AGENT_PROMPT = "
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


@app.cell
def _(mo):
    get_experiments, set_experiments = mo.state([])
    get_status, set_status = mo.state("Ready. No requests have been made.")
    return get_experiments, get_status, set_experiments, set_status


@app.cell
def _(
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
            report = run_experiment(dict(values), AGENT_PROMPT, dict(support_tools))
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
    ## Run an experiment

    {entity} {project}

    {model} {variant} {examples}

    Each question uses at most four model requests, each capped at 512 output
    tokens. Give the first run the name `baseline`, then edit code and name the
    next run `candidate`. Submitting again intentionally starts a new experiment.
    """)
        .batch(
            entity=mo.ui.text(value=os.getenv("WANDB_ENTITY", ""), label="W&B team"),
            project=mo.ui.text(
                value=os.getenv("WANDB_PROJECT", "aria-support-desk"), label="Project"
            ),
            model=mo.ui.text(
                value="meta-llama/Llama-3.3-70B-Instruct",
                label="Forge Inference model",
                full_width=True,
            ),
            variant=mo.ui.text(value="baseline", label="Experiment name"),
            examples=mo.ui.slider(1, len(AGENT_SCENARIOS), value=2, label="Questions"),
        )
        .form(
            submit_button_label="Run agent experiment",
            on_change=submit_agent_experiment,
        )
    )
    agent_form
    return agent_form, submit_agent_experiment


@app.cell
def _(get_experiments, get_status, json, mo):
    _runs = get_experiments()
    _sections = [mo.md(get_status())]
    for _run in _runs:
        _sections.append(
            mo.md(
                f"### {_run['variant']}\n[W&B run]({_run['run_url']}) · [Agent Lens]({_run['agent_lens_url']})"
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
        _sections.append(mo.ui.table(_run["rows"], selection=None))
    if _runs:
        _latest = _runs[-1]
        _baseline = _runs[0]
        _prompt = (
            f"I am building the fictional Trail Supply support assistant. Compare {_baseline['run_url']} "
            f"with {_latest['run_url']}. Read their evaluation artifacts, config, and experiment.py. "
            "Find a failed tool/outcome contract or a regression on a shared scenario. "
            "Explain it using the logged evidence. Suggest one small Python or prompt edit and a test. "
            "The automated score does not judge every factual claim; also inspect the customer answers. "
            "Do not launch another paid experiment. I will apply your change in marimo."
        )
        _sections += [
            mo.md(
                f"## Ask ARIA\nOpen [this project]({_latest['project_url']}), click **Ask ARIA**, and paste:"
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
    ## Make it yours

    - Ask ARIA why an unknown order should trigger a handoff rather than a made-up ETA.
    - Add a tool over `PRODUCTS`, then add a scenario and its expected behavior.
    - Change the prompt to check return eligibility before promising a refund.
    - Compare failures on the same scenarios; adding easier questions can inflate the average.

    Record what you changed, what ARIA helped explain, and what remained confusing.
    Save or download your edited notebook. After a restart, use your W&B run links
    to recover the durable evidence; the comparison list above belongs to this session.
    """)
    return


if __name__ == "__main__":
    app.run()
