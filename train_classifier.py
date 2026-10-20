# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "marimo==0.25.2",
#   "scikit-learn==1.9.1",
#   "joblib==1.6.0",
#   "wandb==0.31.0",
#   "python-dotenv==1.2.4",
# ]
# ///
# ruff: noqa: PLR1711
"""A synthetic support-ticket experiment. No credentials needed for CPU training."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import hashlib
    import inspect
    import json
    import os
    import tempfile
    import uuid
    from pathlib import Path

    import joblib
    import marimo as mo
    from dotenv import load_dotenv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from sklearn.pipeline import Pipeline

    load_dotenv(Path.cwd() / ".env", override=False)
    return (
        LogisticRegression,
        Path,
        Pipeline,
        TfidfVectorizer,
        accuracy_score,
        classification_report,
        confusion_matrix,
        copy,
        f1_score,
        hashlib,
        inspect,
        joblib,
        json,
        mo,
        os,
        tempfile,
        uuid,
    )


@app.cell
def _(mo):
    mo.md("""
    # Train a support-ticket classifier

    Trail Supply is a **fictional outdoor shop**. Train a small CPU model to send
    its synthetic support tickets to the right queue. No ML project, API key, or
    GPU is needed to begin.

    1. Inspect the baseline's development mistakes below.
    2. Connect your student project, upload the comparison, and ask ARIA about a mistake.
    3. Edit the candidate's Python settings or add training examples. Compare again.
    4. When you have chosen a candidate, evaluate the held-out test set once.

    A score can go down after a change. Explain the result rather than chasing a
    perfect score on this small teaching dataset. Store credentials in molab
    Secrets or an ignored local `.env` file; never paste keys into code cells.
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
    return DATASET_SHA256, DATASET_VERSION, LABELS, TICKETS


@app.cell
def _(LABELS, TICKETS):
    def validate_tickets(tickets):
        ids = [row["id"] for row in tickets]
        if len(ids) != len(set(ids)):
            raise ValueError("Ticket IDs must be unique.")
        families = {}
        for row in tickets:
            if row["label"] not in LABELS or row["split"] not in {
                "train",
                "dev",
                "test",
            }:
                raise ValueError("Unknown label or split.")
            families.setdefault(row["family"], set()).add(row["split"])
        if any(len(splits) != 1 for splits in families.values()):
            raise ValueError("A scenario family appears in multiple splits.")
        texts = [row["text"].casefold().strip() for row in tickets]
        if len(texts) != len(set(texts)):
            raise ValueError("Duplicate ticket text found.")

    validate_tickets(TICKETS)
    train_tickets = [row for row in TICKETS if row["split"] == "train"]
    dev_tickets = [row for row in TICKETS if row["split"] == "dev"]
    test_tickets = [row for row in TICKETS if row["split"] == "test"]
    return dev_tickets, test_tickets, train_tickets, validate_tickets


@app.cell
def _(DATASET_VERSION, LABELS, TICKETS, mo, train_tickets):
    mo.vstack(
        [
            mo.md(f"""
        ## Read the training data

        Dataset `{DATASET_VERSION}` contains {len(TICKETS)} fictional tickets.
        Each scenario family stays in one split, including its paraphrases.
        The labels are `{", ".join(LABELS)}`. A request about a purchase's
        delivery is `order_status`; a request about an item's features before
        buying is `product_question`. Safety, privacy, fraud, and requests for
        human help belong in `human_escalation`.

        You can inspect the fixture cell, but use development errors to guide
        edits. Do not copy development or test tickets into training.
        """),
            mo.ui.table(train_tickets, selection=None, page_size=8),
        ]
    )
    return


@app.cell
def _():
    # Keep this baseline fixed so experiments remain comparable with the SFT notebook.
    BASELINE_CONFIG = {"ngram_range": (1, 1), "C": 1.0, "class_weight": None}

    # EDIT HERE. Try (1, 2) word phrases, a different C, or class_weight="balanced".
    CANDIDATE_CONFIG = {"ngram_range": (1, 2), "C": 2.0, "class_weight": None}

    # Add your own synthetic training scenarios. Do not paraphrase held-out tickets.
    # Example: {"text": "Is the wool hat available in green?", "label": "product_question"}
    EXTRA_TRAINING_EXAMPLES = []
    return BASELINE_CONFIG, CANDIDATE_CONFIG, EXTRA_TRAINING_EXAMPLES


@app.cell
def _(
    LABELS,
    LogisticRegression,
    Pipeline,
    TfidfVectorizer,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
):
    def make_classifier(config):
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=tuple(config["ngram_range"]), lowercase=True
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=config["C"],
                        class_weight=config["class_weight"],
                        max_iter=1000,
                        random_state=7,
                    ),
                ),
            ]
        )

    def evaluate_classifier(model, tickets):
        expected = [row["label"] for row in tickets]
        predicted = model.predict([row["text"] for row in tickets]).tolist()
        report = classification_report(
            expected, predicted, labels=LABELS, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(expected, predicted, labels=LABELS).tolist()
        return {
            "metrics": {
                "accuracy": float(accuracy_score(expected, predicted)),
                "macro_f1": float(
                    f1_score(
                        expected,
                        predicted,
                        labels=LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
            },
            "per_class": [{"label": label, **report[label]} for label in LABELS],
            "confusion_matrix": [
                {"actual": label, **dict(zip(LABELS, counts, strict=True))}
                for label, counts in zip(LABELS, matrix, strict=True)
            ],
            "predictions": [
                {**row, "prediction": prediction, "correct": row["label"] == prediction}
                for row, prediction in zip(tickets, predicted, strict=True)
            ],
        }

    return evaluate_classifier, make_classifier


@app.cell
def _(
    BASELINE_CONFIG,
    CANDIDATE_CONFIG,
    DATASET_SHA256,
    EXTRA_TRAINING_EXAMPLES,
    LABELS,
    dev_tickets,
    evaluate_classifier,
    hashlib,
    inspect,
    json,
    make_classifier,
    test_tickets,
    train_tickets,
):
    _held_out_texts = {
        row["text"].casefold().strip() for row in dev_tickets + test_tickets
    }
    for _example in EXTRA_TRAINING_EXAMPLES:
        if _example["label"] not in LABELS or not _example["text"].strip():
            raise ValueError(
                "Each extra example needs nonempty text and one of LABELS."
            )
        if _example["text"].casefold().strip() in _held_out_texts:
            raise ValueError("Keep held-out tickets out of training.")
    candidate_training = train_tickets + [
        {**row, "id": f"student-{i}", "family": f"student-{i}", "split": "train"}
        for i, row in enumerate(EXTRA_TRAINING_EXAMPLES)
    ]
    baseline_model = make_classifier(BASELINE_CONFIG).fit(
        [row["text"] for row in train_tickets], [row["label"] for row in train_tickets]
    )
    candidate_model = make_classifier(CANDIDATE_CONFIG).fit(
        [row["text"] for row in candidate_training],
        [row["label"] for row in candidate_training],
    )
    baseline_dev = evaluate_classifier(baseline_model, dev_tickets)
    candidate_dev = evaluate_classifier(candidate_model, dev_tickets)
    experiment_id = hashlib.sha256(
        json.dumps(
            [
                DATASET_SHA256,
                BASELINE_CONFIG,
                CANDIDATE_CONFIG,
                EXTRA_TRAINING_EXAMPLES,
                inspect.getsource(make_classifier),
                inspect.getsource(evaluate_classifier),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    return (
        baseline_dev,
        baseline_model,
        candidate_dev,
        candidate_model,
        candidate_training,
        experiment_id,
    )


@app.cell
def _(baseline_dev, candidate_dev, experiment_id, mo):
    mo.vstack(
        [
            mo.md(
                f"## Compare development results\n\nExperiment `{experiment_id}`. Higher accuracy and macro-F1 are better."
            ),
            mo.ui.table(
                [
                    {"model": "baseline", **baseline_dev["metrics"]},
                    {"model": "candidate", **candidate_dev["metrics"]},
                ],
                selection=None,
            ),
            mo.accordion(
                {
                    "Baseline mistakes": mo.ui.table(
                        [r for r in baseline_dev["predictions"] if not r["correct"]],
                        selection=None,
                    ),
                    "Candidate mistakes": mo.ui.table(
                        [r for r in candidate_dev["predictions"] if not r["correct"]],
                        selection=None,
                    ),
                    "Baseline per-class scores": mo.ui.table(
                        baseline_dev["per_class"], selection=None
                    ),
                    "Candidate per-class scores": mo.ui.table(
                        candidate_dev["per_class"], selection=None
                    ),
                    "Baseline confusion matrix (rows actual, columns predicted)": mo.ui.table(
                        baseline_dev["confusion_matrix"], selection=None
                    ),
                    "Candidate confusion matrix (rows actual, columns predicted)": mo.ui.table(
                        candidate_dev["confusion_matrix"], selection=None
                    ),
                }
            ),
            mo.md(
                "Read a misclassified ticket. Is the label clear? Does the model miss a phrase or lack a related training scenario? Ask ARIA for one small change and its rationale."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    get_final_result, set_final_result = mo.state(None)
    get_upload_result, set_upload_result = mo.state(None)
    return get_final_result, get_upload_result, set_final_result, set_upload_result


@app.cell
def _(
    baseline_model,
    candidate_model,
    evaluate_classifier,
    experiment_id,
    mo,
    set_final_result,
    test_tickets,
):
    def _evaluate_final(_click):
        set_final_result(
            {
                "experiment_id": experiment_id,
                "baseline": evaluate_classifier(baseline_model, test_tickets),
                "candidate": evaluate_classifier(candidate_model, test_tickets),
            }
        )

    final_test_button = mo.ui.button(
        label="Evaluate final test", on_click=_evaluate_final
    )
    mo.vstack(
        [
            mo.md(
                "## Final test\n\nChoose your candidate using development results first. This button evaluates held-out scenarios. Once you inspect test errors, the test set is no longer an unbiased guide for further edits."
            ),
            final_test_button,
        ]
    )
    return (final_test_button,)


@app.cell
def _(experiment_id, get_final_result, mo):
    _final = get_final_result()
    if _final is not None:
        mo.output.append(
            mo.md(f"Final results for experiment `{_final['experiment_id']}`.")
        )
        if _final["experiment_id"] != experiment_id:
            mo.output.append(
                mo.callout(
                    "You edited the candidate after this final evaluation. These results describe the earlier experiment.",
                    kind="warn",
                )
            )
        mo.output.append(
            mo.ui.table(
                [
                    {"model": name, **_final[name]["metrics"]}
                    for name in ("baseline", "candidate")
                ],
                selection=None,
            )
        )
        mo.output.append(
            mo.accordion(
                {
                    f"{name.title()} test details": mo.vstack(
                        [
                            mo.ui.table(
                                _final[name]["confusion_matrix"], selection=None
                            ),
                            mo.ui.table(_final[name]["per_class"], selection=None),
                            mo.ui.table(_final[name]["predictions"], selection=None),
                        ]
                    )
                    for name in ("baseline", "candidate")
                }
            )
        )
    return


@app.cell
def _(Path, inspect, make_classifier, evaluate_classifier):
    def capture_source():
        # No credentials are read here. Keep all secrets out of notebook cells.
        try:
            return Path(__file__).read_text(encoding="utf-8")
        except (NameError, OSError):
            return "\n\n".join(
                inspect.getsource(fn) for fn in (make_classifier, evaluate_classifier)
            )

    return (capture_source,)


@app.cell
def _(Path, joblib, json, os, tempfile, uuid):
    def upload_comparison(snapshot, models, entity, project, publish_status):
        import wandb

        if not entity.strip() or not project.strip():
            raise ValueError("Enter your W&B team and student project.")
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError(
                "Set WANDB_API_KEY in molab Secrets or an ignored .env file, then restart the notebook."
            )
        run_ids = {name: uuid.uuid4().hex[:12] for name in ("baseline", "candidate")}
        status = {
            "state": "uploading",
            "entity": entity,
            "project": project,
            "run_ids": run_ids,
            "urls": {},
            "snapshot": snapshot,
        }
        publish_status(dict(status))
        try:
            with tempfile.TemporaryDirectory(prefix="aria-starter-") as folder:
                root = Path(folder)
                (root / "dataset.json").write_text(
                    json.dumps(snapshot["dataset"], indent=2)
                )
                (root / "experiment.json").write_text(
                    json.dumps(
                        {k: v for k, v in snapshot.items() if k != "source"}, indent=2
                    )
                )
                (root / "train_classifier.py").write_text(snapshot["source"])
                for name in ("baseline", "candidate"):
                    # IDs are allocated before any write; errors retain them in the UI.
                    with wandb.init(
                        entity=entity.strip(),
                        project=project.strip(),
                        id=run_ids[name],
                        name=f"cpu-{name}-{snapshot['experiment_id']}",
                        group=snapshot["experiment_id"],
                        job_type="cpu-classifier",
                        dir=folder,
                        config={
                            "dataset_version": snapshot["dataset_version"],
                            "dataset_sha256": snapshot["dataset_sha256"],
                            "model": name,
                            "settings": snapshot["configs"][name],
                            "seed": 7,
                        },
                        settings=wandb.Settings(disable_git=True, save_code=False),
                    ) as run:
                        status["urls"][name] = run.url
                        publish_status(dict(status))
                        for split in ("dev", "test"):
                            if split not in snapshot:
                                continue
                            result = snapshot[split][name]
                            run.log(
                                {
                                    f"{split}/{key}": value
                                    for key, value in result["metrics"].items()
                                }
                            )
                            columns = [
                                "id",
                                "family",
                                "text",
                                "label",
                                "prediction",
                                "correct",
                            ]
                            run.log(
                                {
                                    f"{split}/predictions": wandb.Table(
                                        columns=columns,
                                        data=[
                                            [row[key] for key in columns]
                                            for row in result["predictions"]
                                        ],
                                    )
                                }
                            )
                        evidence = wandb.Artifact(
                            f"cpu-{name}-{run_ids[name]}-evidence", type="experiment"
                        )
                        for filename in (
                            "dataset.json",
                            "experiment.json",
                            "train_classifier.py",
                        ):
                            evidence.add_file(str(root / filename))
                        run.log_artifact(evidence)
                        model_path = root / f"{name}.joblib"
                        joblib.dump(models[name], model_path)
                        artifact = wandb.Artifact(
                            f"cpu-{name}-{run_ids[name]}-model", type="model"
                        )
                        artifact.add_file(str(model_path))
                        run.log_artifact(artifact)
            status["state"] = "complete"
        except Exception as exc:  # noqa: BLE001 -- Keep run IDs visible for any SDK failure.
            # Do not render API response bodies, which may include credential material.
            status["state"] = "failed"
            status["error"] = (
                f"Upload stopped ({type(exc).__name__}). Check your team, project permissions, API key, and network. Keep the run IDs below; no automatic retry was submitted."
            )
        publish_status(dict(status))

    return (upload_comparison,)


@app.cell
def _(
    BASELINE_CONFIG,
    CANDIDATE_CONFIG,
    DATASET_SHA256,
    DATASET_VERSION,
    EXTRA_TRAINING_EXAMPLES,
    TICKETS,
    baseline_dev,
    baseline_model,
    candidate_dev,
    candidate_model,
    capture_source,
    copy,
    experiment_id,
    get_final_result,
    mo,
    os,
    set_upload_result,
    upload_comparison,
):
    def _upload(values):
        if values is None:
            return
        # This callback runs only for a new submission. Reactive cell execution
        # creates a form; it never sends a request. Capture results before I/O.
        snapshot = copy.deepcopy(
            {
                "experiment_id": experiment_id,
                "dataset_version": DATASET_VERSION,
                "dataset_sha256": DATASET_SHA256,
                "dataset": TICKETS,
                "configs": {"baseline": BASELINE_CONFIG, "candidate": CANDIDATE_CONFIG},
                "extra_training_examples": EXTRA_TRAINING_EXAMPLES,
                "dev": {"baseline": baseline_dev, "candidate": candidate_dev},
                "source": capture_source(),
            }
        )
        final = get_final_result()
        if final is not None and final["experiment_id"] == experiment_id:
            snapshot["test"] = {name: final[name] for name in ("baseline", "candidate")}
        try:
            upload_comparison(
                snapshot,
                {
                    "baseline": copy.deepcopy(baseline_model),
                    "candidate": copy.deepcopy(candidate_model),
                },
                str(values["entity"]),
                str(values["project"]),
                set_upload_result,
            )
        except ValueError as exc:
            set_upload_result({"state": "blocked", "error": str(exc)})

    upload_form = mo.ui.dictionary(
        {
            "entity": mo.ui.text(
                label="W&B team", value=os.environ.get("WANDB_ENTITY", "")
            ),
            "project": mo.ui.text(
                label="Your student project", value=os.environ.get("WANDB_PROJECT", "")
            ),
        }
    ).form(submit_button_label="Upload comparison to W&B", on_change=_upload)
    mo.vstack(
        [
            mo.md("""
        ## Ask ARIA about this experiment

        Uploading creates two W&B runs with metrics, predictions, model artifacts,
        and a snapshot of this notebook and dataset. It does not launch paid training.
        Every submission creates a new pair of runs. Opening the notebook or editing
        a cell does not upload anything. Use your facilitator's team and your own
        project so ARIA can find your experiments.
        """),
            upload_form,
        ]
    )
    return (upload_form,)


@app.cell
def _(get_upload_result, json, mo):
    _uploaded = get_upload_result()
    if _uploaded is not None:
        if "error" in _uploaded:
            mo.output.append(mo.callout(_uploaded["error"], kind="warn"))
        if "run_ids" in _uploaded:
            mo.output.append(
                mo.md(
                    f"Upload: **{_uploaded['state']}**. Retain these run IDs: `{json.dumps(_uploaded['run_ids'])}`."
                )
            )
            for _name, _url in _uploaded["urls"].items():
                mo.output.append(mo.md(f"[{_name.title()} run]({_url})"))
        if _uploaded["state"] == "complete":
            _snapshot = _uploaded["snapshot"]
            _mistakes = {
                name: [
                    r for r in _snapshot["dev"][name]["predictions"] if not r["correct"]
                ]
                for name in ("baseline", "candidate")
            }
            _prompt = f"""I am learning to classify fictional outdoor-shop support tickets.
My objective is to understand one failure and test a small change, using development results only.
Project: {_uploaded["entity"]}/{_uploaded["project"]}
Baseline: {_uploaded["urls"]["baseline"]} (run {_uploaded["run_ids"]["baseline"]})
Variant: {_uploaded["urls"]["candidate"]} (run {_uploaded["run_ids"]["candidate"]})
Dataset: {_snapshot["dataset_version"]} / {_snapshot["dataset_sha256"]}
Changed settings: {json.dumps(_snapshot["configs"], sort_keys=True)}
Added training examples: {json.dumps(_snapshot["extra_training_examples"])}
Observed development failures: {json.dumps(_mistakes)}
The evidence artifacts contain the exact notebook code, dataset, and comparison results.
Explain one confusion, suggest one small code or training-data edit and why it might help, and tell me how to compare it against this baseline. Do not use test examples to choose changes."""
            mo.output.append(
                mo.md(
                    "Open either run, go to its project, and choose **Ask ARIA**. Copy this prompt. Apply suggestions yourself in the editable Python cell, then compare and upload again."
                )
            )
            mo.output.append(
                mo.ui.text_area(
                    value=_prompt, label="Copy into Ask ARIA", full_width=True, rows=12
                )
            )
    return


if __name__ == "__main__":
    app.run()
