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


@app.cell(hide_code=True)
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

    _environment_loaded = load_dotenv(Path.cwd() / ".env", override=False)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Route support tickets for an outdoor shop

    You have joined the support desk at **Trail Supply, a fictional outdoor shop**. Customers send messages about
    late deliveries, returns, and products. Staff first need to decide which queue should handle each message.
    A **ticket** is one customer message. A **label** is its correct queue. A **classifier** is a program that predicts
    a label from the message. In this lab it chooses a queue; it does not write a reply to the customer.

    Your job is to train a small classifier, find a ticket it routes incorrectly, and test one change.
    The dataset and working code are supplied. You need to be comfortable editing a Python dictionary or list;
    you do not need an ML project, GPU, API key, or previous machine-learning experience to start.

    **What you will make:** a baseline/candidate comparison, one explained mistake, and a short account of whether
    your change helped. The CPU experiment is free to run locally. Uploading its evidence enables the ARIA part.
    Complete sections 1–5 before using the final test in section 6.

    ### Open and run this notebook

    If you are in molab, fork this notebook into your own workspace before editing.
    If you downloaded this `.py` file, open a terminal in its folder and run the command below.
    [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
    are available if your terminal says `uvx` is not found. The first launch installs the pinned packages.

    ```sh
    uvx marimo edit --sandbox train_classifier.py
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
    ## The task you are teaching the classifier

    | Queue name in Python | What belongs here | Fictional example |
    | --- | --- | --- |
    | `order_status` | Tracking, dispatch, or delivery of a purchase | “Where is my order? Please send a tracking link.” |
    | `returns` | Returning an item, refunds, or exchanges | “How do I return this jacket?” |
    | `product_question` | Features, sizing, or availability before buying | “Is this backpack waterproof?” |
    | `human_escalation` | Safety, fraud, privacy, or an explicit request for a person | “Please let me speak to a human.” |

    Read each message and decide its queue yourself. For example, “Where is my order?” should go to `order_status`.
    The supplied labels are the answer key used to check the program. A wrong prediction means the program disagreed
    with that key; if the key itself seems ambiguous, record that before trying to improve the model.

    We split the 64 supplied tickets into three groups:

    | Split | Tickets | What you use it for |
    | --- | --- | --- |
    | Training | 32 | Examples the model learns from |
    | Development (`dev`) | 16 | Unseen examples you inspect while choosing changes |
    | Test | 16 | A final check after you stop choosing changes |

    Each pair of related phrasings stays in one split. Otherwise a model could see almost the same question in training
    and evaluation, making the score look better than its ability to handle a new situation.
    Do not move a hard development ticket into training just to make it pass.

    ### How the supplied Python learns

    The first step, **TF-IDF**, turns a ticket into numbers describing which words occur and how distinctive they are
    among the training tickets. The second, **logistic regression**, learns which word patterns tend to go with each queue.
    Despite its name, logistic regression here chooses a category. `fit` learns from labeled examples;
    `predict` applies what was learned to other messages. Neither step calls a language-model service.

    A **baseline** is the unchanged reference model. A **candidate** is the version whose settings you can change.
    Both predict the same development tickets, so a score difference can be traced to the choices you changed.
    """)
    return


@app.cell(hide_code=True)
def _(DATASET_VERSION, LABELS, TICKETS, mo, train_tickets):
    mo.vstack(
        [
            mo.md(f"""
        ## 1. Inspect the supplied training data

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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Edit here: candidate settings

    The **fixed baseline** uses single words, `C=1.0`, and no class weighting. Keep it unchanged.
    The supplied candidate below already differs in **two** ways: it includes two-word phrases and uses `C=2.0`.
    Run it once to see a working comparison. For your **first controlled experiment**, copy the baseline values
    into `CANDIDATE_CONFIG`, then change **only** `ngram_range` to `(1, 2)`. Leave extra examples empty.

    - `ngram_range=(1, 1)` counts single words; `(1, 2)` also counts phrases such as “return window”.
    - `C` controls how freely the classifier fits the training data. Regularization discourages overly specific word rules. A larger `C` weakens that restraint: it can fit training examples more closely but do worse on new messages (overfitting).
    - `class_weight=None` weights examples equally. `"balanced"` gives more weight to less frequent training labels.

    Change one choice at a time so you can explain the difference. CPU results recompute automatically when you run an edited cell; uploading is a separate action.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Supplied implementation: fixed reference, shared with the SFT CPU comparison.
    BASELINE_CONFIG = {"ngram_range": (1, 1), "C": 1.0, "class_weight": None}
    return (BASELINE_CONFIG,)


@app.cell
def _():
    # Edit here: first try C=1.0 so only word phrases differ from the baseline.
    CANDIDATE_CONFIG = {"ngram_range": (1, 2), "C": 2.0, "class_weight": None}
    return (CANDIDATE_CONFIG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Edit here: extra training examples

    Leave this list empty for the first feature experiment. Later, add an original synthetic ticket
    with a valid label to test a data hypothesis. Do not copy or paraphrase development/test tickets
    into training. Otherwise the model can memorize the evaluation instead of learning the task.
    """)
    return


@app.cell
def _():
    # Each entry needs text and one of the four supplied labels.
    # Example: {"text": "Is the wool hat available in green?", "label": "product_question"}
    EXTRA_TRAINING_EXAMPLES = []
    return (EXTRA_TRAINING_EXAMPLES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Run this: read your first result

    Initialization fits both models and fills the table below. If it is empty, run all stale cells.
    The table has one row for `baseline` and one for `candidate`. Keep this first result as your reference.
    With the supplied data and defaults, both currently get 12 of 16 development tickets right: accuracy `0.75`
    and macro-F1 about `0.7431`. Equal scores are a valid result; changing a setting need not help.

    **Accuracy** is correct predictions divided by all predictions: 12 ÷ 16 = 0.75.
    It is useful, but a model could do well on a common queue and ignore a rare one.
    **Macro-F1** checks each queue separately and averages their scores equally. For each queue, F1 balances
    “of the tickets sent here, how many belong here?” with “of the tickets that belong here, how many did we find?”
    Both scores range from 0 to 1; higher is better on these examples.

    1. Expand **Baseline mistakes**. Pick one row and read its `text`.
    2. Compare `label` (the supplied answer key) with `prediction` (the model's choice).
    3. Write down the ticket ID and why you think it belongs to the expected queue.
    4. Find the same ticket under **Candidate mistakes**. Its absence means the candidate got it right;
       its presence means it is still wrong.

    The **per-class scores** give each queue its own precision, recall, and F1. The **confusion matrix** groups
    all predictions: rows are expected queues and columns are predicted queues. Off-diagonal counts are mistakes.
    Start with one ticket before interpreting the whole matrix.
    """)
    return


@app.cell(hide_code=True)
def _(baseline_dev, candidate_dev, experiment_id, mo):
    mo.vstack(
        [
            mo.md(
                f"### Current comparison\n\nExperiment `{experiment_id}`. Higher accuracy and macro-F1 are better."
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


@app.cell(hide_code=True)
def _(baseline_dev, mo):
    _mistake = next(
        (row for row in baseline_dev["predictions"] if not row["correct"]), None
    )
    if _mistake is not None:
        mo.output.append(
            mo.md(f"""
        ### Work through one actual mistake

        Ticket **`{_mistake["id"]}`** says: **{_mistake["text"]}**

        The supplied queue is **`{_mistake["label"]}`**, but the baseline chose
        **`{_mistake["prediction"]}`**. This is one of the mistakes counted in its score.
        Before asking ARIA, explain in your own words why the supplied queue makes sense.
        The result alone does not tell us which words caused the mistake; ask ARIA to inspect
        the saved training data and modeling code rather than guessing a cause from the score.

        You can use this ticket ID in the investigation prompt in section 5.
        """)
        )
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

    ### Where your results will appear

    Use **Upload comparison to W&B** below only when you are ready to save the current comparison.
    It creates two runs and returns clickable links. Open a run to inspect its **Overview** (including configuration), summary scores,
    and **Artifacts** (saved data, source, model, and predictions). In the project workspace, select the baseline
    and candidate to compare `dev/accuracy` and `dev/macro_f1`. Exact failures are in the saved prediction evidence.

    This CPU lab has no language-model conversation or tool calls, so its evidence belongs in W&B runs and artifacts.
    It does not create Agent Lens conversations or Weave evaluation traces. No paid training is launched by this upload.
    """)
    return


@app.cell(hide_code=True)
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
        ## 4. Run this: upload evidence for ARIA

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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Use ARIA to investigate, then run one controlled edit

    **First, investigate.** Wait for the upload to say **complete**. Open one of the returned run links, go to that
    project, and select **Ask ARIA**. Copy the handoff text above; it contains the run links and evidence.
    Add: “Focus on ticket [the ID you chose]. Explain why it may be confused with the other queue.
    Use the saved training examples and settings. I am new to ML; explain any terms before suggesting one change.”
    If the upload failed, fix that first; ARIA cannot inspect evidence that never arrived.

    **Next, state a hypothesis.** A hypothesis is a prediction you can check. For this first experiment, try:
    “Using two-word phrases will help the classifier distinguish messages whose individual words occur in several queues.”
    You are testing that idea, not assuming it is true. Keep the original upload links for comparison.

    Return to **Edit here: candidate settings** and replace the dictionary with this exact first experiment:

    ```python
    CANDIDATE_CONFIG = {"ngram_range": (1, 2), "C": 1.0, "class_weight": None}
    ```

    The baseline uses `(1, 1)`, so only phrase length now differs between the models. Leave
    `EXTRA_TRAINING_EXAMPLES = []`. Run the edited cell with its play button and return to section 3.
    CPU fitting and predictions update locally. Click **Upload comparison to W&B** again to save this new version;
    a local edit alone is invisible to ARIA.

    **Finally, compare.** Read the targeted ticket, the two aggregate scores, and any newly wrong tickets.
    Use the new handoff in ARIA and include your hypothesis: “Did this change fix the targeted mistake?
    Which examples regressed? Do these runs support my hypothesis, or is the result inconclusive?”
    If nothing changes, that is evidence too. Do not add test examples or change a second setting to manufacture a win.

    For a second experiment, change only one of `C`, class weighting, or extra training examples.
    A useful finish is a sentence such as: “I changed __ because __; ticket __ changed from __ to __;
    the score changed from __ to __, but __ still failed.” Save your edited notebook and run links.
    """)
    return


@app.cell(hide_code=True)
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
                "## 6. Run this: final held-out test\n\nChoose your candidate using development results first. This button evaluates held-out scenarios. Once you inspect test errors, the test set is no longer an unbiased guide for further edits."
            ),
            final_test_button,
        ]
    )
    return (final_test_button,)


@app.cell(hide_code=True)
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
    return DATASET_SHA256, DATASET_VERSION, LABELS, TICKETS


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    get_final_result, set_final_result = mo.state(None)
    get_upload_result, set_upload_result = mo.state(None)
    return get_final_result, get_upload_result, set_final_result, set_upload_result


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


if __name__ == "__main__":
    app.run()
