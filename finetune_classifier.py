# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo==0.25.2",
#   "openai==2.54.0",
#   "openpipe-art==0.5.21",
#   "python-dotenv==1.2.4",
#   "scikit-learn==1.9.1",
#   "wandb==0.31.0",
# ]
# ///
# Adapted from OpenPipe/art-notebooks, Apache-2.0, Copyright 2025 OpenPipe Inc.
# https://github.com/OpenPipe/art-notebooks/blob/74c819447bd1e03318369d30fc0acb0f1106d5b6/examples/sft/train_from_file.ipynb
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Changes: synthetic ticket classification, explicit submissions, durable job
# records, and comparison with a CPU baseline. See licenses/art-notebooks.txt.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import asyncio
    import hashlib
    import inspect
    import json
    import os
    import re
    import textwrap
    import time
    import uuid
    from pathlib import Path

    import marimo as mo
    from dotenv import load_dotenv
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline

    # Local .env is optional; molab Secrets already populate the environment.
    _environment_loaded = load_dotenv(Path.cwd() / ".env", override=False)
    # ART must not discover a different ancestor .env when imported later.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    return (
        LogisticRegression,
        Path,
        TfidfVectorizer,
        accuracy_score,
        asyncio,
        confusion_matrix,
        f1_score,
        hashlib,
        inspect,
        json,
        make_pipeline,
        mo,
        os,
        re,
        time,
        textwrap,
        uuid,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Teach a language model to route support tickets

    **Trail Supply is a fictional outdoor shop.** Customers ask about deliveries, returns, and products; staff need
    to route each message to the right queue. A **ticket** is one customer message, and its **label** is the queue
    that should receive it. This notebook teaches a language model to output a queue name, not a customer reply.
    You can start here without having run either other notebook: all data and the CPU reference are included.

    A language model has already learned general text patterns. **Supervised fine-tuning (SFT)** shows it examples
    of the answers we want and updates its internal weights (the numbers controlling its predictions).
    This differs from editing a prompt, which changes instructions without changing those weights.
    We will compare the same model before and after that update, alongside a small local classifier.

    **What you will make:** one training job and a separate comparison of three classifiers on the same questions.
    The first small **smoke run** checks whether the services work. It is not enough data to judge which model is better.
    A full exercise uses the supplied training split after the smoke path works.

    **Before starting:** you need basic Python, your W&B team/project and API key, Forge serverless training access
    and credits, and ARIA access. Both training and model comparison use paid services and have separate submit buttons.
    Do the CPU notebook instead if those accounts are not ready; nothing in this file requires its output.

    ### Open and run this notebook

    If you are in molab, fork this notebook into your own workspace before editing.
    If you downloaded this `.py` file, open a terminal in its folder and run the command below.
    [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
    are available if your terminal says `uvx` is not found. The first launch installs the pinned packages.

    ```sh
    uvx marimo edit --sandbox finetune_classifier.py
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

    ## Understand the training task before submitting a job

    | Queue name in Python | What belongs here | Fictional example |
    | --- | --- | --- |
    | `order_status` | Tracking, dispatch, or delivery of a purchase | “Where is my order? Please send a tracking link.” |
    | `returns` | Returning an item, refunds, or exchanges | “How do I return this jacket?” |
    | `product_question` | Features, sizing, or availability before buying | “Is this backpack waterproof?” |
    | `human_escalation` | Safety, fraud, privacy, or an explicit request for a person | “Please let me speak to a human.” |

    All examples are synthetic. Of the 64 tickets, 32 are training examples, 16 are **development** examples for
    choosing changes, and 16 are **test** examples reserved for your final assessment. Related phrasings stay together
    in one split. Repeatedly editing until test examples pass makes them part of your development process.

    For training, each labeled ticket becomes a short conversation:

    ```text
    system: Choose exactly one of the four queue labels.
    user: Where is my order? Please send a tracking link.
    assistant: order_status
    ```

    `system` contains the instructions, `user` the input ticket, and `assistant` the desired answer. The upload stores
    one such record on each line of a **JSONL** file. Python prepares that file for you; you do not need to upload data
    manually or write JSON. At evaluation time, the expected answer is withheld and the model generates its own label.

    ### The three models in your comparison

    | Result name | How it gets its predictions | What it helps you check |
    | --- | --- | --- |
    | `cpu` | A local TF-IDF/logistic-regression classifier learns word patterns from all 32 training tickets | Whether a simpler, cheaper model already handles the task |
    | `step0` | The pretrained language model follows your prompt before this shop's fine-tuning | What the starting model already knows |
    | `trained` | The same language model after seeing the selected training tickets | What changed after the training update |

    A **checkpoint** is a saved version of the model's weights. Step 0 is already pretrained; it is not random or
    trained from scratch. A later step is a saved update. A completed job produces that checkpoint, but you still
    need a separate inference comparison to find out what it predicts.
    """)
    return


@app.cell(hide_code=True)
def _(LABELS, TICKETS, mo):
    mo.vstack(
        [
            mo.md(
                "### Preview the supplied training data\nRead a few tickets and their `label` values. `id` identifies an example; `family` groups related phrasings. The trainer will build the upload from these rows."
            ),
            mo.ui.table(
                [
                    {
                        "label": label,
                        "training_tickets": sum(
                            row["split"] == "train" and row["label"] == label
                            for row in TICKETS
                        ),
                    }
                    for label in LABELS
                ],
                selection=None,
            ),
            mo.ui.table(
                [row for row in TICKETS if row["split"] == "train"],
                selection=None,
                page_size=8,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Edit here: classification prompt

    The same prompt is used for training examples and checkpoint evaluation. The model must return exactly one
    of the four labels, with no explanation. Invalid labels count as wrong predictions; extra prose can therefore lower
    the score even if its meaning sounds right. Keep the prompt fixed for your first training run.
    After examining a development failure with ARIA, you can test one clarification here.
    """)
    return


@app.cell
def _(LABELS):
    # Edit this prompt, then submit a NEW experiment ID. The prompt used for
    # checkpoint 0 and trained-checkpoint evaluation is saved with the experiment.
    CLASSIFIER_PROMPT = (
        "Classify a ticket for a fictional outdoor shop. Reply with exactly one "
        "of these labels and no explanation: " + ", ".join(LABELS) + ". "
        "Use human_escalation for fraud, safety, privacy, accessibility, legal "
        "issues, or a request to speak to a person."
    )
    return (CLASSIFIER_PROMPT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Edit here: training settings

    - `base_model` selects the starting model; your team needs access to it. The default can be supplied through `FORGE_TRAINING_MODEL`.
    - `learning_rate` controls the size of weight updates. Larger updates can learn faster or damage useful behavior. Allowed: `1e-6` to `1e-3`.
    - `batch_size` is the number of examples the trainer groups for one update. Allowed: `2`, `4`, or `8`. Change it separately from learning rate.

    The submitted configuration is saved with the prompt and data. Recovery and checkpoint comparison use those saved
    values even if you later edit this cell. The exercise stays at **one pass** through at most **32 training tickets**.
    **Smoke: 2 tickets** checks the training connection; **Full: 32 tickets** runs the classification exercise.
    The two smoke tickets are not enough to judge model quality. Start with smoke and review access and costs with your facilitator.
    """)
    return


@app.cell
def _(os):
    # Edit here: keep defaults for the smoke run; change one setting per later hypothesis.
    SFT_CONFIG = {
        "base_model": os.getenv("FORGE_TRAINING_MODEL", "Qwen/Qwen3.8-27B"),
        "learning_rate": 2e-4,
        "batch_size": 2,
    }
    return (SFT_CONFIG,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Serving limitation before you spend credits

    The recorded live rehearsal completed training but the full checkpoint comparison encountered intermittent
    Forge HTTP 500 serving errors. This lab does **not** resolve that service issue. A complete training job is not
    proof that both checkpoints can currently serve predictions. See the [validation record](https://github.com/ash0ts/aria-starter/blob/main/validation.md) for the evidence.
    If serving fails, read the existing job status and consult your facilitator before spending credits on a retry;
    do not train another model just to repair an inference error.
    """)
    return


@app.cell(hide_code=True)
def _(SFT_CONFIG, mo):
    mo.md(f"**Selected training settings:** `{SFT_CONFIG}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Make the first run a connection check

    Keep the prompt and `SFT_CONFIG` unchanged. In the next form, keep **Smoke: 2 tickets** and the generated
    **Experiment ID**. Copy that ID somewhere before clicking **Submit paid training** once.
    The ID names this deliberate training attempt; it is also how you find it after closing the notebook.

    The smoke job uses the first two training tickets for one pass. **One pass** means each selected example is used
    once in the uploaded training sequence. Full mode uses all 32 training tickets, also for one pass.
    A **batch** groups examples for an update; the **learning rate** sets the update size. Leave these alone until
    you have a complete baseline comparison to investigate.

    The form below starts training only. It does not automatically run paid comparison requests.
    When training returns, read **Status**. `complete` with a `trained_step` greater than zero means a checkpoint is recorded.
    If the status is `training`, `needs_recovery`, or `checkpoint_pending`, use recovery below with the same ID.
    Do not create a new training ID simply because a page stopped waiting.
    """)
    return


@app.cell(hide_code=True)
def _(mo, uuid):
    # Keep ID generation independent of editable configuration: an edit must not
    # create a new submission identity for an already submitted form.
    suggested_experiment_id = "sft-" + uuid.uuid4().hex[:12]
    training_form = (
        mo.md("""
    ## 3. Run this: submit training

    Set `WANDB_API_KEY`, `WANDB_ENTITY`, and `WANDB_PROJECT` in molab Secrets or
    an ignored `.env`. Your organizer must enable training and credits.
    A new experiment ID is required for each deliberate training attempt.

    {experiment_id}
    {mode}

    Submitting creates one training job. Editing code or rerunning a submitted
    ID does not submit it again. Keep the experiment ID for recovery.

    If access fails, check the key and team/project permissions with your
    organizer. If a model is unavailable or credits are exhausted, have the
    organizer resolve that before starting another ID. A timeout does not mean
    the remote job stopped; use recovery first.
    """)
        .batch(
            experiment_id=mo.ui.text(
                value=suggested_experiment_id, label="Experiment ID"
            ),
            mode=mo.ui.dropdown(
                ["Smoke: 2 tickets", "Full: 32 tickets"],
                value="Smoke: 2 tickets",
                label="Training size",
            ),
        )
        .form(submit_button_label="Submit paid training")
    )
    training_form
    return suggested_experiment_id, training_form


@app.cell(hide_code=True)
async def _(
    CLASSIFIER_PROMPT,
    SFT_CONFIG,
    DATASET_SHA256,
    DATASET_VERSION,
    TICKETS,
    mo,
    source_snapshot,
    train_experiment,
    training_form,
):
    mo.stop(training_form.value is None, mo.md("No training submitted."))
    training_result = await train_experiment(
        training_form.value,
        TICKETS,
        DATASET_VERSION,
        DATASET_SHA256,
        CLASSIFIER_PROMPT,
        source_snapshot,
        dict(SFT_CONFIG),
    )
    mo.vstack(
        [
            mo.md(
                f"**Status:** {training_result['status']}  \nExperiment: `{training_result['experiment_id']}`\n\n[Open the W&B training record]({training_result.get('run_url', '')}). Save this link and experiment ID for recovery."
            ),
            mo.json(training_result),
        ]
    )
    return (training_result,)


@app.cell(hide_code=True)
def _(mo, suggested_experiment_id):
    recovery_form = (
        mo.md("""
    ## Recover an interrupted run

    The record is stored in W&B and `.aria-starter/`. molab may discard local
    files at shutdown. Paste the original experiment ID to read its job status.
    Recovery never submits training. If an ID is missing after a network failure,
    ask the facilitator to inspect the run before creating another experiment.

    {experiment_id}
    """)
        .batch(
            experiment_id=mo.ui.text(
                value=suggested_experiment_id, label="Existing experiment ID"
            )
        )
        .form(submit_button_label="Read existing job status")
    )
    recovery_form
    return (recovery_form,)


@app.cell(hide_code=True)
async def _(mo, recover_experiment, recovery_form):
    mo.stop(recovery_form.value is None)
    recovered_result = await recover_experiment(recovery_form.value["experiment_id"])
    mo.vstack(
        [
            mo.md(
                f"**Recovered status:** {recovered_result['status']}. [W&B training record]({recovered_result.get('run_url', '')}). This read did not start new training."
            ),
            mo.json(recovered_result),
        ]
    )
    return (recovered_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Find the training evidence and recover safely

    The training result includes `run_url`, `experiment_id`, `job_id`, `model_id`, and checkpoint names.
    `run_url` opens the W&B record; its input artifact contains `train.jsonl`, the dataset, and experiment source.
    Keep the experiment ID and link. The large JSON result is a saved receipt, not Python you need to edit.

    If the notebook closes or a request times out, paste the **original experiment ID** into **Read existing job status**.
    It reads the existing remote job and updates the receipt. It never starts another training job.
    If the status says `rejected`, inspect permissions/model access/credits with your facilitator.
    If no job ID was received, the submission's outcome may be uncertain; inspect the existing run before trying a new ID.

    Only proceed to comparison when the receipt says `complete` and shows a trained checkpoint.
    Changing `SFT_CONFIG` later does not change this saved model. A different training experiment requires a new ID.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Read the comparison metrics

    **Accuracy** is the fraction of correct labels. **Macro-F1** gives each of the four labels equal weight;
    F1 balances finding that label's tickets with avoiding false matches. A **confusion matrix** shows expected
    labels in rows and predictions in columns. Invalid model outputs count as failures.
    For example, getting 12 out of 16 tickets right gives accuracy 0.75. Macro-F1 prevents a common queue
    from hiding a neglected queue: it scores each queue's ability to find its tickets without misrouting
    others, then averages those four scores. Both metrics range from 0 to 1. Check individual tickets too.
    Compare identical tickets across all three models and read individual predictions before drawing conclusions.
    """)
    return


@app.cell(hide_code=True)
def _(mo, suggested_experiment_id, uuid):
    evaluation_form = (
        mo.md("""
    ## 4. Run this: compare on development tickets

    Copy the successful training **Experiment ID** into **Completed experiment ID**.
    Keep the generated **Evaluation ID** and the split `dev`. Click **Submit paid comparison** once.
    The evaluation ID names this inference attempt; it is different from the training ID.
    If you typed a custom training ID, replace the form's suggested ID with that exact value.

    Smoke evaluation uses the same two development tickets for all three models.
    Both smoke tickets are order-status questions. Macro-F1 still averages all
    four labels, with zero for absent classes; these smoke scores are a connection
    check, not a quality estimate. Full evaluation uses all 16 tickets in the
    chosen split. It makes at most
    32 inference requests, each limited to 512 completion tokens, with retries off.
    The CPU model always uses the full training split.

    Use `dev` while iterating; choose `test` only for your final comparison.
    Each evaluation ID is submitted once. Use a new ID to deliberately retry.

    {experiment_id}
    {evaluation_id}
    {split}
    """)
        .batch(
            experiment_id=mo.ui.text(
                value=suggested_experiment_id, label="Completed experiment ID"
            ),
            evaluation_id=mo.ui.text(
                value="eval-" + uuid.uuid4().hex[:12], label="Evaluation ID"
            ),
            split=mo.ui.dropdown(
                ["dev", "test"], value="dev", label="Evaluation split"
            ),
        )
        .form(submit_button_label="Submit paid comparison")
    )
    evaluation_form
    return (evaluation_form,)


@app.cell(hide_code=True)
async def _(
    DATASET_SHA256, TICKETS, compare_checkpoints, evaluation_form, mo, source_snapshot
):
    mo.stop(evaluation_form.value is None, mo.md("No comparison submitted."))
    comparison = await compare_checkpoints(
        evaluation_form.value, TICKETS, DATASET_SHA256, source_snapshot
    )
    _comparison_notice = comparison.get("notice", "")
    if comparison.get("http_status") == 500:
        _comparison_notice += (
            " The checkpoint may still be becoming available. Refresh the existing "
            "job status, wait, then submit a new evaluation ID if you choose to "
            "retry. Inference requests are never repeated automatically."
        )
    mo.vstack(
        [
            mo.md(
                f"**Comparison status:** {comparison['status']}\n\n{_comparison_notice}"
            ),
            mo.ui.table(
                [
                    {
                        "model": name,
                        **{
                            key: result[key]
                            for key in ("accuracy", "macro_f1", "invalid_labels")
                        },
                    }
                    for name, result in comparison["results"].items()
                ]
            ),
            mo.accordion(
                {
                    name: mo.vstack(
                        [
                            mo.md("**Predictions**"),
                            mo.ui.table(result["predictions"], selection=None),
                            mo.md(
                                "**Confusion matrix**: rows are expected labels; columns are predictions."
                            ),
                            mo.ui.table(
                                [
                                    {
                                        "expected": label,
                                        **dict(
                                            zip(
                                                result["confusion_labels"],
                                                counts,
                                                strict=True,
                                            )
                                        ),
                                    }
                                    for label, counts in zip(
                                        result["confusion_labels"],
                                        result["confusion_matrix"],
                                        strict=True,
                                    )
                                ],
                                selection=None,
                            ),
                        ]
                    )
                    for name, result in comparison["results"].items()
                }
            ),
        ]
    )
    return (comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Read the comparison before asking for a change

    A completed comparison has three rows: `cpu`, `step0`, and `trained`. Expand a model's prediction table and
    read the same ticket across all three. `label` is the supplied label, `prediction` is the answer returned,
    and `correct` is whether they match. An explanation such as “This is an order-status question” is not the exact
    label `order_status`; it counts as invalid even if its meaning is understandable.

    Smoke uses two `order_status` questions. Even if both are correct, macro-F1 is only 0.25 because this calculation
    averages four labels and the other three have no examples. That is a property of this connection check,
    not evidence the model failed three queues. Full mode evaluates all 16 development tickets across all labels.

    If only the CPU or step-0 row is present, the comparison is incomplete. A serving error is not a wrong prediction.
    Use the status and existing job record to diagnose it; do not claim the missing trained model is better or worse.
    After your facilitator resolves access/serving, a new evaluation ID explicitly retries inference against the same
    training experiment. It does not require training again.

    The **Evaluation run** link below opens W&B. Inspect its saved settings and the results artifact for raw outputs
    and scores. This notebook records W&B runs and artifacts; it does not create Agent Lens conversations or Weave
    agent/evaluation traces. ARIA should be given the training and evaluation run evidence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Inspect and ask ARIA

    After a completed development comparison, open one model's prediction table and
    choose a failed ticket. The handoff below includes the saved configuration and
    result IDs. Open Ask ARIA from your target project and ask it to explain that
    failure using the uploaded evidence. For an incomplete run, inspect its status
    and recover the existing job before asking about model quality.
    """)
    return


@app.cell(hide_code=True)
def _(comparison, json, mo):
    failures = {
        name: [row for row in result["predictions"] if not row["correct"]]
        for name, result in comparison["results"].items()
    }
    aria_prompt = f"""Help me investigate a fictional outdoor-shop ticket classifier.
Goal: compare CPU TF-IDF/logistic regression, untrained checkpoint 0, and SFT.
Project: {comparison["settings"]["entity"]}/{comparison["settings"]["project"]}
Training experiment: {comparison["experiment_id"]}
Evaluation: {comparison["evaluation_id"]} ({comparison["split"]}, {comparison["mode"]})
Evaluation status: {comparison["status"]}; completed models: {", ".join(comparison["results"])}
Dataset: {comparison["dataset_version"]} / {comparison["dataset_sha256"]}
Trained step: {comparison["trained_step"]}
Saved training configuration: {json.dumps(comparison.get("training_config", {}))}
Exact modeling code is in the inputs artifact's experiment.py; the prompt is:
{comparison["prompt"]}
Failures: {json.dumps(failures)}
Investigate one failed development example using the saved runs. Help me state
a one-sentence hypothesis for one edit to CLASSIFIER_PROMPT or SFT_CONFIG.
Hold other settings constant. For my next comparison, inspect the targeted
example, macro-F1, and regressions: does the evidence support the hypothesis?
Do not assume SFT improved performance. I will edit the named cell in marimo,
submit a new training experiment ID, then explicitly submit a development comparison.
"""
    if comparison["split"] == "test":
        aria_prompt = (
            f"Interpret my final held-out comparison {comparison['evaluation_id']}: "
            f"{comparison.get('run_url', '')}. Training: {comparison['experiment_id']}. "
            f"Saved settings: {json.dumps(comparison.get('training_config', {}))}. "
            "Explain the observed scores and limitations using the saved predictions. "
            "Do not use test mistakes to select another change; any later edit would be test-informed."
        )
    if comparison["status"] != "complete":
        aria_prompt = (
            f"Help diagnose incomplete evaluation {comparison['evaluation_id']} for training "
            f"{comparison['experiment_id']}. Status: {comparison['status']}; "
            f"HTTP status: {comparison.get('http_status')}. Inspect its saved run "
            f"{comparison.get('run_url', '')} and checkpoint availability before model-quality advice. "
            "I will read the existing job status first. Do not submit training or inference."
        )
    mo.vstack(
        [
            mo.md(
                f"### Copy the ARIA handoff\n\nOpen Ask ARIA in your target project. [Evaluation run]({comparison.get('run_url', '')}). Copy this prompt. For a completed development comparison, follow the change-and-compare checkpoint below. For incomplete work, follow recovery guidance first."
            ),
            mo.ui.text_area(
                value=aria_prompt, label="Copy to ARIA", full_width=True, rows=14
            ),
        ]
    )
    return (aria_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Change one choice and compare again

    **Investigate:** use the handoff above after a completed development comparison. Choose one failed ticket and ask
    ARIA to point to the prompt, prediction, and saved training configuration. Smoke scores are only a connection check.
    For incomplete runs, use **Read existing job status** and the recovery guidance before requesting model-quality advice.

    **Change:** state a one-sentence hypothesis. For example, “Explicitly telling the model that safety reports
    require `human_escalation` will reduce safety tickets routed to `returns`.” Only try that hypothesis if the
    development evidence shows that confusion. Add the instruction inside **CLASSIFIER_PROMPT**, or test one
    value in **SFT_CONFIG** for a separate experiment;
    hold the other settings, mode, and evaluation split constant. Return to section 3, enter a **new experiment ID**,
    and click **Submit paid training**. After training completes, enter that ID and a **new evaluation ID** in section 4
    and click **Submit paid comparison** with `dev`. An edit or reuse of an existing ID cannot submit a second training job.

    **Compare:** give ARIA both training/evaluation IDs and your hypothesis. Ask: “Did the targeted example change?
    Compare macro-F1, accuracy, invalid labels, and regressions on shared tickets. Does the evidence support my hypothesis?”
    The copyable prompt describes the current evaluation; keep the previous run link to compare iterations.

    **Moving from smoke to a learning exercise:** once smoke training and comparison both complete, select
    **Full: 32 tickets** with a new training ID and keep the original prompt/settings. This becomes your full-mode
    baseline. Its comparison uses 16 development tickets per model. Make your next prompt/configuration change
    against that full baseline, also in full mode, so training size is not another changing factor.

    ## 7. Final held-out test

    Only after choosing your candidate on development results, return to the comparison form and select `test` with
    a fresh evaluation ID. This is an explicit paid comparison. Once you inspect test mistakes, that split is no longer
    independent evidence for later edits. Label any subsequent changes as informed by the test set.
    Save your notebook, run links, hypothesis, and one example that explains what changed.
    """)
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
def _(LogisticRegression, TfidfVectorizer, make_pipeline):
    def fit_cpu_classifier(training_rows):
        """The same baseline as train_classifier.py; fit on training data only."""
        classifier = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 1), lowercase=True),
            LogisticRegression(C=1.0, max_iter=1000, random_state=7),
        )
        classifier.fit(
            [row["text"] for row in training_rows],
            [row["label"] for row in training_rows],
        )
        return classifier

    return (fit_cpu_classifier,)


@app.cell(hide_code=True)
def _(LABELS, accuracy_score, confusion_matrix, f1_score):
    def score_predictions(rows, raw_predictions):
        """Invalid labels remain incorrect and get their own confusion column."""
        if len(rows) != len(raw_predictions) or not rows:
            raise ValueError("Evaluation needs one prediction per nonempty ticket set.")
        predictions = [
            value.strip() if value.strip() in LABELS else "INVALID"
            for value in raw_predictions
        ]
        expected = [row["label"] for row in rows]
        classes = LABELS + ["INVALID"]
        return {
            "accuracy": float(accuracy_score(expected, predictions)),
            "macro_f1": float(
                f1_score(
                    expected,
                    predictions,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "invalid_labels": predictions.count("INVALID"),
            "confusion_labels": classes,
            "confusion_matrix": confusion_matrix(
                expected, predictions, labels=classes
            ).tolist(),
            "predictions": [
                {
                    **row,
                    "prediction": prediction,
                    "raw_prediction": raw,
                    "correct": prediction == row["label"],
                }
                for row, prediction, raw in zip(
                    rows, predictions, raw_predictions, strict=True
                )
            ],
        }

    return (score_predictions,)


@app.cell(hide_code=True)
def _(Path, hashlib, json, os, re):
    def connection_settings():
        """Read keys only at submission. Never put a key into a notebook cell."""
        required = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
        missing = [name for name in required if not os.getenv(name, "").strip()]
        if missing:
            raise ValueError(
                "Set these environment variables or molab Secrets: "
                + ", ".join(missing)
            )
        if os.getenv("WANDB_MODE", "online") != "online":
            raise ValueError(
                "Paid jobs require online W&B logging for durable recovery."
            )
        return {
            "entity": os.environ["WANDB_ENTITY"].strip(),
            "project": os.environ["WANDB_PROJECT"].strip(),
            "training_url": os.getenv(
                "WANDB_TRAINING_BASE_URL", "https://api.training.wandb.ai/v1"
            ).rstrip("/"),
        }

    def validate_identity(value):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,63}", value):
            raise ValueError(
                "Use 3-64 letters, numbers, hyphens, or underscores for the ID."
            )
        return value

    def record_path(settings, identity):
        destination = hashlib.sha256(
            f"{settings['entity']}/{settings['project']}".encode()
        ).hexdigest()[:16]
        directory = Path(".aria-starter") / destination
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{validate_identity(identity)}.json"

    def save_local(path, record):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(path)

    def claim_submission(settings, identity, record):
        """An exclusive file protects reruns, including a lost network response."""
        path = record_path(settings, identity)
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2)
            return path, None
        except FileExistsError:
            return path, json.loads(path.read_text(encoding="utf-8"))

    def failure_details(error):
        # Keep only the API's error message, never headers or the request object.
        detail = {"error_type": type(error).__name__}
        status = getattr(error, "status_code", None)
        if status is not None:
            detail["http_status"] = status
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            message = body.get("error", body.get("detail", body))
            if isinstance(message, dict):
                message = message.get("message", "")
            if isinstance(message, str) and message:
                key = os.getenv("WANDB_API_KEY", "")
                if key:
                    message = message.replace(key, "[redacted]")
                detail["service_message"] = message[:500]
        return detail

    return (
        claim_submission,
        connection_settings,
        failure_details,
        record_path,
        save_local,
        validate_identity,
    )


@app.cell(hide_code=True)
def _(Path, json, os, time):
    def reserve_remote(settings, identity, record, files):
        """resume='never' rejects the same ID after local/molab storage is lost."""
        import wandb

        with wandb.init(
            entity=settings["entity"],
            project=settings["project"],
            id=identity,
            name=identity,
            job_type="aria-starter-sft",
            resume="never",
            reinit="create_new",
            mode="online",
            config={"starter": "aria-university", "record": record},
            settings=wandb.Settings(disable_code=True, disable_git=True),
        ) as run:
            record["run_url"] = run.url
            # A scalar survives servers that flatten nested summary dictionaries.
            run.summary["recovery_record_json"] = json.dumps(record, sort_keys=True)
            metadata = {}
            if "train.jsonl" in files:
                metadata = {
                    "format": "jsonl",
                    "num_trajectories": len(files["train.jsonl"].strip().splitlines()),
                }
            artifact = wandb.Artifact(
                f"{identity}-inputs", type="dataset", metadata=metadata
            )
            for name, content in files.items():
                with artifact.new_file(name, mode="w") as stream:
                    stream.write(content)
            logged = run.log_artifact(artifact).wait()
            return f"wandb-artifact:///{settings['entity']}/{settings['project']}/{logged.name}"

    def publish_record(settings, identity, record):
        import wandb

        remote = wandb.Api(api_key=os.environ["WANDB_API_KEY"]).run(
            f"{settings['entity']}/{settings['project']}/{identity}"
        )
        # Write the recovery snapshot atomically; nested summaries can be flattened
        # into record.job_id etc. and leave stale fields from an earlier snapshot.
        remote.summary["recovery_record_json"] = json.dumps(record, sort_keys=True)
        remote.summary.update()

    def read_remote_record(settings, identity):
        import wandb

        for attempt in range(5):
            # A just-written summary can lag behind the successful update.
            # Each retry uses a fresh API instance and only reads existing data.
            remote = wandb.Api(api_key=os.environ["WANDB_API_KEY"]).run(
                f"{settings['entity']}/{settings['project']}/{identity}"
            )
            snapshot = remote.summary.get("recovery_record_json")
            if isinstance(snapshot, str):
                record = json.loads(snapshot)
                if not isinstance(record, dict):
                    raise ValueError(
                        "The saved recovery snapshot is not a JSON object."
                    )
                return record
            if attempt < 4:
                time.sleep(1)
        raise ValueError(
            "The recovery snapshot is not visible yet. Refresh the existing job "
            "status after a few seconds. Keep the local record and ask your "
            "facilitator to inspect it if this persists; do not submit another job."
        )

    def publish_results(settings, identity, record):
        import wandb

        with wandb.init(
            entity=settings["entity"],
            project=settings["project"],
            id=identity,
            resume="must",
            reinit="create_new",
            mode="online",
            settings=wandb.Settings(disable_code=True, disable_git=True),
        ) as run:
            # A scalar survives servers that flatten nested summary dictionaries.
            run.summary["recovery_record_json"] = json.dumps(record, sort_keys=True)
            for name, result in record.get("results", {}).items():
                run.log(
                    {
                        f"{name}/{key}": result[key]
                        for key in ("accuracy", "macro_f1", "invalid_labels")
                    }
                )
                run.log(
                    {
                        f"{name}/predictions": wandb.Table(
                            columns=["id", "text", "label", "prediction", "correct"],
                            data=[
                                [
                                    row[key]
                                    for key in (
                                        "id",
                                        "text",
                                        "label",
                                        "prediction",
                                        "correct",
                                    )
                                ]
                                for row in result["predictions"]
                            ],
                        )
                    }
                )
            artifact = wandb.Artifact(f"{identity}-results", type="evaluation")
            with artifact.new_file("results.json", mode="w") as stream:
                json.dump(record, stream, indent=2)
            run.log_artifact(artifact).wait()

    return publish_record, publish_results, read_remote_record, reserve_remote


@app.cell(hide_code=True)
def _(asyncio, time):
    async def wait_for_training(client, job_id, seconds=1200):
        """Only poll an existing job. A timeout never creates another job."""
        deadline = time.monotonic() + seconds
        after = None
        while time.monotonic() < deadline:
            query = {"training_job_id": job_id}
            if after:
                query["after"] = after
            async for event in client.sft_training_jobs.events.list(**query):
                after = event.id
                if event.type == "training_ended":
                    return "complete"
                if event.type == "training_failed":
                    return "remote_failed"
            await asyncio.sleep(2)
        return "pending"

    return (wait_for_training,)


@app.cell(hide_code=True)
def _(
    claim_submission,
    connection_settings,
    failure_details,
    json,
    os,
    publish_record,
    reserve_remote,
    save_local,
    validate_identity,
    wait_for_training,
    validate_sft_config,
):
    async def train_experiment(
        submission, tickets, version, checksum, prompt, source, configuration
    ):
        """One explicit submission, one remote job, and a durable recovery record."""
        effective = validate_sft_config(configuration)
        settings = connection_settings()
        identity = validate_identity(submission["experiment_id"].strip())
        model_name = identity + "-model"
        if submission["mode"] not in ("Smoke: 2 tickets", "Full: 32 tickets"):
            raise ValueError("Choose a smoke or full run.")
        base_model = effective["base_model"]
        rows = [dict(row) for row in tickets if row["split"] == "train"]
        if submission["mode"].startswith("Smoke"):
            rows = rows[:2]
        if len(rows) > 32 or len(rows) < 2:
            raise ValueError("This exercise supports 2-32 training tickets.")
        record = {
            "experiment_id": identity,
            "model_name": model_name,
            "base_model": base_model,
            "settings": settings,
            "dataset_version": version,
            "dataset_sha256": checksum,
            "prompt": prompt,
            "training_rows": rows,
            "epochs": 1,
            "batch_size": effective["batch_size"],
            "learning_rate": effective["learning_rate"],
            "training_config": dict(effective),
            "mode": submission["mode"],
            "status": "reserved",
        }
        path, previous = claim_submission(settings, identity, record)
        if previous is not None:
            return {
                **previous,
                "notice": "This ID was already submitted. Use recovery; a rerun does not train again.",
            }
        backend = client = model = None
        remote_reserved = False
        try:
            import art
            from art.serverless.backend import ServerlessBackend
            from art.serverless.client import Client

            lines = [
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": row["text"]},
                            {"role": "assistant", "content": row["label"]},
                        ]
                    }
                )
                for row in rows
            ]
            record["training_data_url"] = reserve_remote(
                settings,
                identity,
                record,
                {
                    "train.jsonl": "\n".join(lines) + "\n",
                    "dataset.json": json.dumps(tickets, indent=2),
                    "experiment.py": source
                    + "\n# Effective values captured at submission.\nSFT_CONFIG = "
                    + repr(effective)
                    + "\nCLASSIFIER_PROMPT = "
                    + repr(prompt)
                    + "\n",
                },
            )
            remote_reserved = True
            save_local(path, record)
            backend = ServerlessBackend(
                api_key=os.environ["WANDB_API_KEY"], base_url=settings["training_url"]
            )
            client = Client(
                api_key=os.environ["WANDB_API_KEY"], base_url=settings["training_url"]
            )
            model = art.TrainableModel(
                name=model_name,
                run_name=model_name,
                entity=settings["entity"],
                project=settings["project"],
                base_model=base_model,
            )
            await model.register(backend)
            record.update(
                {
                    "model_id": model.id,
                    "model_run_id": model.run_id,
                    "step0_name": model.get_inference_name(step=0),
                    "status": "submitting",
                }
            )
            save_local(path, record)
            # This synchronous update must succeed before any training request.
            publish_record(settings, identity, record)
            # ART's convenience train_sft_from_file hides the returned job ID.
            # Its underlying client allows saving that ID before waiting.
            job = await client.sft_training_jobs.create(
                model_id=model.id,
                training_data_url=record["training_data_url"],
                config={
                    "batch_size": record["batch_size"],
                    "learning_rate": record["learning_rate"],
                    "assistant_turns": "last",
                },
            )
            record.update({"job_id": job.id, "status": "training"})
            save_local(path, record)
            publish_record(settings, identity, record)
            record["status"] = await wait_for_training(client, job.id)
            if record["status"] == "complete":
                record["trained_step"] = await model.get_step()
                record["trained_name"] = model.get_inference_name(
                    step=record["trained_step"]
                )
                if record["trained_step"] <= 0:
                    record["status"] = "checkpoint_pending"
            save_local(path, record)
            publish_record(settings, identity, record)
        except BaseException as error:
            record.update({"status": "needs_recovery", **failure_details(error)})
            if record.get("http_status") in (
                400,
                401,
                403,
                404,
                422,
            ) and not record.get("job_id"):
                record["status"] = "rejected"
            save_local(path, record)
            if remote_reserved:
                try:
                    publish_record(settings, identity, record)
                except Exception:
                    pass  # Local recovery record still contains every received ID.
            if not isinstance(error, Exception):
                raise
        finally:
            # The direct job client normally leaves this unset. Close it if
            # a future ART version creates a local model metrics run.
            if model is not None and model._wandb_run is not None:
                model._wandb_run.finish(
                    exit_code=0 if record["status"] == "complete" else 1
                )
            if client is not None:
                await client.close()
            if backend is not None:
                await backend.close()
        return record

    return (train_experiment,)


@app.cell(hide_code=True)
def _(
    connection_settings,
    os,
    publish_record,
    read_remote_record,
    record_path,
    save_local,
    validate_identity,
    wait_for_training,
):
    async def recover_experiment(identity):
        """Read the known job and checkpoints. Never register or submit training."""
        from art.serverless.client import Client

        settings = connection_settings()
        identity = validate_identity(identity.strip())
        path = record_path(settings, identity)
        # Remote storage survives a molab restart; the local record may be newer
        # if the last upload failed after the service returned a job ID.
        record = read_remote_record(settings, identity)
        if path.exists():
            import json

            local = json.loads(path.read_text())
            if local.get("job_id"):
                record.update(local)
        if record.get("status") == "rejected":
            return {
                **record,
                "notice": "The service rejected this request. Resolve the saved error before submitting a new experiment ID.",
            }
        if not record.get("model_id") or not record.get("job_id"):
            return {
                **record,
                "notice": "Submission outcome is uncertain. Inspect the model run with your facilitator; do not resubmit this ID.",
            }
        client = Client(
            api_key=os.environ["WANDB_API_KEY"],
            base_url=record["settings"]["training_url"],
        )
        try:
            record["status"] = await wait_for_training(
                client, record["job_id"], seconds=5
            )
            if record["status"] == "complete":
                async for checkpoint in client.models.checkpoints.list(
                    model_id=record["model_id"], order="desc", limit=1
                ):
                    record["trained_step"] = checkpoint.step
                    record["trained_name"] = (
                        record["step0_name"].rsplit(":step", 1)[0]
                        + f":step{checkpoint.step}"
                    )
                    break
                if record.get("trained_step", 0) <= 0:
                    record["status"] = "checkpoint_pending"
            save_local(path, record)
            publish_record(settings, identity, record)
            return record
        finally:
            await client.close()

    return (recover_experiment,)


@app.cell(hide_code=True)
def _(
    claim_submission,
    connection_settings,
    failure_details,
    fit_cpu_classifier,
    json,
    os,
    publish_record,
    publish_results,
    read_remote_record,
    reserve_remote,
    save_local,
    score_predictions,
    validate_identity,
):
    async def compare_checkpoints(submission, tickets, checksum, source):
        settings = connection_settings()
        identity = validate_identity(submission["evaluation_id"].strip())
        experiment_id = validate_identity(submission["experiment_id"].strip())
        training = read_remote_record(settings, experiment_id)
        if training.get("status") != "complete" or training.get("trained_step", 0) <= 0:
            raise ValueError(
                "Training must be complete. Use recovery to refresh its record."
            )
        if training["dataset_sha256"] != checksum:
            raise ValueError(
                "Dataset changed since training. Compare using the saved dataset or train a new experiment."
            )
        if submission["split"] not in ("dev", "test"):
            raise ValueError("Choose development or final test tickets.")
        rows = [dict(row) for row in tickets if row["split"] == submission["split"]]
        if training["mode"].startswith("Smoke"):
            rows = rows[:2]
        if not 1 <= len(rows) <= 16:
            raise ValueError("Evaluate 1-16 tickets per checkpoint.")
        record = {
            "evaluation_id": identity,
            "experiment_id": experiment_id,
            "settings": settings,
            "dataset_sha256": checksum,
            "dataset_version": training["dataset_version"],
            "split": submission["split"],
            "mode": training["mode"],
            "training_run_url": training.get("run_url"),
            "model_id": training["model_id"],
            "trained_step": training["trained_step"],
            "prompt": training["prompt"],
            "training_config": training.get(
                "training_config",
                {
                    key: training.get(key)
                    for key in ("base_model", "learning_rate", "batch_size")
                },
            ),
            "status": "reserved",
            "results": {},
            "completion_token_limit": 512,
        }
        path, previous = claim_submission(settings, identity, record)
        if previous is not None:
            return {
                **previous,
                "notice": "This evaluation ID was already submitted. No requests were repeated.",
            }
        remote_reserved = False
        try:
            from openai import AsyncOpenAI

            reserve_remote(
                settings,
                identity,
                record,
                {
                    "evaluation.json": json.dumps(rows, indent=2),
                    "experiment.py": source
                    + "\n# Evaluate the saved training experiment, even after notebook edits.\nSFT_CONFIG = "
                    + repr(record["training_config"])
                    + "\nCLASSIFIER_PROMPT = "
                    + repr(record["prompt"])
                    + "\n",
                },
            )
            remote_reserved = True
            cpu = fit_cpu_classifier(
                [row for row in tickets if row["split"] == "train"]
            )
            record["results"]["cpu"] = score_predictions(
                rows, list(cpu.predict([row["text"] for row in rows]))
            )
            record["status"] = "evaluating"
            save_local(path, record)
            publish_record(settings, identity, record)
            # Trained models use the ART training endpoint, not inference.wandb.ai.
            async with AsyncOpenAI(
                base_url=training["settings"]["training_url"],
                api_key=os.environ["WANDB_API_KEY"],
                project=f"{settings['entity']}/{settings['project']}",
                max_retries=0,
                timeout=90,
            ) as client:
                for name, checkpoint in (
                    ("step0", training["step0_name"]),
                    ("trained", training["trained_name"]),
                ):
                    predictions = []
                    for row in rows:
                        completion = await client.chat.completions.create(
                            model=checkpoint,
                            messages=[
                                {"role": "system", "content": training["prompt"]},
                                {"role": "user", "content": row["text"]},
                            ],
                            max_completion_tokens=512,
                            temperature=0,
                        )
                        predictions.append(completion.choices[0].message.content or "")
                        record["partial_predictions"] = {
                            "checkpoint": name,
                            "predictions": predictions,
                        }
                        save_local(path, record)
                        publish_record(settings, identity, record)
                    record["results"][name] = score_predictions(rows, predictions)
                    save_local(path, record)
                    publish_record(settings, identity, record)
            record["status"] = "complete"
            record.pop("partial_predictions", None)
            publish_results(settings, identity, record)
        except BaseException as error:
            record.update({"status": "incomplete", **failure_details(error)})
            if remote_reserved:
                try:
                    publish_results(settings, identity, record)
                except Exception:
                    pass
            if not isinstance(error, Exception):
                raise
        finally:
            save_local(path, record)
        return record

    return (compare_checkpoints,)


@app.cell(hide_code=True)
def _(
    CLASSIFIER_PROMPT,
    SFT_CONFIG,
    compare_checkpoints,
    fit_cpu_classifier,
    inspect,
    score_predictions,
    textwrap,
    train_experiment,
):
    # Log the editable modeling choices, not environment variables or secrets.
    source_snapshot = (
        "CLASSIFIER_PROMPT = "
        + repr(CLASSIFIER_PROMPT)
        + "\nSFT_CONFIG = "
        + repr(SFT_CONFIG)
        + "\n\n"
        + textwrap.dedent(inspect.getsource(fit_cpu_classifier))
        + "\n\n"
        + textwrap.dedent(inspect.getsource(score_predictions))
        + "\n\n"
        + textwrap.dedent(inspect.getsource(train_experiment))
        + "\n\n"
        + textwrap.dedent(inspect.getsource(compare_checkpoints))
    )
    return (source_snapshot,)


@app.cell(hide_code=True)
def _():
    def validate_sft_config(configuration):
        """Freeze bounded teaching settings before reserving or submitting work."""
        model = configuration.get("base_model")
        rate = configuration.get("learning_rate")
        batch = configuration.get("batch_size")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("SFT_CONFIG.base_model must name a Forge training model.")
        if type(rate) not in (int, float) or not 1e-6 <= rate <= 1e-3:
            raise ValueError("SFT_CONFIG.learning_rate must be between 1e-6 and 1e-3.")
        if type(batch) is not int or batch not in (2, 4, 8):
            raise ValueError("SFT_CONFIG.batch_size must be 2, 4, or 8.")
        return {"base_model": model.strip(), "learning_rate": rate, "batch_size": batch}

    return (validate_sft_config,)


if __name__ == "__main__":
    app.run()
