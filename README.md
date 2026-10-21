# Build, train, and improve a support desk with ARIA

Start with a fictional outdoor shop, supplied tickets, and working Python code.
Choose an activity, inspect a mistake, ask ARIA for an explanation, then change
the code and compare the results. You need basic Python familiarity, but no
existing ML project or dataset.

| Activity | Open in molab | Python source | Try changing |
| --- | --- | --- | --- |
| Train a ticket classifier, no account required | [Open classifier](https://molab.marimo.io/github/ash0ts/aria-starter/blob/main/train_classifier.py) | [train_classifier.py](train_classifier.py) | Training examples, text features, or model settings |
| Build a support assistant with Forge Inference and Agent Lens | [Open agent](https://molab.marimo.io/github/ash0ts/aria-starter/blob/main/build_agent.py) | [build_agent.py](build_agent.py) | The prompt, order and policy tools, or escalation behavior |
| Fine-tune a language model with Forge serverless training | [Open fine-tuning](https://molab.marimo.io/github/ash0ts/aria-starter/blob/main/finetune_classifier.py) | [finetune_classifier.py](finetune_classifier.py) | Examples or training settings, then compare against the CPU classifier |

All tickets, orders, policies, and customers are synthetic. The agent's tools
simulate support actions; they do not contact customers or change real orders.

## Open a notebook

**In your browser:** use an Open in molab link above, then fork the notebook into
your own molab workspace. The Python cells are editable. Each fork needs your
own credentials for connected activities. If outputs are waiting, open marimo's
command palette and choose **Run all stale cells** (or **Re-run all cells**).
This initializes the lab; it does not upload results or submit paid work.

**On your computer:** install [uv](https://docs.astral.sh/uv/getting-started/installation/),
open a terminal in this directory, and run:

```sh
uvx marimo edit --sandbox train_classifier.py
```

Replace the filename with `build_agent.py` or `finetune_classifier.py` to switch
activities. Each notebook includes its dataset and pinned dependencies, so it
can run on its own. The first launch downloads its Python packages.

## Connect to Forge and ARIA

Try the CPU classifier first if your account is not ready. To upload results,
use the agent, or fine-tune a model, your facilitator supplies a W&B team and a
project name just for you. Obtain your own API key through your account settings.

| Variable | Value |
| --- | --- |
| `WANDB_ENTITY` | The team name from your facilitator |
| `WANDB_PROJECT` | Your assigned project name |
| `WANDB_API_KEY` | Your own API key |

In molab, add these values in the notebook sidebar's **Secrets** panel. Locally,
copy [.env.example](.env.example) to `.env`, fill in the values, and launch with:

```sh
uvx --env-file .env marimo edit --sandbox train_classifier.py
```

Keep keys in Secrets or `.env`; do not paste them into Python cells. Your `.env`
is ignored by Git, and [molab does not include it when a notebook is forked](https://marimo.io/pages/molab/storage).

ARIA requires a team project on W&B Multi-tenant Cloud and an organization with
Smart features enabled. Inference and training also require access and credits.
Your facilitator should provision these before the activity. See the
[ARIA prerequisites](https://docs.coreweave.com/products/aria).

## Follow the guided lab

Each notebook has numbered sections and three cell labels: **Edit here** for
experiment choices, **Run this** for actions and results, and **Supplied
implementation** for provided code. Expand collapsed cells with **Show code**
(the code icon or cell menu in the editor) to inspect fixtures, scoring, logging,
and recovery. Those helpers remain ordinary Python.

| Notebook | Visible edit points | First controlled change |
| --- | --- | --- |
| CPU classifier | `CANDIDATE_CONFIG`, `EXTRA_TRAINING_EXAMPLES` | Copy the fixed baseline values, then change only `ngram_range` to `(1, 2)`. The supplied candidate also changes `C`, so reset it to `1.0` for this experiment. |
| Support agent | `AGENT_PROMPT`, `AGENT_CONFIG`, optional tools and schemas | Clarify what to do when an order ID is missing; hold inference settings and scenarios constant. |
| Optional SFT | `CLASSIFIER_PROMPT`, `SFT_CONFIG` | After a successful comparison, test one prompt clarification or one training setting. |

The configuration cells are the source of truth for model and training settings.
Submission forms show the selected settings and collect experiment identifiers
and evaluation choices. Editing CPU choices recomputes local results. Uploads,
agent inference, training, and checkpoint comparisons each require their own
explicit action. Recovery uses the original saved configuration after edits.

## Run an experiment

1. Run the baseline and read a failed prediction or conversation. The classifier
   uses development examples by default. Paid agent inference and fine-tuning
   require an explicit submission in the notebook.
2. Upload the comparison when offered, then open its project link and click
   **Ask ARIA**. Paste the notebook's handoff prompt. It identifies the runs,
   relevant code, and mistakes for ARIA to investigate.
3. State a one-sentence hypothesis and ask for one change you can explain. For example: “Why does this model confuse
   returns with escalation? Point to an example and suggest one change to my
   Python code.”
4. Apply the change in a visible Python cell and run the comparison again. Read
   the examples as well as the aggregate score. Record what improved and what
   became worse. Ask ARIA whether the evidence supports your hypothesis.
5. For classification, finish experimenting before evaluating the held-out test
   examples once. Keep your notebook and result links for the discussion.

The CPU notebook's **Upload comparison to W&B** button saves a baseline and
candidate run. In the agent notebook, use **Run agent experiment**, name your
first run `baseline`, then edit the code and name the next run `candidate`.

The agent exercise scores tool behavior and structured outcomes; read the
answer to check whether it helps the customer. Classifiers report accuracy,
macro-F1, and per-label mistakes. An invalid fine-tuned label counts as a
failure. These small fixtures teach comparison; they do not establish readiness
for a real support desk. A useful experiment can make the score worse.

The fine-tuning notebook starts in **Smoke: 2 tickets** mode to check the
connection and training path. Select **Submit paid training**, then **Submit
paid comparison** to compare the CPU baseline, untrained checkpoint, and trained
checkpoint. Use **Full: 32 tickets** only after reviewing the settings with your
facilitator. Save the experiment ID before leaving a session. After a failure,
use **Read existing job status** with that ID before considering a new attempt.

The current live validation completed SFT training but encountered intermittent
HTTP 500 errors while serving the checkpoint comparison. The notebook calls out
this limitation before submission. See the [validation record](validation.md);
inspect incomplete runs and recover their status before asking ARIA about model
quality or choosing a paid retry.

## Keep your work

Download your edited notebook from molab. Connected activities store results
and artifacts in W&B; local scratch files are not a durable archive on molab.
The notebooks provide context to ARIA through their project and result links.
You review ARIA's suggestions and make edits yourself in marimo.

For the discussion, bring your baseline and changed result links, the code you
changed, and an example that explains the difference. Tell us separately what
made setup difficult and where ARIA helped or left you stuck.

See [facilitator notes](facilitator.md) for provisioning and pilot checks, and
[sources](sources.md) for the examples adapted here and their license notices.
The [validation record](validation.md) distinguishes completed checks from
the remaining classroom checks.

The starter is licensed under [Apache-2.0](LICENSE), with retained upstream
notices in [licenses](licenses).
