# Facilitating the ARIA support-desk activity

Students start with supplied code and synthetic data. The learning objective is
to explain a failure, make a deliberate change, and compare its effects. Do not
grade success by whether a metric increases or require an existing ML project.
For students who are still learning Python, start with the CPU classifier and
pair them with someone who can help edit a list or dictionary.

## Prepare accounts

- Give each participant a project within a W&B team. A personal entity does not
  meet [ARIA's prerequisites](https://docs.coreweave.com/products/aria). Confirm
  the organization enables Smart features and the participant can open Ask ARIA.
- Give participants their own API keys and confirm access to the assigned
  project. Set `WANDB_ENTITY`, `WANDB_PROJECT`, and `WANDB_API_KEY` through molab
  Secrets or an ignored `.env` file.
- Provision Forge Inference access and credits for the agent activity. Provision
  serverless training access and credits before offering the optional SFT path.
  Check the configured models against the account's current availability. The
  agent defaults to `meta-llama/Llama-3.3-70B-Instruct`; SFT defaults to
  `Qwen/Qwen3.8-27B`. Selecting a model is not evidence of a successful live run.
- Use a dedicated facilitator project for the live rehearsal. Keep its run and
  checkpoint links so participants have a reference if their setup fails.

## Share the assignment

The public source is [ash0ts/aria-starter](https://github.com/ash0ts/aria-starter).
Share the GitHub-backed molab links from its [README](README.md); they open the
notebooks on `main`. For a cohort, select a release ref and replace `main` in
the URLs so a later change does not alter the assignment during the session.

Open each link with a participant account and fork it into a fresh workspace.
Follow the student README using the fork before including the links in an
invitation. Preserve [LICENSE](LICENSE), the upstream [license notices](licenses),
and [sources.md](sources.md) when making a copy of the starter.

The URL shape and fork workflow come from [marimo's educator guide](https://molab.marimo.io/for-educators).

## Run the activity

Ask students to choose a question they can judge themselves: “Did the assistant
look up the order before answering?” or “Why is this ticket marked for a human?”
Use development examples throughout iteration. Scenario families are assigned
to a split before their paraphrases are written; do not move near-duplicate
tickets across training, development, and test data.

The CPU exercise compares a fixed baseline with editable model settings and
extra training examples. The agent exercise compares tool outcomes and leaves
answer quality for the student to review. SFT uses the same classification
labels and fixed evaluation examples as the CPU exercise. Start SFT with
**Smoke: 2 tickets**; this uses a small subset to check execution. Use
**Full: 32 tickets** for the complete classification exercise. Both train for
one pass through the selected data. Submit evaluation separately after training.

Each agent question allows at most four model requests, capped at 512 output
tokens each. An SFT comparison allows up to 512 completion tokens per request,
with two model checkpoints evaluated on two tickets in smoke mode or the
selected development/test split in full mode. These caps bound an attempt;
repeated deliberate submissions consume additional credits.

Have each student upload the relevant results, open Ask ARIA from their project,
and paste the supplied handoff. Ask them to explain the proposed change before
editing. If ARIA cannot see a result, confirm the project and run links in the
handoff instead of asking the student to describe the whole experiment again.
Students apply edits in marimo themselves.

Before the final comparison, have students state their hypothesis in one
sentence. Use the held-out test split only after their last development change.
The final discussion should include a failure that remained, even if the
aggregate score improved.

## Rehearse before inviting a cohort

Record the release commit, notebook dependency pins, selected models, date,
project link, inference run links, and SFT checkpoint in your pilot notes.
Library installation alone does not establish that a model or account works.

| Check | Evidence to retain |
| --- | --- |
| Fresh local installation and molab fork open all notebooks | Environment and source revision; CPU baseline output |
| CPU baseline runs before authentication | Baseline metrics and an example mistake |
| Opening a notebook or editing explanatory text does not spend credits | No inference or training run created before submission |
| Agent submission reaches the assigned project | Conversation and child model/tool traces visible in Agent Lens |
| CPU upload reaches the assigned project | Baseline/variant runs, predictions, dataset and source artifacts |
| SFT smoke submission completes and can be inspected after reopening | Model identifier, training run, checkpoint, and comparison results |
| Missing credentials, denied access, unavailable models, or exhausted credits have useful messages | Observed message and next step, with secrets omitted |
| Interrupted work does not silently submit again | Existing remote identifiers and recovery path |
| Notebook forks and exports exclude credentials | Fresh fork requires its own Secrets; inspect the exported source |
| A participant completes baseline → ARIA → edit → comparison | Notebook, run links, and participant explanation |

Treat successful live inference and SFT readback, and a fresh molab walkthrough,
as release requirements. Record any check that remains unverified; do not label
the package classroom-ready on the basis of offline checks alone.

Persist result evidence in W&B and download edited notebooks. Molab preserves
notebook source, uploaded files, and its exact `.env` secrets file according to
its [storage policy](https://marimo.io/pages/molab/storage), but ordinary
generated workspace files may disappear at shutdown. After a training failure,
inspect the existing remote model/run before deciding to submit a new attempt.

## Collect feedback

Record setup friction separately from ARIA feedback so an account or notebook
problem does not become a misleading assessment of ARIA.

| Setup feedback | ARIA feedback |
| --- | --- |
| Where did you need help opening or running the notebook? | What did ARIA explain that you could not explain before? |
| Which account, permission, or credit step stopped you? | Which suggestion did you apply, and why? |
| Which Python edit was difficult? | Did the cited run or example support the explanation? |
| Could you recover your work after leaving the notebook? | Where were you still stuck after the response? |

Keep the student's notebook and result links alongside their responses. Record
whether they completed the experiment, which assistance they needed, and how
they interpreted a result that did not improve.
