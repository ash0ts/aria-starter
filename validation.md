# Validation record

Self-contained lab explanations and live checks were verified on 2026-10-21
with Python 3.12.13 and the dependencies recorded in each notebook's PEP 723
metadata and adjacent lockfile. This is an engineering rehearsal, not a
completed student pilot.

## Offline checks

- All three notebooks installed and ran independently in locked uv environments.
- Notebook parsing, lint, formatting, and HTML export passed.
- The regression suite passed 65 tests, including fixture parity, family-level
  split separation, CPU reproducibility, scoring, and opening notebooks without
  service calls or final-test evaluation.
- Mocked failures covered missing credentials, unavailable models, rejected
  authentication/access, exhausted rate or credit allowance, artifact upload
  interruption, and interrupted training. Paid requests remained bounded and
  existing remote IDs were retained. These are simulated failure checks, not
  evidence of every live account error.

The guided-lab checks additionally cover exposed agent settings in requests,
traces, saved configurations, and source artifacts; SFT settings in registration,
training requests, and saved evidence; invalid bounds rejected before service
setup; and configuration edits retaining an already submitted job. A checkpoint
comparison uses the saved training prompt and settings after notebook edits.

The local marimo editor rendered the CPU lab and reproduced development accuracy
0.75 and macro-F1 0.7431. HTML exports and cell metadata were inspected across
all three notebooks for section order, edit points, and collapsed support code.
The fresh live checks below exercised the default agent and SFT settings.
The wider configurable ranges have offline coverage. The first controlled CPU
experiment changes only `ngram_range`; its candidate keeps `C=1.0`.

## Live checks

Result links below require access to the validation team's project. They do not
grant access to participants and are not needed to run the supplied examples.

| Check | Evidence | Result |
| --- | --- | --- |
| CPU baseline and first controlled edit | [Baseline](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/71c41288a590), [candidate](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/b1b6a4916698) | Both scored 0.75 accuracy and 0.7431 macro-F1 on 16 development tickets. The candidate changed only single words to single words plus two-word phrases. Metrics, prediction evidence, source and model artifacts were read back from W&B. |
| Agent baseline and missing-ID prompt edit | [Baseline](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/agent-bb32ae5742e3), [candidate](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/agent-93b70bd2751f) | Both completed six scenarios and passed three contracts. The candidate fixed missing-ID handling but regressed on the safety scenario. Saved configurations/results and all 12 conversations' tool names, arguments, and results matched server readback. |
| ARIA evidence review | [Baseline conversation](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/weave/agents/conversations/0399a3ce-a876-4c22-9edc-2911dadf6723), [candidate conversation](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/weave/agents/conversations/24c03db9-5702-43d2-93cd-44879915d7cc) | ARIA accessed both runs' source/configuration/results artifacts and the actual conversations. It confirmed the missing-ID improvement and safety regression, and noted that one paired run does not establish reliable causality. |
| Serverless training and recovery | [Smoke training](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/sft-guide-375d167a4f), [model run](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/sft-guide-375d167a4f-model) | Two-ticket Qwen/Qwen3.8-27B training completed and produced checkpoint 1. Recovery from an empty local directory retrieved the same job, model and checkpoint IDs. The checkpoint artifact and recovery snapshot were read back from W&B. No duplicate training job was submitted. |
| SFT checkpoint comparison | [Incomplete comparison](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/eval-guide-6eec7125f0) | The CPU reference completed. The checkpoint 0 inference request returned HTTP 500, so trained-checkpoint evaluation was not attempted. The saved incomplete record and results artifact matched server readback. This exercise is not verified end to end. |

### Dashboard tour

- [W&B project workspace](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/workspace): open the CPU run links above for development accuracy, macro-F1, and prediction tables. Its run dashboard rendered those panels in the browser.
- [Agent Lens overview](https://forge.coreweave.com/agent-lens/wandb/aria-starter-validation): shows conversation activity, tools, latency, model/provider usage, and cost. The overview and conversation list rendered in the browser.
- [Weave Agents](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/weave/agents): the same agent evidence within the W&B project. Its overview rendered the fresh conversations, token totals and tool activity in the browser. The conversation links above identify the exact missing-ID examples to inspect. This project has zero legacy Weave Calls; these notebooks do not create `weave.Evaluation` runs.

The ARIA check used the service API. The notebooks provide a copyable prompt and
project links for students to use in Ask ARIA. ARIA's fresh review turn was
`01a15ed0-b0f3-7820-aec2-c31f6a92e9b6`. This establishes evidence access, not that
a beginner can complete the browser workflow without help.

Immediately after training, the first comparison attempt read an older remote
status and stopped before inference. Recovery refreshed the existing record;
the same unused evaluation ID then reached inference and encountered HTTP 500.
The notebook's recovery instructions cover this propagation delay. A completed
training record is not evidence that checkpoint serving works.

Checkpoint-serving errors were also observed on 2026-10-20, including a failed
trained-checkpoint request. Their cause remains unestablished. The notebook
does not automatically repeat paid requests, and no training job was resubmitted
to work around serving errors. Keep the optional SFT limitation visible before
submission until a complete smoke comparison succeeds.

## Publication

The standalone package is hosted at [ash0ts/aria-starter](https://github.com/ash0ts/aria-starter).
The molab links use this repository while organization hosting is pending.

## Remaining release checks

- Open the published molab links, fork with a participant account, and verify
  credential exclusion from forks and exports. Local editor inspection does
  not replace this participant-account check.
- Resolve or revalidate the checkpoint-serving HTTP 500 responses and complete
  the live SFT comparison before treating that exercise as release-ready.
- Run the facilitator's student walkthrough and collect setup and ARIA feedback
  separately.

Use the [facilitator rehearsal](facilitator.md#rehearse-before-inviting-a-cohort)
before inviting students. The package is not yet classroom-validated.
