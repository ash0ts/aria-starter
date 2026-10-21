# Validation record

Guided-lab changes checked on 2026-10-21 with Python 3.12.13 and the dependencies
recorded in each notebook's PEP 723 metadata and adjacent lockfile. Live checks
below were recorded on 2026-10-20, before the guided-lab changes. This is an
engineering rehearsal, not a completed student pilot.

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
No paid requests were submitted for this change. The new configurable ranges
have offline coverage; the live evidence below covers the original defaults.

## Live checks

Result links below require access to the validation team's project. They do not
grant access to participants and are not needed to run the supplied examples.

| Check | Evidence | Result |
| --- | --- | --- |
| CPU logging | [Baseline](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/f646ba12c7ed), [candidate](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/7ac2a3bdba5f) | Both development evaluations had accuracy 0.75 and macro-F1 0.7431. Predictions, dataset/source evidence, and model artifacts were read back from W&B. |
| Forge agent baseline | [Run](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/agent-62a458e0e695) | `meta-llama/Llama-3.3-70B-Instruct` completed six scenarios and passed three tool/outcome contracts. Evidence artifacts and Agent Lens messages, tool calls, and token usage were read back. |
| ARIA handoff and manual edit | [Comparison run](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/agent-47a97a3f6ee9) | ARIA accessed the baseline's saved source, results, and missing-order conversation. Its suggested prompt edit fixed the missing-ID case, while a safety case regressed. The overall score remained three of six. |
| Serverless SFT training and recovery | [Smoke run](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/sft-smoke-2557077751) | A two-ticket `Qwen/Qwen3.8-27B` job completed and produced checkpoint 1. Recovery from an empty local workspace retrieved the same job, model, and checkpoint IDs without submitting training again. |
| SFT checkpoint comparison | [Partial comparison](https://forge.coreweave.com/wandb/wandb/aria-starter-validation/runs/eval-smoke-17c68dc734) | CPU and checkpoint 0 predictions completed on the same two development tickets. The trained-checkpoint request returned HTTP 500. Individual requests to both checkpoints succeeded separately, but a complete notebook comparison remains unverified. Partial results and identifiers were retained. |

The ARIA check used the service API. The notebooks themselves provide a copyable
prompt and project links for students to use in Ask ARIA. This check does not
establish that a beginner can complete the browser workflow without help.

The live checks found and fixed compatibility issues: W&B 0.31 no longer
provides `wandb.util.generate_id`, and ART requires JSONL artifacts to include
`format` and `num_trajectories` metadata. W&B summary reads also flatten nested
objects; recovery now uses a scalar JSON snapshot and bounded read retries for
propagation. Regression tests cover these behaviors.

The checkpoint-serving errors were intermittent. The cause has not been
established. The notebook does not automatically repeat paid requests; its
error guidance asks the student to inspect the existing job before a deliberate
new evaluation. No training job was resubmitted to work around serving errors.

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
