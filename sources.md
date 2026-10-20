# Sources and attribution

The starter's notebooks are newly written implementations informed by a
support-desk teaching scenario and existing Forge notebook patterns. The SFT
notebook also adapts the public ART example listed below. Its ticket dataset is
newly authored, synthetic fixture data. It does not download an external
dataset or include real customer records.

## Example sources

| Source | Revision | Use here |
| --- | --- | --- |
| [W&B agentic support-bot demo](https://github.com/wandb/agentic-support-bot-demo/tree/f9d4bb2394d3963eacccb3cdce3f5f4dfdb669aa) and its [marimo guide](https://github.com/wandb/agentic-support-bot-demo/blob/f9d4bb2394d3963eacccb3cdce3f5f4dfdb669aa/marimo-guide.py) | `f9d4bb2394d3963eacccb3cdce3f5f4dfdb669aa` | Support-desk scenario and the trace, investigate, edit, evaluate teaching sequence. Its deployment stack is not copied. |
| [Forge Agent Lens notebook](https://github.com/wandb/core/blob/4ac588e51f88c15876a815674236f9dcab0b5a8a/services/gorilla/domain/notebooks/templates/agent_lens_monitor.py) | Core `4ac588e51f88c15876a815674236f9dcab0b5a8a` | Reference for fictional shop tools and scripted conversations. The starter supplies its own fixtures, tools, agent loop, and Forge SDK instrumentation. |
| [Forge SFT notebook](https://github.com/wandb/core/blob/4ac588e51f88c15876a815674236f9dcab0b5a8a/services/gorilla/domain/notebooks/templates/post-training-sft-train_from_file.py) | Core `4ac588e51f88c15876a815674236f9dcab0b5a8a` | Reference for serverless registration and checkpoint inference. The starter implements a classification exercise with persistent job records. |
| [ART train-from-file example](https://github.com/OpenPipe/art-notebooks/blob/74c819447bd1e03318369d30fc0acb0f1106d5b6/examples/sft/train_from_file.ipynb) | `74c819447bd1e03318369d30fc0acb0f1106d5b6` | ART SFT API usage. The starter uses ART's underlying job client to capture job IDs, and adds classification fixtures, explicit submission, recovery, and comparisons. |

The Core links require repository access and record design references. Private
Core source files are not bundled with this starter. The notebooks do not fetch
source code at runtime. Dependencies are pinned inside each notebook rather
than installed from an upstream branch.

## License notices

- Newly authored starter code and documentation use [Apache-2.0](LICENSE).
- The W&B support-bot demo is MIT licensed, copyright (c) 2025 Weights & Biases.
  Its complete notice is retained in
  [licenses/agentic-support-bot-demo.txt](licenses/agentic-support-bot-demo.txt).
- The ART notebook is Apache-2.0 licensed, copyright 2025 OpenPipe, Inc. Its
  upstream license file is retained in
  [licenses/art-notebooks.txt](licenses/art-notebooks.txt). The fine-tuning
  notebook identifies that it was modified for this exercise.
- ART's upstream license file also mentions adaptations from other projects.
  Its [THIRD-PARTY-NOTICES at the recorded revision](https://github.com/OpenPipe/art-notebooks/blob/74c819447bd1e03318369d30fc0acb0f1106d5b6/THIRD-PARTY-NOTICES)
  associates those notices with other notebook and implementation paths, not
  `examples/sft/train_from_file.ipynb`. Those other files are not copied here.

## Product references

- [ARIA](https://docs.coreweave.com/products/aria): team-project and Smart
  features prerequisites, opening Ask ARIA, and attaching experiment context.
- [marimo educator guide](https://molab.marimo.io/for-educators): GitHub preview
  links and per-student notebook forks.
- [molab storage](https://marimo.io/pages/molab/storage): notebook and Secrets
  persistence, fork behavior, and scratch-file limitations.
- [marimo command reference](https://docs.marimo.io/cli/): local editing with
  sandboxed dependencies.

Revalidate model availability, dependency compatibility, and remote execution
for each release using [the facilitator rehearsal](facilitator.md#rehearse-before-inviting-a-cohort).
These source revisions document what was consulted; they are not evidence that
the starter has passed a live rehearsal.
