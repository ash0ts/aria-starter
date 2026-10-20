# ARIA starter maintenance

Read [README.md](README.md) for the student workflow and
[facilitator.md](facilitator.md) for provisioning and release validation.

- Each notebook must run independently from a downloaded `.py` file. Keep
  required fixtures inline and direct dependencies pinned in PEP 723 metadata.
- Keep the classification fixture version and contents identical in the CPU
  and SFT notebooks. Assign scenario families to splits before writing their
  paraphrases; reserve test examples for an explicit final evaluation.
- Keep paid inference and training behind fresh user submission. A source edit,
  notebook open, or reactive cell rerun must not submit a second paid job.
  Preserve identifiers before remote work so failures can be investigated.
- Keep credentials in the environment or `.env`. Do not capture environment
  contents in source snapshots, artifacts, error messages, or exported cells.
- Preserve notices in [sources.md](sources.md) and [licenses](licenses), including
  the modification notice in the fine-tuning notebook.
- Route offline regression checks through the standalone repository's
  `.github/workflows/check.yml`; do not add test-running hooks.
  Run its starter check commands locally before handing off changes. Keep
  agent-only check commands out of student READMEs.
- Report live inference, SFT, and molab checks separately from offline checks.
  Do not claim a configuration was live-tested without its result links.
