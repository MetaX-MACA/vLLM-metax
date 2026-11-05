# Environment preparation

Use [environment-check](../../environment-check/SKILL.md) before upgrade work. It owns
Python selection, target/checkouts, the single preparation-plan approval, installation,
private image locks, temporary dependency restoration and final machine verification.

Read its [inventory rules](../../environment-check/references/inventory.md) for the
independent collector, SDK/GPU/package/source evidence and supplemental `collect_env.py`
provenance. Read [preparation](../../environment-check/references/preparation.md) for
commands and durable approval/verification records.

Full upgrades require successful source installations and automatic verification under
the original approved plan before adaptation. Missing/failed installation is a blocker;
compatibility source edits cannot be used to bypass it. Do not request repeated snapshot
confirmation during a long run. Record out-of-plan issues and continue only independent
work whose existing gates are satisfied.
