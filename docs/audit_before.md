# Policy audit (before reward fixes)

Generated with `python scripts\audit_policies.py --output docs/audit_before.md` over seeds 0-4. Values are terminal grader score mean ± population standard deviation.

| Task | noop | grap2 | grap4 | threshold | harmful |
|---|---|---|---|---|---|
| easy | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 |
| medium | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 |
| hard | 0.2500 ± 0.0298 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 | 0.0010 ± 0.0000 |
| expert | 0.2674 ± 0.0100 | 0.1494 ± 0.0212 | 0.1291 ± 0.0337 | 0.1918 ± 0.0275 | 0.1161 ± 0.0343 |
| crisis | 0.4374 ± 0.0047 | 0.4729 ± 0.0055 | 0.3741 ± 0.0045 | 0.5028 ± 0.0054 | 0.3850 ± 0.0045 |
