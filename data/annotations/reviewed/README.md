# Independent annotation reviews

Place one reviewed YAML file here for each of the 20 frozen audit run IDs. Start from
the corresponding file in the parent directory, inspect the MCAP and episode summary,
and add distinct `review.first_reviewer` and `review.second_reviewer` names. Set
`review.adjudication_status` to `agreed` or `adjudicated` only after both reviews.

Do not copy model predictions into this directory and do not modify the automatic
baseline in the parent directory. Check the gate with:

```bash
source scripts/env_research2.sh
python3 scripts/check_manual_audit_gate.py
```
