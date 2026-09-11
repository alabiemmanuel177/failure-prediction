SHELL := /bin/bash
.DEFAULT_GOAL := verify-premodel

.PHONY: verify-premodel test boundary extraction-gate training-gate pilot-plan pilot-continue ros-build completion-audit review-audit dataset-inventory-check recovery-guard-check monitor-validation monitor-targeted raw-payload-check research1-catalog-check

test:
	python3 -m pytest -q

boundary:
	python3 scripts/check_research1_boundary.py

extraction-gate:
	python3 scripts/check_readiness.py --stage extraction

training-gate:
	python3 scripts/check_readiness.py --stage training

pilot-plan:
	python3 scripts/run_balanced_pilot.py --dry-run

pilot-continue:
	bash -lc 'source scripts/env_research2.sh && python3 scripts/run_balanced_pilot_continuous.py'

completion-audit:
	python3 scripts/audit_project_completion.py

review-audit:
	python3 scripts/manual_audit_app.py

dataset-inventory-check:
	python3 scripts/validate_development_dataset_inventory.py

.PHONY: validation-inventory-check
validation-inventory-check:
	python3 scripts/validate_validation_dataset_inventory.py

.PHONY: targeted-inventory-check
targeted-inventory-check:
	python3 scripts/validate_targeted_dataset_inventory.py

.PHONY: finalize-validation finalize-targeted
finalize-validation:
	python3 scripts/finalize_nonmodel_collection.py validation

finalize-targeted:
	python3 scripts/finalize_nonmodel_collection.py targeted

recovery-guard-check:
	python3 scripts/verify_recovery_guards.py

monitor-validation:
	python3 scripts/monitor_campaign_status.py \
		--manifest data/manifests/balanced_validation_v1.yaml \
		--output reports/status/balanced_validation_monitor.yaml

monitor-targeted:
	python3 scripts/monitor_campaign_status.py \
		--manifest data/manifests/targeted_development_v1.yaml \
		--output reports/status/targeted_development_monitor.yaml

raw-payload-check:
	python3 scripts/audit_raw_artifact_payloads.py

research1-catalog-check:
	python3 scripts/validate_research1_structural_catalog.py

.PHONY: derive-development derive-validation decisions-development decisions-validation await-targeted
derive-development:
	bash -lc 'source scripts/env_research2.sh && nice -n 19 python3 scripts/extract_dataset_sequences.py --inventory data/manifests/balanced_pilot_v1.episodes.jsonl --dataset-id balanced_pilot_v1-development-648 --workers 4'

derive-validation:
	bash -lc 'source scripts/env_research2.sh && nice -n 19 python3 scripts/extract_dataset_sequences.py --inventory data/manifests/balanced_validation_v1.episodes.jsonl --dataset-id balanced_validation_v1-validation-324 --workers 4'

decisions-development:
	python3 scripts/assemble_dataset_decisions.py --dataset-id balanced_pilot_v1-development-648

decisions-validation:
	python3 scripts/assemble_dataset_decisions.py --dataset-id balanced_validation_v1-validation-324

await-targeted:
	python3 scripts/await_campaign_and_derive.py targeted --dataset-id targeted_development_v1-development-1212

.PHONY: nonmodel-audit
nonmodel-audit:
	python3 scripts/audit_nonmodel_completion.py

ros-build:
	bash -lc 'source scripts/env_research2.sh && cd ros_ws && colcon build --symlink-install --packages-select failure_experiment recovery_manager'

verify-premodel: test boundary extraction-gate pilot-plan
	python3 scripts/research_log.py verify
	@echo 'PRE-MODEL VERIFICATION PASS (training remains governed by make training-gate)'

# ---- model development (validation-only selection; nothing here freezes anything) ----
.PHONY: test-models model-development-preliminary model-development-final calibration-select alarm-policy-check freeze-dry-run ablations-plan
test-models:
	.venv/bin/python -m pytest -q tests/test_models_*.py

model-development-preliminary:
	python3 scripts/run_model_development.py --tag preliminary_v1 --train-dataset balanced_pilot_v1-development-648 --selection-dataset balanced_validation_v1-validation-324

# The final pass runs only once every admitted development dataset has been derived.
FINAL_TRAIN_DATASETS ?= --train-dataset balanced_pilot_v1-development-648 --train-dataset targeted_development_v1-development-1212 --train-dataset development_supplement_v1-development-504
model-development-final:
	python3 scripts/run_model_development.py --tag final_v1 $(FINAL_TRAIN_DATASETS) --selection-dataset balanced_validation_v1-validation-324

calibration-select:
	python3 scripts/select_calibration.py $(PRED) --model-id $(MODEL)

alarm-policy-check:
	python3 scripts/validate_alarm_policy.py $(PRED)

freeze-dry-run:
	python3 scripts/freeze_model.py $(FREEZE_ARGS) --dry-run

ablations-plan:
	python3 scripts/run_ablations.py $(ABLATION_ARGS) --dry-run

# ---- Research 1 reuse and the development supplement ----
.PHONY: research1-adapter-audit supplement-plan
research1-adapter-audit:
	bash -lc 'source scripts/env_research2.sh && nice -n 19 python3 scripts/audit_research1_causal_adapter.py --workers 3'

supplement-plan:
	python3 scripts/plan_development_supplement.py --dry-run

# ---- confirmatory (post-freeze only; every target fails closed before the freeze) ----
.PHONY: confirmatory-gate assign-protected-split confirmatory-manifests held-out-continue unseen-family-plan hypotheses
confirmatory-gate:
	python3 scripts/check_readiness.py --stage confirmatory

assign-protected-split:
	python3 scripts/assign_protected_split.py --dry-run

confirmatory-manifests:
	python3 scripts/build_confirmatory_manifests.py --dry-run

held-out-continue:
	bash -lc 'source scripts/env_research2.sh && python3 scripts/run_balanced_pilot_continuous.py --manifest data/manifests/held_out_map_v1.yaml'

unseen-family-plan:
	python3 scripts/run_unseen_family_folds.py --stage fit --dry-run

hypotheses:
	python3 scripts/analyze_hypotheses.py

# ---- recovery study and release ----
.PHONY: train-recovery-selector recovery-manifest-plan recovery-manifest recovery-analysis figures tables reproduce release-report finalize-cards fill-manuscript
train-recovery-selector:
	python3 scripts/train_recovery_selector.py $(COST_TABLE) --output models/recovery_selector/r3_cost_sensitive_ridge_v1.json

recovery-manifest-plan:
	python3 scripts/build_recovery_campaign_manifest.py --dry-run --fake-split

recovery-manifest:
	python3 scripts/build_recovery_campaign_manifest.py --allow-protected-after-freeze

recovery-analysis:
	python3 scripts/analyze_paired_recovery.py $(RECOVERY_OUTCOMES) --allow-protected-after-freeze

figures:
	python3 scripts/build_figures.py

tables:
	python3 scripts/build_tables.py

reproduce:
	python3 scripts/reproduce_release.py

release-report:
	python3 scripts/build_release_report.py

finalize-cards:
	python3 scripts/finalize_cards.py

fill-manuscript:
	python3 scripts/fill_manuscript_results.py
