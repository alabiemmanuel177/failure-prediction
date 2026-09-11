# Frozen 20-episode manual audit

These timelines contain telemetry and label-only onset/event markers, but no model outputs. Review the MCAP when a plotted signal is ambiguous.

Persistence-based localisation/immobilisation candidates for each run are in
`reports/manual-audit/operational-events/<run_id>.yaml`; the aggregate integrity result
is `reports/integrity/operational_event_audit_v1.yaml`. These are automatic evidence,
not a replacement for the primary human reviewer and is not independent-review evidence.

| Episode key | Fault family | Outcome | Timeline | Automatic annotation |
|---|---|---|---|---|
| camera_occlusion-high-201 | camera_occlusion | success | [05ed9e9d-a50c-4d2d-890a-7c231e67e65c](05ed9e9d-a50c-4d2d-890a-7c231e67e65c.svg) | `data/annotations/05ed9e9d-a50c-4d2d-890a-7c231e67e65c.yaml` |
| camera_occlusion-low-201 | camera_occlusion | success | [097aa26f-e58d-4e83-967e-93e31bade745](097aa26f-e58d-4e83-967e-93e31bade745.svg) | `data/annotations/097aa26f-e58d-4e83-967e-93e31bade745.yaml` |
| camera_occlusion-medium-201 | camera_occlusion | success | [1b9bac59-84b0-403e-aa0d-aaa52d8cbba1](1b9bac59-84b0-403e-aa0d-aaa52d8cbba1.svg) | `data/annotations/1b9bac59-84b0-403e-aa0d-aaa52d8cbba1.yaml` |
| control-s0-202 | none | success | [3a410953-ae25-48eb-a0da-ea58d0ec5097](3a410953-ae25-48eb-a0da-ea58d0ec5097.svg) | `data/annotations/3a410953-ae25-48eb-a0da-ea58d0ec5097.yaml` |
| lidar_dropout-high-201 | lidar_dropout | success | [45d7bca0-006f-45ab-9a45-2277a44540b7](45d7bca0-006f-45ab-9a45-2277a44540b7.svg) | `data/annotations/45d7bca0-006f-45ab-9a45-2277a44540b7.yaml` |
| localisation_perturbation-medium-201 | localisation_perturbation | false_arrival | [47cc659b-2820-4d25-8290-13e1d3ea2b40](47cc659b-2820-4d25-8290-13e1d3ea2b40.svg) | `data/annotations/47cc659b-2820-4d25-8290-13e1d3ea2b40.yaml` |
| control-s3-201 | none | success | [4a778017-f9eb-4616-9c2a-65bf7b8f5291](4a778017-f9eb-4616-9c2a-65bf7b8f5291.svg) | `data/annotations/4a778017-f9eb-4616-9c2a-65bf7b8f5291.yaml` |
| semantic_corruption-medium-201 | semantic_corruption | success | [51493307-1a2f-457e-9301-434e5c9aed53](51493307-1a2f-457e-9301-434e5c9aed53.svg) | `data/annotations/51493307-1a2f-457e-9301-434e5c9aed53.yaml` |
| planner_oscillation-medium-201 | planner_oscillation | planner_failure | [611ce5e7-2984-4c99-b503-428a6c9d807d](611ce5e7-2984-4c99-b503-428a6c9d807d.svg) | `data/annotations/611ce5e7-2984-4c99-b503-428a6c9d807d.yaml` |
| lidar_dropout-medium-201 | lidar_dropout | success | [81d9b0fd-23f8-402e-847e-4c99ab6ff099](81d9b0fd-23f8-402e-847e-4c99ab6ff099.svg) | `data/annotations/81d9b0fd-23f8-402e-847e-4c99ab6ff099.yaml` |
| semantic_corruption-low-201 | semantic_corruption | success | [8402ee7a-2186-4cd2-a772-e781413c05de](8402ee7a-2186-4cd2-a772-e781413c05de.svg) | `data/annotations/8402ee7a-2186-4cd2-a772-e781413c05de.yaml` |
| wheel_slip-low-201 | wheel_slip | success | [8d722d33-c440-466d-9968-517118a74f9d](8d722d33-c440-466d-9968-517118a74f9d.svg) | `data/annotations/8d722d33-c440-466d-9968-517118a74f9d.yaml` |
| lidar_dropout-low-201 | lidar_dropout | success | [ab2f96ce-4486-4a9b-84a7-3bac35285543](ab2f96ce-4486-4a9b-84a7-3bac35285543.svg) | `data/annotations/ab2f96ce-4486-4a9b-84a7-3bac35285543.yaml` |
| dynamic_blockage-low-201 | dynamic_blockage | success | [af068cd4-784e-43e7-a0b4-c7e37335e914](af068cd4-784e-43e7-a0b4-c7e37335e914.svg) | `data/annotations/af068cd4-784e-43e7-a0b4-c7e37335e914.yaml` |
| planner_oscillation-low-201 | planner_oscillation | success | [bd56dede-8a88-4c6d-93a6-e3fb9bdf5aae](bd56dede-8a88-4c6d-93a6-e3fb9bdf5aae.svg) | `data/annotations/bd56dede-8a88-4c6d-93a6-e3fb9bdf5aae.yaml` |
| localisation_perturbation-low-201 | localisation_perturbation | success | [cc8b7541-2e11-4ed6-b1f7-4e32b8a02f8a](cc8b7541-2e11-4ed6-b1f7-4e32b8a02f8a.svg) | `data/annotations/cc8b7541-2e11-4ed6-b1f7-4e32b8a02f8a.yaml` |
| control-s0-201 | none | success | [dce46d22-c55b-42ed-9bf6-adda0a0b6043](dce46d22-c55b-42ed-9bf6-adda0a0b6043.svg) | `data/annotations/dce46d22-c55b-42ed-9bf6-adda0a0b6043.yaml` |
| dynamic_blockage-medium-201 | dynamic_blockage | success | [e3d7de09-b6f8-43bc-8931-8019fe4823e5](e3d7de09-b6f8-43bc-8931-8019fe4823e5.svg) | `data/annotations/e3d7de09-b6f8-43bc-8931-8019fe4823e5.yaml` |
| control-s3-202 | none | success | [f5aaf759-5702-4fe2-8f72-0f39b8ab6db5](f5aaf759-5702-4fe2-8f72-0f39b8ab6db5.svg) | `data/annotations/f5aaf759-5702-4fe2-8f72-0f39b8ab6db5.yaml` |
| wheel_slip-medium-201 | wheel_slip | success | [f62c7142-0ec6-4078-adf3-de58f942658c](f62c7142-0ec6-4078-adf3-de58f942658c.svg) | `data/annotations/f62c7142-0ec6-4078-adf3-de58f942658c.yaml` |

Training admission remains forbidden until `python3 scripts/check_manual_audit_gate.py` passes.
