# Paper experiment code scope

This source-only selection is bound by SHA-256 to the current English manuscript. It preserves only paper experiment code and necessary shared dependencies/configuration; it contains no experimental outputs.

| Directory | Code retained for |
| --- | --- |
| exp_002_main_reproduction | 20M behavioral baseline and the 4.4M starting checkpoint; initialization 5, training 2. |
| exp_014_icl_ciwl_pair_patch | ICL exemplar swaps at 4.4M/8M/20M; complete label K and eight L1-source-head-to-label-K interventions. |
| exp_016_embedding_layer1_refinement | 16 early/late module combinations, fixed late downstream modules, standard ICL. |
| exp_017_l1_norm_attention_split | Four early/late LayerNorm and attention combinations, standard ICL. |
| exp_023_ciwl_divot_window | Shared implementation and fixed-window input support for the reported CIWL suppression and label-K validation. No separate divot-search results are released. |
| exp_024_l2_label_k_path | Protocol for fixed-window label-K interventions; implementation is shared with exp023. |
| exp_030_functional_update_constraints | Primary-trajectory ordinary training, embedding freezing, task-loss and combined-constraint controls. |
| exp_031_path_only_protection | Primary-trajectory P25, functional confirmation, and update diagnostics. |
| exp_033_random_path_only_protection | Primary-trajectory matched random-pathway protection PR25. |
| exp_034_icl_replay | Direct data replay and uniform L2 regularization; shared implementation for these two controls. |
| exp_035_l2_regularization | Definition of the reported uniform L2 baseline; implementation is shared with the replay control. |
| exp_036_multiseed_path_protection | Independent init7/train4 A/P25/PR25 replication implementation and configuration. |
| exp_038_protection_strength | P/C/D at rho 0.10/0.25/0.50/0.75 on the primary trajectory. |
| exp_040_scale_generalization_v2 | 2L/64 and 2L/128; five init/train pairs s13--s17 in each architecture; ten retained two-layer trajectories. |
| exp_041_emnist_generalization | Frozen primary-trajectory A/P25 models on EMNIST Balanced. |
| exp_042_seed_and_horizon | s18--s22 at 10M, prespecified s18--s20 at 20M--40M, separately identified later s21/s22 supplement. |
| exp_043_kuzushiji49_generalization | Frozen 10M A/P25 model pairs s18--s22 on Kuzushiji-49. |
| exp_044_cifar100_generalization | Frozen 10M A/P25 model pairs s18--s22 on CIFAR-100. |
| exp_045_training_cost | RTX 4070 Laptop, FP32; A/C25/PR25/P25; four processes, each with 100 warm-up and 10,000 timed steps. |
| exp_046_single_constraint_task_loss | Primary-trajectory MEAN25: one rho=0.25 constraint on the equally weighted mean of the two task losses. |
| exp_047_cross_layer_label_k | Frozen driver/kernel utilities imported by exp054/055; no older diagnostic run data or execution protocol. |
| exp_054_uncapped_five_seeds | Four-layer trajectories s18, s19, s23, s24 and s25; code and manuscript IDs are identical. |
| exp_055_uncapped_four_seeds | Four-layer trajectories s20, s21 and s22; only these three manuscript trajectories are configured. |

Select the manuscript ID directly when invoking an adapter with `--seed`. The two four-layer protocols contain exactly the eight reported trajectories.

The primary, independent, two-layer, long-term and four-layer groups use separate seed identities. Full details are in `reproducibility/manuscript_seed_registry.json`.

Seed IDs are scoped by experiment group and architecture. The primary and independent-replication run IDs are `s5` and `s7`; their role descriptions are not alternate IDs.
