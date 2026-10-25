# Experiment scope

The release is bound to the current formal manuscript, including its appendices. Every experiment directory below has an explicit manuscript anchor. The source for CIWL window construction is shared with the label-K validation; it contains no separate search results.

| Directory | Manuscript anchors | Released condition |
|---|---|---|
| `exp_002_main_reproduction` | `app:baseline` | 20M behavioral baseline and the 4.4M starting checkpoint; initialization 5, training 2. |
| `exp_014_icl_ciwl_pair_patch` | `sec:path-validation`, `fig:causal-path` | ICL exemplar swaps at 4.4M/8M/20M; complete label K and eight L1-source-head-to-label-K interventions. |
| `exp_016_embedding_layer1_refinement` | `sec:localization`, `fig:localization` | 16 early/late module combinations, fixed late downstream modules, standard ICL. |
| `exp_017_l1_norm_attention_split` | `fig:localization` | Four early/late LayerNorm and attention combinations, standard ICL. |
| `exp_023_ciwl_divot_window` | `sec:path-validation`, `fig:causal-path` | Shared implementation and fixed-window input support for the reported CIWL suppression and label-K validation. No separate divot-search results are released. |
| `exp_024_l2_label_k_path` | `sec:path-validation`, `fig:causal-path` | CIWL N/S/K/V/KB interventions over the eleven fixed checkpoints at 4.15M--4.65M. |
| `exp_030_functional_update_constraints` | `sec:baselines`, `tab:protection-results`, `app:additional-arms` | Primary-trajectory ordinary training, embedding freezing, task-loss and combined-constraint controls. |
| `exp_031_path_only_protection` | `sec:path`, `tab:protection-results`, `tab:update-size`, `app:functional-confirmation` | Primary-trajectory P25, functional confirmation, and update diagnostics. |
| `exp_033_random_path_only_protection` | `sec:baselines`, `tab:protection-results` | Primary-trajectory matched random-pathway protection PR25. |
| `exp_034_icl_replay` | `sec:baselines`, `tab:protection-results` | Direct data replay and uniform L2 regularization; shared implementation for these two controls. |
| `exp_035_l2_regularization` | `sec:baselines`, `tab:protection-results` | Definition of the reported uniform L2 baseline; implementation is shared with the replay control. |
| `exp_036_multiseed_path_protection` | `sec:experimental-setup`, `sec:protection-results` | Only the reported initialization 7 / training 4 replication, A/P25/PR25 saved arrays. |
| `exp_038_protection_strength` | `fig:s5-strength`, `tab:selectivity`, `app:protection-strength`, `tab:strength-results` | P/C/D at rho 0.10/0.25/0.50/0.75 on the primary trajectory. |
| `exp_040_scale_generalization_v2` | `sec:scale-results`, `sec:boundary-results`, `fig:s5-scale`, `app:scale-protocol`, `tab:scale-localization`, `tab:scale-all-arms`, `tab:scale-primary` | 2L/64, 2L/128, 4L/128; s13--s17; all 15 trajectories including failed functional confirmation. |
| `exp_041_emnist_generalization` | `tab:emnist`, `app:emnist` | Frozen primary-trajectory A/P25 models on EMNIST Balanced. |
| `exp_042_seed_and_horizon` | `sec:horizon-results`, `fig:s5-horizon`, `app:horizon`, `tab:horizon-all-budgets`, `tab:horizon-seeds`, `tab:horizon-capability-bounds` | s18--s22 at 10M, prespecified s18--s20 at 20M--40M, separately identified later s21/s22 supplement. |
| `exp_043_kuzushiji49_generalization` | `fig:s5-external`, `tab:external-five-seed`, `app:external-supplement`, `tab:external-individual-seeds` | Frozen 10M A/P25 model pairs s18--s22 on Kuzushiji-49. |
| `exp_044_cifar100_generalization` | `fig:s5-external`, `tab:external-five-seed`, `app:external-supplement`, `tab:external-individual-seeds` | Frozen 10M A/P25 model pairs s18--s22 on CIFAR-100. |
| `exp_045_training_cost` | `sec:compute-cost`, `tab:compute-cost`, `app:compute-cost`, `tab:compute-cost-repeats` | RTX 4070 Laptop, FP32; A/C25/PR25/P25; four processes, each with 100 warm-up and 10,000 timed steps. |
| `exp_046_single_constraint_task_loss` | `sec:baselines`, `sec:protection-results`, `tab:protection-results` | Primary-trajectory MEAN25: one rho=0.25 constraint on the equally weighted mean of the two task losses. |

Only standard ICL, CIWL, and original-task results are presented. Fixed training-label swaps remain part of the protection data and are not an extra performance evaluator. Shared model and statistical modules support the listed experiments.
