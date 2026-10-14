# SSL for ECG classification

This project evaluates self-supervised pretraining on PTB-XL with generic and ECG domain augmentation policies. The ECG study compares supervised learning, SimCLR, BYOL, an adapted CLOCS CMSC baseline and seven SimCLR leave-one-out augmentation variants across ten complete training seeds.

Training and evaluation use a declared patient-separated protocol. Results are derived from saved predictions and no performance benefit is assumed before the controlled comparisons finish.

## Setup on Apple Silicon

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

PyTorch must provide Apple MPS support. The device selector prefers CUDA, then MPS, then CPU. An explicit `--device mps` fails if MPS is unavailable; it does not silently switch to CPU. Some restricted shell environments hide Metal devices, so training may require an ordinary local terminal or approved unrestricted execution.

The default dataset root is `Data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`. It must contain `ptbxl_database.csv`, `scp_statements.csv` and all `records100` waveforms. Override it with `--data-root /absolute/path`. A validated, normalised waveform cache is built once in `data_cache/`.

## Run the ECG experiments

First run the small real-data workflow check:

```bash
.venv/bin/python run_experiments.py --smoke --device mps
```

This runs all 13 conditions for one seed, using at most 256 pretraining records, 128 labelled records and 256 records per evaluation partition, with one pretraining epoch and two downstream epochs. These outputs are smoke checks and must not be reported as scientific results.

Run the publication study:

```bash
.venv/bin/python run_experiments.py --device mps
```

The study contains 130 runs: 13 conditions × seeds 42, 52, 62, 72, 82, 92, 102, 112, 122 and 132. Every SSL run trains a separate encoder and downstream head. Interrupted runs resume at the last completed epoch with saved optimiser, scheduler and RNG state. A change to configuration, training source or data provenance is rejected rather than silently mixing protocols. Use another output directory for a changed study.

Optional subsets and budgets:

```bash
.venv/bin/python run_experiments.py --device mps --conditions supervised simclr_generic simclr_domain --seeds 42 52 62 --output-dir results/pilot
.venv/bin/python run_experiments.py --analyse-only --output-dir results/study
```

Do not use pilot outcomes on the held-out test set to select augmentation policies. Primary settings are prespecified in `configs/experiment_study.json`.

## Protocol

- Data version: PTB-XL 1.0.3; actual counts are computed from metadata.
- Patient-separated recommended folds: 1–8 training, 9 validation, 10 test. Patient overlap is checked.
- SSL uses every training-fold waveform without disease labels. Records with no diagnostic superclass are excluded from supervised training and evaluation rather than treated as known negatives.
- Signals: complete ten-second 12-lead recordings at 100 Hz, 1,000 samples, per-record and per-lead standardisation. No fitted validation/test normalisation statistics are used.
- Label fraction: exactly 1,708 records, 10% of the 17,084 diagnostic-labelled training records after rounding. Multilabel stratification is followed by a deterministic adjustment to enforce the record budget while minimising distortion of marginal class counts. The subset is shared across methods for each seed and stored with ECG and patient IDs.
- Encoder: six convolutional blocks with widths 64, 64, 128, 128, 256 and 256, kernels 11, 7, 7, 5, 5 and 3, followed by global average pooling.
- SimCLR/BYOL/CLOCS pretraining: 20 epochs, batch size 128, Adam at 0.001 and cosine LR decay. Steps, examples seen and elapsed time are recorded. BYOL retains its required predictor and 256-dimensional projector; SimCLR/CLOCS project to 128 dimensions.
- Supervised baseline: end-to-end training. SSL primary evaluation: frozen encoder with a linear head; BatchNorm buffers are also frozen. All downstream conditions train for 20 epochs with batch size 64, Adam at 0.001 and focal loss (positive alpha 0.25, gamma 2).
- Oversampling: record weights average inverse positive-class frequencies across each record's positive labels. Sampling is with replacement; it does not guarantee equal class counts per batch.
- Checkpoint selection: highest validation macro-F1 at threshold 0.5. The selected checkpoint is reloaded before held-out test evaluation.
- Primary outcome: test macro-F1 at threshold 0.5. Secondary outcome: macro-AUROC. Additional outputs include PR-AUC, micro sensitivity/specificity, class-level confusion counts, Brier scores and reliability plots.
- Validation-tuned class thresholds are reported separately and are never selected on test data.

## Augmentation policies

Both policies share additive jitter (sigma 0.03, probability 0.9) and scaling (0.9–1.1, probability 0.8). The generic policy adds zero-filled temporal masking over 5–20% of the recording with probability 0.3. The domain policy adds the seven transforms below with independent per-record gates. There is no hidden outer strong-augmentation gate.

| Domain transform | Probability | Parameters |
|---|---:|---|
| Frequency/time warp | 0.50 | Monotone piecewise linear time map, interior anchors displaced up to 5% of length |
| Mixup | 0.40 | Beta(1,1), anchor weight at least 0.5; shared batch partners and coefficients across views |
| Bandpass variation | 0.30 | Third-order Butterworth, zero-phase; low 0.5–1.5 Hz, high 30–40 Hz at 100 Hz |
| Segment CutMix | 0.25 | 10–30% temporal segment with tapered boundaries and shared source partners |
| Motion/baseline variation | 0.20 | 0.5–2 Hz sinusoid, amplitude 0.01–0.05 in normalised units |
| Channel noise | 0.60 | Independent lead noise, sigma 0.005–0.02 times lead SD |
| Temporal dropout | 0.30 | 5–20% segment, linear gap interpolation |

Motion amplitude is in normalised units, not mV. At 500 Hz the filter's upper range is 40–60 Hz, but that resolution is not the default study. Mixing uses no diagnostic labels or cross-class constraints. These transformations are hypotheses about useful invariance; their clinical realism and empirical benefit are not assumed. Each ablation disables one domain transform while keeping all other probabilities and shared weak transforms fixed. Every run stores the full policy specification.

## Prior ECG SSL baseline

`clocs_cmsc` implements adjacent non-overlapping temporal views and patient-specific cross-view NCE from [CLOCS, ICML 2021](https://proceedings.mlr.press/v139/kiyasseh21a.html). It uses the study's shared 12-lead backbone and projection head. This is an explicitly labelled adaptation, not an exact reproduction of the original architecture. It does not add the domain augmentation library.

## Results and inference

Publication outputs are in `results/study/`; smoke outputs are in `results/smoke/`.

- `study.json` and `status.json`: configuration, current progress and failures.
- Each condition/seed directory: provenance, split IDs, augmentation specification, epoch history, resumable checkpoints, validation-selected checkpoint, validation/test probabilities and metrics.
- `summary.json`: coverage, means, sample SDs, bootstrap 95% intervals of seed means and paired method differences.
- `table5.csv`, `seed_results.csv`, `per_class.csv`: symmetric method statistics and reproducible table inputs.
- `convergence.png` and per-run `reliability.png`: validation learning curves and calibration diagnostics.

Paired comparisons join by seed ID and require complete declared coverage for both methods. Two-sided sign-flip permutation tests are used; Holm adjustments are applied separately to headline and ablation comparison families for each outcome. Macro-F1 is primary; AUROC is secondary. Seed uncertainty does not include finite test-patient sampling uncertainty. A single SSL point outside a supervised interval does not establish significance, and higher F1 does not establish better calibration.

Failed or missing seeds remain visible. The study is marked complete only when every declared run has a result. Smoke and partial pilot results cannot substantiate the manuscript's causal claims.

## Individual training entry points

```bash
.venv/bin/python -m ssrl_ecg.train_ssl_simclr --device mps --config generic --out checkpoints/simclr_generic.pt
.venv/bin/python -m ssrl_ecg.train_ssl_simclr --device mps --config no-frequency-warp --out checkpoints/simclr_without_warp.pt
.venv/bin/python -m ssrl_ecg.train_ssl_byol --device mps --momentum-tau 0.999 --out checkpoints/byol_domain.pt
.venv/bin/python -m ssrl_ecg.train_finetune --device mps --ssl-checkpoint checkpoints/simclr_generic.pt --freeze-encoder --out checkpoints/simclr_generic_probe.pt
.venv/bin/python -m ssrl_ecg.train_supervised --device mps --out checkpoints/supervised.pt
```

These entry points share the shared study engine and retain their provenance and prediction outputs next to the exported checkpoint. `train_supervised_multiseed.py` delegates to the same supervised protocol. Reconstruction, transfer, ResNet, traditional classifiers and label-budget entry points are validated separately; their results are reported with their own task definitions and coverage.

## License

See `LICENSE`.

## Full fine-tuning and reference architecture

The 280 runs at 10% labelled PTB-XL data are complete: 28 conditions across ten seeds. The 130-run linear-probe suite is in `results/study`; the 150-run suite of BYOL omissions, full fine-tuning and native 500 Hz CLOCS reference controls is in `results/experiments`. Checkpoint ancestry is checksum-locked and paired labelled subsets remain identical. Combined tables, seed-paired tests and patient/seed uncertainty are in `results/combined/analysis`.

```bash
.venv/bin/python run_additional_experiments.py --device mps
.venv/bin/python scripts/launch_study.py --suite additional
.venv/bin/python scripts/study_report.py
.venv/bin/python scripts/patient_uncertainty.py
.venv/bin/python scripts/final_analysis.py
.venv/bin/python scripts/study_figures.py
```

Patient-cluster and joint patient/seed intervals quantify held-out-subject variability while retaining repeated records from each subject. The reference CLOCS encoder matches the author's architecture and uses lead II at native 500 Hz; task labels and folds follow this study. See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for input, inference and provenance details. Final analyses require all declared seeds.

At threshold 0.5, supervised macro-F1 is 0.5802. Frozen domain SimCLR and BYOL score 0.2605 and 0.3118; full domain fine-tuning reaches 0.6061 and 0.6067. Their paired F1 gains survive the dedicated 10% fine-tuning Holm adjustment and have positive joint patient/seed intervals; their AUROC gains against supervised learning do not survive adjustment. Fine-tuned shared-backbone CLOCS improves AUROC without a detectable adjusted F1 advantage. Augmentation omission effects differ by framework and metric, including harmful effects. The manuscript and analysis report these qualifications and negative findings.

```bash
.venv/bin/python scripts/validate_pipeline.py --device mps
.venv/bin/python scripts/download_mitbih.py
.venv/bin/python -m ssrl_ecg.transfer_mitbih --mitbih-root Data/mit-bih-arrhythmia-database-1.0.0 --checkpoint results/study/simclr_domain/seed_42/pretrained.pt --use-ssl --device mps
```

XGBoost and LightGBM use the OpenMP library bundled with Torch on macOS. The traditional classifier entry point configures its loader path automatically and never silently substitutes a different requested model.

The evaluation entry points recompute saved prediction metrics before reporting comparisons. `generate_visualizations.py` reads measured epoch histories and predictions; it does not generate example performance data. `scripts/study_progress.py` reports actual seed coverage and durations. `scripts/validate_models.py` checks all seven model paths on real PTB-XL training signals. MIT-BIH is fully downloaded and validated before the folder is installed under `Data/`.


The original 280 runs remain complete and immutable. The label-efficiency and transfer suite is separate:

```bash
.venv/bin/python scripts/analyse_extensions.py
.venv/bin/python scripts/launch_study.py --suite extensions --output-dir results/extensions
.venv/bin/python scripts/monitor_training.py --study-dir results/extensions
# Final analysis of the complete 290-run suite:
.venv/bin/python scripts/finish_extensions.py
.venv/bin/python scripts/verify_studies.py
.venv/bin/python scripts/extension_figures.py
```

This suite adds generic gate/RMS matching, fully fine-tuned comparisons at 1%, 5%, 20% and 50% labelled PTB-XL records, and five matched MIT-BIH transfer conditions across ten training seeds. It reuses checksum-locked encoders and preserves original outputs. MIT-BIH uses the previously evaluated seed-42 subject partition and a binary ten-second window task. It is corroborative transfer evidence, not a new untouched clinical validation dataset. Protocol details and inference families are recorded in the existing experiment protocol. Post-hoc policy/calibration analyses retain their timing labels and original statistical families are unchanged.

All 570 scientific runs are complete and verified: 520 PTB-XL runs and 50 MIT-BIH runs across 57 conditions. The final manuscript contains 18 tables and six figures. Results and paired pointwise uncertainty are in `results/extensions/analysis/study_report.txt` and `panel_comparisons.csv`; all fingerprints, saved prediction metrics and encoder ancestry passed verification. Matched generic controls qualify the domain-policy interpretation: frozen SimCLR domain AUROC is lower than matched generic AUROC; frozen BYOL retains domain advantages, but neither framework establishes a full fine-tuning advantage over matched generic controls. No five-fraction label-efficiency or MIT-BIH contrast survives its panel Holm adjustment. The dedicated 10% fine-tuning family remains unchanged. Equal epochs are not equal compute, and matching RMS does not establish clinical fidelity.
