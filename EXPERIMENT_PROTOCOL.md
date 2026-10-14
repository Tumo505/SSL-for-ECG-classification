# ECG experiment protocol

This study evaluates ECG domain augmentation, self-supervised representation learning, and supervised classification on PTB-XL v1.0.3. Settings are declared before training and held fixed. Test predictions are used only for evaluation and uncertainty analysis.

## Data and splits

The complete dataset contains 21,799 recordings from 18,869 patients. Recommended folds 1–8 train, fold 9 validates, and fold 10 tests; patients do not cross partitions. SSL uses all 17,418 training records. Supervised diagnostic eligibility leaves 17,084 training records. A multilabel stratified sample of exactly 1,708 labelled records is saved independently for each seed and shared across paired conditions. Diagnostic-eligible validation and test sets contain 2,146 and 2,158 records.

The five outputs, in order, are NORM, CD, HYP, MI and STTC. Conduction disturbance is the meaning of CD. Labels can co-occur. Normalization uses each record's per-lead population standard deviation, without fitted test-population statistics.

## Experiment conditions

The completed linear-probe study comprises 13 conditions across seeds 42, 52, 62, 72, 82, 92, 102, 112, 122 and 132: supervised focal loss with oversampling; generic and domain SimCLR; generic and domain BYOL; shared-backbone CLOCS CMSC; and seven SimCLR domain leave-one-out conditions.

The completed encoder-adaptation suite comprises 15 conditions across the same ten seeds: seven BYOL domain leave-one-out conditions; full fine-tuning of generic/domain SimCLR, generic/domain BYOL and shared-backbone CLOCS; and supervised, frozen CMSC and fine-tuned CMSC with the CLOCS reference architecture. Fine-tuning reuses checksum-locked encoders from the corresponding independently trained seed and verifies identical labelled/validation/test record IDs.

The seven independently gated domain transformations are time warp, mixup, bandpass filtering, CutMix, motion artifact, channel noise and temporal dropout. Generic jitter and scaling remain common to every domain omission. Omission effects are conditional on the rest of the library; they do not identify all interactions.

## Training and evaluation

The common six-block backbone receives twelve-lead, 100 Hz, ten-second waveforms. SSL pretraining uses 20 epochs, batch size 128, Adam at 0.001 and cosine learning-rate decay. Downstream training uses 20 epochs, batch size 64, Adam at 0.001, focal alpha 0.25/gamma 2 and weighted record oversampling. SSL linear probing freezes encoder parameters and batch-normalization statistics. Full fine-tuning trains every encoder parameter. The best checkpoint is selected using validation macro-F1 at threshold 0.5 and is reloaded before test evaluation.

The CLOCS reference architecture uses the author's three convolution blocks, kernel 7/stride 3, channels 1–4–16–32, batch normalization, ReLU, max pooling and dropout 0.1, followed by a 320-to-256 linear embedding. CMSC pairs adjacent five-second segments of native 500 Hz lead II. Downstream evaluation uses the first five seconds. Its random-initialized supervised control uses the identical architecture and input. The patient-specific objective follows the author's implementation. These conditions evaluate the reference architecture on this study's five-superclass task; they do not reproduce the original paper's downstream datasets and label definitions. Author source: https://github.com/danikiyasseh/CLOCS and https://proceedings.mlr.press/v139/kiyasseh21a.html.

Fixed-threshold macro-F1 is primary; macro-AUROC is secondary. Per-class AUROC/PR-AUC, Brier score, ECE, reliability diagrams and validation representation variance/effective rank are diagnostics. Validation-only class thresholds provide a secondary operating-point analysis and never alter AUROC.

## Statistical analysis

Every condition reports all ten seed values, mean, sample SD and bootstrap 95% interval. Paired differences use matched seed identities and exact sign-flip tests. Holm adjustment is applied separately by metric within the headline, SimCLR omission, BYOL omission, fine-tuning and reference-architecture families. Failed or missing seeds remain visible; incomplete coverage cannot support final inference.

Patient-cluster bootstrap samples subjects with replacement and retains every recording from each sampled subject. Identical patient draws are shared across conditions. Joint intervals additionally resample training seeds and estimate the mean single-model metric; they are not ensemble metrics. These intervals remain conditional on the observed training/validation subjects and selected thresholds. Training-population uncertainty and external-population transport remain limitations.

## Provenance and execution

The completed 130 runs remain in results/study. The completed 150 encoder-adaptation and omission runs reside in results/experiments. All 280 runs have ten-seed coverage; results/combined/analysis contains verified paired inference, 2,000 patient-cluster and joint patient/seed bootstrap draws, calibrated operating-point diagnostics, tables and figures. Source hashes, configuration, data identity, parent checkpoint hashes, RNG state, histories, split IDs and prediction arrays are retained. Administrative path changes preserve original run fingerprints. Changed source cannot resume an interrupted run with an old fingerprint. Resume only when the study lock is free; do not alter settings during training.

Use run_experiments.py for the original suite and run_additional_experiments.py for the additional suite. scripts/launch_study.py manages detached MPS execution. scripts/study_report.py, scripts/patient_uncertainty.py and scripts/final_analysis.py generate results. scripts/build_manuscript.py creates artifacts/Deep Learning Indaba.docx from measured outputs, with document rendering and page-by-page inspection before delivery.

Legacy entry points are validated separately through scripts/validate_pipeline.py. MIT-BIH transfer is a subject-separated binary ten-second annotation-window task, with records 201 and 202 assigned together. Paced/unknown windows are excluded and one measured lead is repeated across the pretrained twelve-channel interface. This is not an AAMI beat-classification benchmark. Raw datasets and generated caches/results remain ignored by Git.


## Label efficiency and transfer experiments

The original 280 runs remain immutable. A separate suite declares 290 runs across ten paired training seeds before its full launch. These settings follow observation of the original outcomes and must not be described as part of the original statistical plan.

The generic matching control applies seven generic transformation slots using the same per-record gate events as a domain reference view, and scales each generic view's total perturbation to that reference's RMS waveform distance. Jitter and scaling use the same gates. Matching count and RMS magnitude does not match morphology, frequency content, clinical validity or compute. Reference views are generated without labels, on the training minibatch only, and add CPU overhead. Both SimCLR and BYOL receive frozen and fully fine-tuned evaluation, retaining the existing 20-epoch schedules.

Label efficiency compares supervised learning and fully fine-tuned generic/domain SimCLR/BYOL at 1%, 5%, 20% and 50%, reusing checksum-locked encoders. The existing 10% results are retained as a separate historical stratum. Every condition within a seed/fraction uses identical sampled labelled records, validation subjects and test subjects. Training budgets and model selection are unchanged; no test-driven policy or hyperparameter selection is allowed.

MIT-BIH uses five conditions: random initialization and four PTB-XL pretrained encoders, all fully trained for ten epochs with BCE, Adam 0.001 and batch size 32. The subject partition is fixed at split seed 42 for all ten training seeds. Records 201 and 202 remain one subject. This partition has already been evaluated in the earlier single-seed transfer run, so it provides corroborative transfer evidence, not untouched external confirmation. The task remains binary nonoverlapping ten-second annotation windows, with one measured lead repeated for the encoder interface; it is not an AAMI beat-classification benchmark.

Paired F1 and AUROC inference uses separate Holm families per metric for matched generic controls (eight contrasts), label efficiency (30 contrasts including the already-observed 10% stratum) and transfer (six contrasts). Patient-cluster and joint patient/seed confidence intervals are pointwise and conditional on observed training/validation subjects. Comparisons involving existing results are explicitly identified.

Analysis of existing fine-tuned domain/generic differences is post hoc, correcting all four F1/AUROC contrasts together. Post-hoc calibration inference corrects ten Brier/ECE contrasts together. Original inferential families remain unchanged. Compute profiles report measured times, including reuse distinctions; they do not establish a time-matched performance comparison. Pairwise augmentation interactions, exact compute-matched training and prospective clinical validation are outside this suite.

All 290 runs in this suite are complete, yielding 570 verified runs overall. Final panel tests, patient-cluster and joint patient/seed intervals, scientific plots and the standalone manuscript use the complete results. The all-run audit recomputed every fingerprint and saved test metric and verified finite histories, epoch budgets, paired subsets and parent encoder checksums. No test-driven retraining or policy selection followed these outcomes.
