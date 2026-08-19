# TKEO-Based Echolocation Click Detection

Python implementation and experimental study of a Teager–Kaiser Energy Operator
(TKEO) based method for detecting odontocete echolocation clicks in passive
acoustic recordings, using Gabor click modeling, dual moving-average filtering
(MAF1/MAF2), and Filter Difference Ratio (FDR) based thresholding.

---

## Overview

Passive acoustic monitoring (PAM) is widely used to study echolocating marine
mammals such as dolphins and toothed whales. A key step in PAM analysis is
**click detection** -  identifying short, high-energy transient events in noisy
underwater recordings.

This project implements a click detection pipeline based on the Teager–Kaiser
Energy Operator (TKEO), which is well suited to detecting short-duration,
high-frequency transients like echolocation clicks. The implementation follows
the signal-processing stages described in the reference paper, moving from a
synthetic, fully controlled test signal through to real acoustic data.

---

## Research Paper

- **Reference paper:** [Paper title / citation to be added]
- **Key concepts used:** Gabor click modeling, Teager–Kaiser Energy Operator
  (TKEO), Gaussian moving-average filtering (MAF1), rectangular moving-average
  filtering (MAF2), Filter Difference Ratio (FDR), FDR-peak based thresholding.

Notes and summaries derived from the paper are maintained in
[`docs/paper_notes.md`](docs/paper_notes.md).

---

## Objectives

- Reproduce the TKEO-based click detection method described in the paper.
- Validate each stage of the pipeline independently using synthetic Gabor
  clicks before applying it to real data.
- Quantitatively evaluate detection performance (precision, recall, F1) under
  controlled noise conditions.
- Apply the validated pipeline to real echolocation recordings and compare
  results against the paper's reported findings.

---

## Methodology

The detection pipeline follows these sequential stages:

```
Echolocation Click
        ↓
   Gabor Model
        ↓
      TKEO
        ↓
       MAF1  (Gaussian filter)
        ↓
       MAF2  (Rectangular filter)
        ↓
       FDR
        ↓
    FDR Peak
        ↓
   Threshold
        ↓
Click Detection
```

1. **Gabor Model** -  synthetic clicks are generated using a Gabor waveform to
   provide a controlled, ground-truth test signal.
2. **TKEO** -  the Teager–Kaiser Energy Operator is applied to emphasize
   transient, high-frequency energy associated with clicks.
3. **MAF1 / MAF2** -  the TKEO output is smoothed using two different
   moving-average filters (Gaussian and rectangular) in parallel.
4. **FDR** -  the Filter Difference Ratio is computed from the two filtered
   signals to highlight click-like divergence.
5. **Thresholding** -  FDR peaks above a defined threshold are marked as
   detected clicks.

---

## Implementation Pipeline

The implementation is organized into nine incremental phases, each
corresponding to one stage of the methodology above:

| Phase | Description |
|-------|-------------|
| 1 | Synthetic Gabor click generation |
| 2 | Teager–Kaiser Energy Operator (TKEO) |
| 3 | Noise modeling / realistic input signals |
| 4 | MAF1 -  Gaussian moving-average filter |
| 5 | MAF2 -  Rectangular moving-average filter |
| 6 | FDR computation and FDR-peak identification |
| 7 | Detection / thresholding |
| 8 | Quantitative evaluation (precision, recall, F1) |
| 9 | Validation on real echolocation recordings |

Detailed implementation notes are maintained in
[`docs/implementation_notes.md`](docs/implementation_notes.md).

---

## Project Structure

```
tkeo-echolocation-click-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── signal/          # Gabor click generation
│   ├── tkeo/            # TKEO operator
│   ├── filters/         # MAF1 (Gaussian) and MAF2 (rectangular)
│   ├── detection/       # FDR computation and click detector
│   └── utils/           # Plotting and helper utilities
│
├── experiments/         # Standalone scripts for each pipeline stage
├── notebooks/           # Exploratory analysis
├── data/
│   ├── raw/              # Original / real recordings
│   └── processed/        # Preprocessed signals
├── results/
│   ├── figures/           # Generated plots
│   └── metrics/           # Evaluation outputs
├── docs/
│   ├── paper_notes.md
│   └── implementation_notes.md
└── tests/                # Unit tests for each module
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/tkeo-echolocation-click-detection.git
cd tkeo-echolocation-click-detection

python -m venv venv
source venv/bin/activate    # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

Run an individual pipeline stage as a standalone experiment, e.g.:

```bash
python experiments/01_gabor_signal.py
python experiments/02_tkeo.py
```

Once modules are implemented in `src/`, they can be imported directly:

```python
from src.signal.gabor import generate_gabor_click
from src.tkeo.operator import tkeo
from src.filters.maf1 import apply_maf1
from src.filters.maf2 import apply_maf2
from src.detection.fdr import compute_fdr
from src.detection.detector import detect_clicks
```

---

## Experiments

| Script | Purpose |
|--------|---------|
| `01_gabor_signal.py` | Generate and visualize synthetic Gabor clicks |
| `02_tkeo.py` | Apply TKEO and inspect output on clean signals |
| `03_noise_analysis.py` | Evaluate TKEO robustness under varying SNR |
| `04_maf1_maf2.py` | Apply and compare Gaussian vs. rectangular filtering |
| `05_fdr.py` | Compute FDR and identify FDR peaks |
| `06_detection.py` | Run full detection with thresholding |

---

## Results

Evaluation results (precision, recall, F1-score, detection rate) across
different SNR levels will be added here once Phase 8 is complete. Generated
figures are stored in `results/figures/` and metric summaries in
`results/metrics/`.

---

## Evaluation

Performance is measured by comparing detected click locations against known
ground-truth positions in synthetic test signals:

- **True Positives (TP)** -  correctly detected clicks
- **False Positives (FP)** -  spurious detections
- **False Negatives (FN)** -  missed clicks
- **Precision, Recall, F1-score** -  computed from TP/FP/FN
- **Detection Rate** -  overall fraction of ground-truth clicks correctly detected

---

## References
 
1. Madhusudhana, S., Gavrilov, A., and Erbe, C. (2015). "Automatic detection
   of echolocation clicks based on a Gabor model of their waveform." *The
   Journal of the Acoustical Society of America*, 137(6), 3077–3086.
   https://doi.org/10.1121/1.4921609
2. Additional background references to be added as the project progresses.
---

## Status

🚧 Work in progress -  currently at **Phase 1: Synthetic Gabor Click Generation**.