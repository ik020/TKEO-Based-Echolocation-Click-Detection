# TKEO-Based Echolocation Click Detection

Python implementation and experimental study of a Teager–Kaiser Energy Operator
(TKEO) based method for detecting odontocete echolocation clicks in passive
acoustic recordings, using Gabor click modeling, dual moving-average filtering
(MAF1/MAF2), and Filter Difference Ratio (FDR) based thresholding.

---

## Overview

Passive acoustic monitoring (PAM) is widely used to study echolocating marine
mammals such as dolphins and toothed whales. A key step in PAM analysis is
**click detection** - identifying short, high-energy transient events in noisy
underwater recordings.

This project implements a click detection pipeline based on the Teager–Kaiser
Energy Operator (TKEO), which is well suited to detecting short-duration,
high-frequency transients like echolocation clicks. The implementation follows
the signal-processing stages described in the reference paper, moving from a
synthetic, fully controlled test signal through to real acoustic data.

---

## Research Paper

- **Reference paper:** Madhusudhana, S., Gavrilov, A., and Erbe, C. (2015).
  "Automatic detection of echolocation clicks based on a Gabor model of their
  waveform." *J. Acoust. Soc. Am.*, 137(6), 3077–3086.
- **Key concepts used:** Gabor click modeling, Teager–Kaiser Energy Operator
  (TKEO), Gaussian moving-average filtering (MAF1), rectangular moving-average
  filtering (MAF2), Filter Difference Ratio (FDR), FDR-peak based thresholding.

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
     Add Noise
        ↓
       MAF1  (Gaussian filter)
        ↓
       MAF2  (Rectangular filter)
        ↓
       FDR
        ↓
    FDR Peak
        ↓
   Threshold + Refractory Period
        ↓
Click Detection
        ↓
    Evaluation
```

1. **Gabor Model** - synthetic clicks are generated using a Gabor waveform to
   provide a controlled, ground-truth test signal.
2. **TKEO** - the Teager–Kaiser Energy Operator is applied to emphasize
   transient, high-frequency energy associated with clicks.
3. **Noise Injection** - Gaussian noise is added at a controllable SNR to
   simulate realistic recording conditions.
4. **MAF1 / MAF2** - the TKEO output is smoothed using two different
   moving-average filters (Gaussian and rectangular) in parallel, with filter
   parameters (`sigma_g`, `N`) matched to the click's expected TKEO width per
   the paper's Eq. (6) and (11).
5. **FDR** - the Filter Difference Ratio is computed from the two filtered
   signals to highlight click-like divergence, with invalid/non-meaningful
   values bypassed per the paper's constraints.
6. **Thresholding** - FDR peaks above a defined fraction of FDR_peak are
   marked as detected clicks, with a refractory period to prevent a single
   noisy click from being split into multiple detections.
7. **Evaluation** - detections are compared against known ground-truth click
   times to compute precision, recall, and F1-score.

---

## Implementation Pipeline

The implementation is organized into nine incremental phases, each
corresponding to one stage of the methodology above:

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Synthetic Gabor click generation | ✅ Complete |
| 2 | Teager–Kaiser Energy Operator (TKEO) | ✅ Complete |
| 3 | Noise modeling / realistic input signals | ✅ Complete |
| 4 | MAF1 - Gaussian moving-average filter | ✅ Complete |
| 5 | MAF2 - Rectangular moving-average filter | ✅ Complete |
| 6 | FDR computation and FDR-peak identification | ✅ Complete |
| 7 | Detection / thresholding | ✅ Complete |
| 8 | Quantitative evaluation (precision, recall, F1) | ✅ Complete |
| 9 | Validation on real echolocation recordings | 🚧 In Progress |

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
│   ├── signal/          # Gabor click generation, multi-click signal builder
│   ├── tkeo/            # TKEO operator
│   ├── noise/           # SNR-controlled noise injection
│   ├── filters/         # MAF1 (Gaussian) and MAF2 (rectangular)
│   ├── detection/       # FDR computation and click detector
│   ├── evaluation/      # Precision/recall/F1 metrics
│   └── config.py        # Shared signal/filter parameters
│
├── experiments/         # Standalone scripts for each pipeline stage
├── data/
│   ├── raw/              # Original / real recordings
│   └── processed/        # Preprocessed signals
├── results/
│   ├── figures/           # Generated plots
│   └── metrics/           # Evaluation outputs (saved arrays)
├── docs/
│   ├── paper_notes.md
│   └── implementation_notes.md
└── tests/                # Unit and integration tests for each module
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/tkeo-echolocation-click-detection.git
cd tkeo-echolocation-click-detection

python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

Run an individual pipeline stage as a standalone experiment, e.g.:

```bash
python -m experiments.01_gabor_signal
python -m experiments.02_tkeo
```

Once modules are implemented in `src/`, they can be imported directly:

```python
from src.signal.gabor import generate_gabor_click
from src.signal.multi_click import generate_multi_click_signal
from src.tkeo.operator import first_derivative, second_derivative, tkeo
from src.noise.noise import add_noise
from src.filters.maf1 import maf1
from src.filters.maf2 import maf2
from src.detection.fdr import fdr, fdr_peak
from src.detection.detector import detect_clicks
from src.evaluation.metrics import evaluate_detections
```

---

## Experiments

| Script | Purpose |
|--------|---------|
| `01_gabor_signal.py` | Generate and visualize synthetic Gabor clicks |
| `02_tkeo.py` | Apply TKEO and inspect derivatives/energy output on clean signals |
| `03_noise_analysis.py` | Evaluate TKEO robustness under varying SNR (20/10/5/0 dB) |
| `04_maf1.py` | Apply Gaussian moving-average filter (MAF1) to noisy TKEO output |
| `05_maf2.py` | Apply and compare rectangular filter (MAF2) against MAF1 |
| `06_fdr.py` | Compute FDR curve and FDR_peak from MAF1/MAF2 outputs |
| `07_detection.py` | Threshold FDR and detect click locations |
| `08_evaluation.py` | Multi-click synthetic test with precision/recall/F1 scoring |

---

## Results

At **5 dB SNR**, using filter parameters matched to the click's TKEO width
(`sigma_g = SIGMA/√2` in samples, `N = ceil(5·sigma_g)`) and a detection
threshold of **35% of FDR_peak** with a refractory period of `2·SIGMA`:

- Across repeated trials on a 5-click synthetic test signal, the detector
  consistently achieved **Precision = 1.000, Recall = 1.000, F1 = 1.000**
  after excluding detections falling within one filter-window's width of the
  signal boundaries (a known edge artifact of finite-length filtering).
- Without edge exclusion, boundary artifacts occasionally produced 1–2 false
  positives per run at the very start/end of the recording.
- A refractory period was necessary to prevent noise-driven jitter near a
  single click's peak from being split into multiple detections.

Generated figures are stored in `results/figures/` and metric summaries in
`results/metrics/`.

---

## Evaluation

Performance is measured by comparing detected click locations against known
ground-truth positions in synthetic test signals, using one-to-one matching
within a time tolerance:

- **True Positives (TP)** - correctly detected clicks
- **False Positives (FP)** - spurious detections
- **False Negatives (FN)** - missed clicks
- **Precision, Recall, F1-score** - computed from TP/FP/FN

---

## References

1. Madhusudhana, S., Gavrilov, A., and Erbe, C. (2015). "Automatic detection
   of echolocation clicks based on a Gabor model of their waveform." *The
   Journal of the Acoustical Society of America*, 137(6), 3077–3086.
   https://doi.org/10.1121/1.4921609
2. Additional background references to be added as the project progresses.

---

## Status

🚧 Work in progress - Phases 1–8 complete and tested; currently on
**Phase 9: Validation on Real Echolocation Recordings**.