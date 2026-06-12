# rPPG-UMex

**Unsupervised rPPG Heart Rate Estimation Across Fitzpatrick Skin Tones — ISB Cohort (Mexico City)**

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![rPPG-Toolbox](https://img.shields.io/badge/framework-rPPG--Toolbox-green.svg)](https://github.com/ubicomplab/rPPG-Toolbox)

---

## Overview

This repository provides the code and configuration files associated with the paper:

> **Remote Heart Rate Estimation Across Skin Tones: A Benchmark of Unsupervised rPPG Methods in a Mexican University Cohort.**
> *[Vianey Martínez-Pérez (1), Ana Zequeira-Hernández (1), Fátima Cancino-Monroy (2), Andrea Gómez-López (1), Daniel Haro-Mendoza (3) and  Luis Jimenez-Angeles (1) luis.jimenez@ieee.org
(1) Department of Biomedical Systems, Engineering Faculty, UNAM, Mexico City 04510, México. 
(2) Bachelor’s Degree in Physiotherapy, Faculty of Medicine, UNAM, Mexico City 04510, México.  
(3) Department of Mechatronics, Engineering Faculty, UNAM, Mexico City 04510, México.  ] 
— [2nd IFMBE Latin American Conference on Digital Health (CLASD 2026) ], [2026]*

We evaluate seven classical unsupervised remote photoplethysmography (rPPG) algorithms
for heart rate estimation on the ISB dataset — a cohort of 50 Mexican university students
with balanced Fitzpatrick skin tone representation (types I–V). The study quantifies
algorithmic bias across three skin tone groups (FP I+II, FP III, FP IV+V) and evaluates
a Fitzpatrick-stratified linear correction as a post-processing bias mitigation strategy.

---

## Key findings

| Method | MAE (bpm) | Pearson r | Best FP group |
|--------|-----------|-----------|---------------|
| GREEN  | 6.79      | 0.625 **  | FP I+II (4.61 bpm) |
| ICA    | 7.18      | 0.532 **  | FP I+II (5.41 bpm) |
| CHROM  | 9.80      | 0.279 *   | FP I+II (7.36 bpm) |
| POS    | 10.28     | 0.241     | FP IV+V (8.71 bpm) |
| LGI    | 10.08     | 0.134     | FP III  (9.59 bpm) |
| OMIT   | 10.08     | 0.134     | FP III  (9.59 bpm) |
| PBV    | 11.82     | -0.007    | FP IV+V (9.46 bpm) |

\* p < 0.05  \*\* p < 0.001 — No method achieved significance in the FP IV+V group.

---

## Repository structure

```
rPPG-UMex/
│
├── dataset/
│   └── data_loader/
│       └── ISBLoader.py              # Custom rPPG-Toolbox DataLoader for ISB dataset
│
├── configs/
│   └── ISB_UNSUPERVISED.yaml         # Preprocessing and inference configuration
│
├── analysis/
│   ├── analyze_fitzpatrick_isb.py    # Fitzpatrick-stratified HR bias analysis
│   ├── fitzpatrick_linear_correction.py  # LOO-CV linear correction (α, β per group)
│   └── plot_fitzpatrick_isb.py       # Publication-ready figure generation
│
├── data/
│   └── isb_ground_truth.csv          # HR ground truth + Fitzpatrick labels (n=50)
│
└── README.md
```

---

## Dataset

The **ISB dataset** (rPPG_UMex) consists of 50 Mexican university students recruited at
[INSTITUTION], Mexico City. Each subject contributed one 60-second facial video recorded
at 30 fps under controlled indoor illumination, with concurrent heart rate ground truth
from a contact pulse oximeter.

| Attribute | Value |
|-----------|-------|
| Subjects | 50 |
| Age range | 18–25 years |
| Sex | Mixed |
| Fitzpatrick FP I+II | 16 subjects (32%) |
| Fitzpatrick FP III | 17 subjects (34%) |
| Fitzpatrick FP IV+V | 17 subjects (34%) |
| Video duration | ~60 s per subject |
| Frame rate | 30 fps |
| Resolution | 1920 × 1080 px |
| Ground truth | HR (contact pulse oximeter, 1 Hz) |

> **Note:** Raw video files are not publicly available due to ethical restrictions.
> The ground truth CSV (`isb_ground_truth.csv`) contains HR values, Fitzpatrick
> classification, age, and sex — no direct identifiers. Data collection was approved
> by the Ethics Committee of [INSTITUTION] (Protocol No. [XXXX]).

---

## Requirements

### Dependencies

```bash
# Core framework
git clone https://github.com/ubicomplab/rPPG-Toolbox
cd rPPG-Toolbox
pip install -r requirements.txt

# Additional dependencies for analysis scripts
pip install scikit-learn matplotlib scipy pandas numpy
```

### Tested environment

| Package | Version |
|---------|---------|
| Python | 3.8 |
| PyTorch | 1.13 |
| OpenCV | 4.8.1 (headless) |
| NumPy | 1.23 |
| SciPy | 1.9 |
| pandas | 1.5 |
| scikit-learn | 1.2 |
| matplotlib | 3.7 |

---

## Usage

### Step 1 — Integrate ISBLoader into rPPG-Toolbox

```bash
# Copy loader and configuration
cp dataset/data_loader/ISBLoader.py  rPPG-Toolbox/dataset/data_loader/
cp configs/ISB_UNSUPERVISED.yaml     rPPG-Toolbox/configs/infer_configs/
cp data/isb_ground_truth.csv         /path/to/ISB_DATASET/

# Register ISBLoader in __init__.py
echo "from dataset.data_loader.ISBLoader import ISBLoader" \
  >> rPPG-Toolbox/dataset/data_loader/__init__.py
```

Add the following to the dataset dispatcher in `rPPG-Toolbox/main.py`:

```python
from dataset.data_loader.ISBLoader import ISBLoader
# ...
elif config.UNSUPERVISED.DATA.DATASET == "ISB":
    unsupervised_loader = ISBLoader
```

### Step 2 — Update paths in YAML

Edit `configs/ISB_UNSUPERVISED.yaml`:

```yaml
DATA_PATH:   "/path/to/ISB_DATASET/"
CACHED_PATH: "/path/to/ISB_DATASET/PROCESS/"
```

### Step 3 — Run preprocessing and inference

```bash
cd rPPG-Toolbox
conda activate rppg-toolbox

# First run: preprocessing + inference (DO_PREPROCESS: True)
python main.py --config_file configs/infer_configs/ISB_UNSUPERVISED.yaml

# Subsequent runs: inference only (set DO_PREPROCESS: False in YAML)
python main.py --config_file configs/infer_configs/ISB_UNSUPERVISED.yaml
```

### Step 4 — Fitzpatrick bias analysis

```bash
python analysis/analyze_fitzpatrick_isb.py \
    --results_dir runs/ISB_Unsupervised/ \
    --gt_csv     data/isb_ground_truth.csv \
    --output_dir results/ISB_Fitzpatrick/
```

Output files:
- `ISB_global_metrics.csv` — MAE, RMSE, Pearson, Spearman per method
- `ISB_fitzpatrick_metrics.csv` — metrics stratified by FP group
- `ISB_all_predictions.csv` — all predictions with subject metadata

### Step 5 — Linear correction

```bash
python analysis/fitzpatrick_linear_correction.py \
    --predictions_csv results/ISB_Fitzpatrick/ISB_all_predictions.csv \
    --output_dir      results/ISB_Correction/
```

### Step 6 — Generate figures

```bash
python analysis/plot_fitzpatrick_isb.py \
    --results_dir results/ISB_Fitzpatrick/ \
    --output_dir  figures/ISB/
```

---

## Results

After running the full pipeline, results are saved in:

```
results/
├── ISB_Fitzpatrick/
│   ├── ISB_global_metrics.csv
│   ├── ISB_fitzpatrick_metrics.csv
│   └── ISB_all_predictions.csv
├── ISB_Correction/
│   ├── ISB_correction_metrics.csv
│   └── ISB_corrected_predictions.csv
└── figures/ISB/
    ├── Fig1_MAE_barplot.png
    ├── Fig2_MAE_heatmap.png
    ├── Fig3_scatter_panels.png
    ├── Fig4_Pearson_barplot.png
    └── Fig5_MAE_correction.png
```

---

## Citation (in progress)

If you use this code or dataset in your research, please cite:

```bibtex
@inproceedings{[citekey][year],
  title     = {Skin Tone Bias in Unsupervised rPPG Heart Rate Estimation:
               A Fitzpatrick-Stratified Study in Mexican University Students},
  author    = {[Authors]},
  booktitle = {[Conference]},
  year      = {[Year]},
  url       = {https://github.com/[USERNAME]/rPPG-UMex}
}
```

---

## Acknowledgements

This work was supported by [ UNAM-DGAPA-PAPIIT "IT102526: Sistema robótico para la medición sin contacto de variables fisiológicas y composición corporal”]. We thank the student volunteers of [Biomedical Systems Dept. at Engineering Faculty, UNAM] for their participation. The rPPG-Toolbox framework was developed
by Liu et al. (2022) at the University of Washington.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

[Luis Jimenez-Angeles] — [luis.jimenez@ieee.org]
[Department of Biomedical Systems, Engineering Faculty, UNAM, Mexico City 04510, México]
