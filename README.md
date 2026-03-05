# Sleep Apnea Detection — SRIP 2026 Task

Detection of breathing irregularities during sleep using physiological signals.

## Project Structure

```
Project Root/
├── Data/
│   ├── AP01/
│   │   ├── nasal_airflow.txt
│   │   ├── thoracic_movement.txt
│   │   ├── spo2.txt
│   │   ├── flow_events.csv
│   │   └── sleep_profile.csv
│   └── ...
├── Visualizations/
│   └── AP01_visualization.pdf
├── Dataset/
│   └── breathing_dataset.csv
├── models/
│   └── cnn_model.py
├── scripts/
│   ├── vis.py
│   ├── create_dataset.py
│   └── train_model.py
├── results/
│   ├── lopo_results.csv
│   └── confusion_matrix_overall.png
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Visualize a participant's sleep recording
```bash
python scripts/vis.py -name "Data/AP20"
```
Generates a PDF with 4-panel plot: Nasal Airflow, Thoracic Movement, SpO2, Sleep Stages — with breathing events overlaid.

### 2. Create the preprocessed dataset
```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
Applies bandpass filtering (0.17–0.4 Hz), creates 30s windows with 50% overlap, and labels each window (Normal / Hypopnea / Apnea).

### 3. Train the 1D CNN with Leave-One-Participant-Out CV
```bash
python scripts/train_model.py -dataset_path "Dataset/breathing_dataset.csv"
```
Trains 5 models (one per fold), reports Accuracy, Precision, Recall, Confusion Matrix per fold and in aggregate.

## Methods

### Signal Processing
- **Bandpass filter** (Butterworth, order 4): 0.17–0.4 Hz for nasal airflow and thoracic movement (breathing rate range)
- **Low-pass filter**: 0.5 Hz for SpO2
- **Resampling**: SpO2 upsampled from 4 Hz → 32 Hz using polyphase resampling

### Windowing
- 30-second windows, 50% overlap
- Label assignment: if >50% of a window overlaps a breathing event → that event's label; otherwise "Normal"

### Model
- **1D CNN** with 3 convolutional blocks, batch normalization, global average pooling
- Input: (batch, 3 channels, 960 samples)
- Class-weighted cross-entropy loss to handle class imbalance

### Evaluation
- **Leave-One-Participant-Out Cross-Validation** (LOPO): 5 folds, train on 4 participants, test on 1
- Metrics: Accuracy, Precision, Recall (weighted), Confusion Matrix

## Signals

| Signal | Sampling Rate | Description |
|--------|--------------|-------------|
| Nasal Airflow | 32 Hz | Airflow at the nose/mouth |
| Thoracic Movement | 32 Hz | Chest wall expansion/contraction |
| SpO₂ | 4 Hz | Peripheral oxygen saturation |

## AI Assistance Disclosure

This project was completed with the assistance of Claude (Anthropic) as permitted by the assignment guidelines.

AI assistance was used for:
- Debugging file format parsing issues (custom timestamp formats, semicolon separators)
- Writing and fixing the visualization script (vis.py)
- Writing the preprocessing pipeline (create_dataset.py)
- Writing the 1D CNN model and training script (train_model.py, cnn_model.py)
- Resolving Git issues (removing large files from history)

All code has been reviewed and understood by the author. The author is able to explain every part of the implementation including the signal processing pipeline, CNN architecture, and Leave-One-Participant-Out cross-validation strategy.