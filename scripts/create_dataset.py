"""
create_dataset.py - Signal Preprocessing and Dataset Creation
==============================================================
This script reads physiological signals for all participants, applies
bandpass filtering, splits into 30-second windows (50% overlap), and
labels each window based on breathing event annotations.

Usage:
    python create_dataset.py -in_dir "Data" -out_dir "Dataset"

Pipeline overview:
1. For each participant folder in Data/:
   a. Load Nasal Airflow, Thoracic Movement, SpO2
   b. Apply bandpass filter (0.17 Hz – 0.4 Hz) for breathing signals
      SpO2 is low-pass filtered only (different physiological meaning)
   c. Resample SpO2 to match the 32 Hz signals (4 Hz → 32 Hz via interpolation)
   d. Slide a 30-second window with 50% overlap
   e. Label each window based on flow_events annotations

Label rule:
    - If >50% of the window overlaps a known event → assign that event label
    - Otherwise → "Normal"

Output:
    Dataset/breathing_dataset.csv  — one row per window, columns = features + label
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd


# ─────────────────────────────────────────────
# Signal loading (reused from vis.py)
# ─────────────────────────────────────────────
def parse_timestamp(ts_str, recording_start):
    """Convert 'DD.MM.YYYY HH:MM:SS,mmm' to seconds from recording_start."""
    ts_str = ts_str.strip().replace(',', '.')
    dt = pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S.%f')
    return (dt - recording_start).total_seconds()



# ─────────────────────────────────────────────
# Helper: find file with any extension
# ─────────────────────────────────────────────
def find_file(folder, basename):
    """
    Look for basename in folder trying common extensions.
    Returns the full path of the first match found.
    """
    folder = os.path.normpath(folder)
    for ext in ['.txt', '.csv', '.tsv', '.dat']:
        path = os.path.join(folder, basename + ext)
        if os.path.isfile(path):
            return path
    # Not found — list directory to help debug
    abs_folder = os.path.abspath(folder)
    try:
        files = os.listdir(abs_folder)
    except FileNotFoundError:
        files = ['<directory not found>']
    raise FileNotFoundError(
        f"Could not find '{basename}' in {abs_folder}\n"
        f"Files present: {files}"
    )

def load_signal(filepath):
    """
    Load signal file with header:
        Signal Type: ...
        Start Time: 5/30/2024 8:59:00 PM
        Sample Rate: 32
        Length: ...
        Unit:
        <blank>
        Data:
        30.05.2024 20:59:00,000; 120
        ...
    Returns DataFrame with columns: time (seconds from start), value
    """
    lines = open(filepath, encoding='utf-8', errors='replace').readlines()

    recording_start = None
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('Start Time:'):
            recording_start = pd.to_datetime(line.split(':', 1)[1].strip())
        if line.strip() == 'Data:':
            data_start_idx = i + 1
            break

    if recording_start is None:
        raise ValueError(f"Could not find Start Time in {filepath}")

    times, values = [], []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(';')
        if len(parts) < 2:
            continue
        try:
            t = parse_timestamp(parts[0], recording_start)
            v = float(parts[1].strip())
            times.append(t)
            values.append(v)
        except Exception:
            continue

    if not times:
        raise ValueError(f"No data parsed from {filepath}")

    return pd.DataFrame({'time': times, 'value': values})
def load_events(filepath):
    """
    Load event annotations:
        30.05.2024 23:48:45,119-23:49:01,408; 16; Hypopnea; N1
    Returns DataFrame with columns: start (s), end (s), label (str)
    """
    lines = open(filepath, encoding='utf-8', errors='replace').readlines()

    recording_start = None
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('Start Time:'):
            recording_start = pd.to_datetime(line.split(':', 1)[1].strip())
        if line.strip() == '' and recording_start is not None:
            data_start_idx = i + 1
            break

    if recording_start is None:
        raise ValueError(f"Could not find Start Time in {filepath}")

    starts, ends, labels = [], [], []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue
        try:
            parts = [p.strip() for p in line.split(';')]
            time_part = parts[0]
            label = parts[2].strip() if len(parts) > 2 else 'Event'
            date_str   = time_part[:10]
            time_range = time_part[11:]
            t_start_str, t_end_str = time_range.split('-')
            t_start = parse_timestamp(f"{date_str} {t_start_str}", recording_start)
            t_end   = parse_timestamp(f"{date_str} {t_end_str}",   recording_start)
            starts.append(t_start)
            ends.append(t_end)
            labels.append(label)
        except Exception:
            continue

    if not starts:
        raise ValueError(f"No events parsed from {filepath}")

    return pd.DataFrame({'start': starts, 'end': ends, 'label': labels})

# ─────────────────────────────────────────────
# Bandpass filter
# ─────────────────────────────────────────────
def bandpass_filter(signal_values, fs, low_hz=0.17, high_hz=0.4, order=4):
    """
    Apply a Butterworth bandpass filter.

    Why bandpass?
    Human breathing rate is 10–24 breaths/min = 0.17–0.4 Hz.
    Any frequency outside this range is noise (e.g., heartbeat at ~1 Hz,
    high-frequency motion artifacts).

    Parameters:
    -----------
    signal_values : np.array  — raw signal samples
    fs            : float     — sampling frequency in Hz
    low_hz        : float     — lower cutoff (default 0.17 Hz)
    high_hz       : float     — upper cutoff (default 0.40 Hz)
    order         : int       — filter order (higher = steeper roll-off)

    Returns: filtered signal as np.array
    """
    nyquist = fs / 2.0
    low = low_hz / nyquist
    high = high_hz / nyquist

    # Clamp to valid range (must be strictly between 0 and 1 for Butterworth)
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))

    if low >= high:
        print(f"  [WARNING] Invalid filter range ({low_hz}–{high_hz} Hz at fs={fs}), skipping filter")
        return signal_values

    b, a = butter(order, [low, high], btype='band')
    # filtfilt applies the filter twice (forward + backward) → zero phase distortion
    return filtfilt(b, a, signal_values)


def lowpass_filter(signal_values, fs, high_hz=0.5, order=4):
    """
    Low-pass filter for SpO2.
    SpO2 changes slowly; we just remove high-frequency noise.
    """
    nyquist = fs / 2.0
    high = min(high_hz / nyquist, 0.99)
    b, a = butter(order, high, btype='low')
    return filtfilt(b, a, signal_values)


# ─────────────────────────────────────────────
# Resampling SpO2 from 4 Hz to 32 Hz
# ─────────────────────────────────────────────
def resample_signal(signal_values, orig_fs, target_fs):
    """
    Resample a signal from orig_fs to target_fs using polyphase filtering.
    This is more accurate than simple interpolation for physiological signals.

    Example: SpO2 at 4 Hz → 32 Hz (multiply by 8)
    """
    # Find the ratio as integers for resample_poly
    ratio = target_fs / orig_fs
    up = int(target_fs)
    down = int(orig_fs)
    common = gcd(up, down)
    up //= common
    down //= common
    return resample_poly(signal_values, up, down)


# ─────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────
def create_windows(nasal_vals, thoracic_vals, spo2_vals_resampled,
                   times, window_sec=30, overlap=0.5, fs=32):
    """
    Slide a fixed-size window over the signals with given overlap.

    Parameters:
    -----------
    nasal_vals, thoracic_vals, spo2_vals_resampled : np.arrays of same length
    times           : np.array of timestamps (seconds) matching the 32 Hz grid
    window_sec      : window size in seconds (default: 30)
    overlap         : fractional overlap between consecutive windows (default: 0.5 = 50%)
    fs              : sampling frequency in Hz (default: 32)

    Returns:
    --------
    List of dicts, each with:
        - 'start_time'  : start time of window (seconds)
        - 'end_time'    : end time of window (seconds)
        - 'nasal'       : np.array of nasal airflow samples
        - 'thoracic'    : np.array of thoracic movement samples
        - 'spo2'        : np.array of SpO2 samples
    """
    window_samples = int(window_sec * fs)           # e.g. 30 * 32 = 960 samples
    step_samples = int(window_samples * (1 - overlap))  # 50% overlap → step = 480

    windows = []
    i = 0
    while i + window_samples <= len(nasal_vals):
        w_nasal = nasal_vals[i: i + window_samples]
        w_thoracic = thoracic_vals[i: i + window_samples]
        w_spo2 = spo2_vals_resampled[i: i + window_samples]
        t_start = float(times[i])
        t_end = float(times[i + window_samples - 1])

        windows.append({
            'start_time': t_start,
            'end_time': t_end,
            'nasal': w_nasal,
            'thoracic': w_thoracic,
            'spo2': w_spo2
        })
        i += step_samples

    return windows


# ─────────────────────────────────────────────
# Labeling
# ─────────────────────────────────────────────
def get_window_label(w_start, w_end, events_df, overlap_threshold=0.5):
    """
    Assign a label to a window based on event overlap.

    Rule:
    - For each breathing event, compute the overlap duration between the
      window [w_start, w_end] and the event [event_start, event_end].
    - If overlap / window_duration > 50% → label the window with that event.
    - If multiple events qualify, pick the one with highest overlap.
    - If no event qualifies → label is "Normal".

    Parameters:
    -----------
    w_start, w_end   : window start and end times in seconds
    events_df        : DataFrame with columns start, end, label
    overlap_threshold: fraction of window that must be covered (default 0.5)

    Returns: label string
    """
    window_duration = w_end - w_start
    best_label = 'Normal'
    best_overlap = 0.0

    for _, event in events_df.iterrows():
        # Compute overlap between window and event
        overlap_start = max(w_start, event['start'])
        overlap_end = min(w_end, event['end'])
        overlap_duration = max(0.0, overlap_end - overlap_start)
        overlap_fraction = overlap_duration / window_duration

        if overlap_fraction > overlap_threshold and overlap_fraction > best_overlap:
            best_overlap = overlap_fraction
            best_label = event['label']

    return best_label


# ─────────────────────────────────────────────
# Feature extraction (per window)
# ─────────────────────────────────────────────
def extract_features(window):
    """
    Extract simple statistical features from each window.
    These features will be columns in the CSV alongside the raw signal.

    Features extracted per signal channel:
    - mean, std, min, max, range, RMS (root mean square)

    Note: We also save the raw signal data for CNN input (the model
    will use raw windows, not these features). These features are
    useful for traditional ML baselines (e.g., Random Forest).
    """
    feats = {}
    for channel_name, values in [('nasal', window['nasal']),
                                   ('thoracic', window['thoracic']),
                                   ('spo2', window['spo2'])]:
        feats[f'{channel_name}_mean'] = np.mean(values)
        feats[f'{channel_name}_std'] = np.std(values)
        feats[f'{channel_name}_min'] = np.min(values)
        feats[f'{channel_name}_max'] = np.max(values)
        feats[f'{channel_name}_range'] = np.max(values) - np.min(values)
        feats[f'{channel_name}_rms'] = np.sqrt(np.mean(values ** 2))
    return feats


# ─────────────────────────────────────────────
# Process one participant
# ─────────────────────────────────────────────
def process_participant(folder_path, window_sec=30, overlap=0.5):
    """
    Full processing pipeline for one participant.

    Returns:
    --------
    List of row dicts (one per window), ready to be turned into a DataFrame.
    """
    participant_id = os.path.basename(folder_path)
    print(f"\n[create_dataset.py] Processing {participant_id}...")

    # ── Load raw signals ──────────────────────────────────────────────────────
    # find_file() tries .txt, .csv, .tsv, .dat automatically
    nasal_df    = load_signal(find_file(folder_path, 'nasal_airflow'))
    thoracic_df = load_signal(find_file(folder_path, 'thoracic_movement'))
    spo2_df     = load_signal(find_file(folder_path, 'spo2'))
    events_df   = load_events(find_file(folder_path, 'flow_events'))

    fs_resp = 32   # Nasal airflow and thoracic movement sampling rate
    fs_spo2 = 4    # SpO2 sampling rate

    # ── Align signals to a common time grid (32 Hz) ───────────────────────────
    # Use the nasal airflow timestamps as the reference grid.
    # Both nasal and thoracic are already 32 Hz, so we align by time index.
    # SpO2 at 4 Hz will be resampled up to 32 Hz.

    # Sort by time
    nasal_df = nasal_df.sort_values('time').reset_index(drop=True)
    thoracic_df = thoracic_df.sort_values('time').reset_index(drop=True)
    spo2_df = spo2_df.sort_values('time').reset_index(drop=True)

    # Use the shortest signal length to avoid padding issues
    # We interpolate thoracic to match nasal's timestamps exactly
    nasal_times = nasal_df['time'].values
    nasal_vals = nasal_df['value'].values

    # Interpolate thoracic onto nasal time grid
    thoracic_vals = np.interp(nasal_times, thoracic_df['time'].values, thoracic_df['value'].values)

    # Resample SpO2 from 4 Hz to 32 Hz, then interpolate to nasal time grid
    print(f"  Resampling SpO2 from {fs_spo2} Hz to {fs_resp} Hz...")
    spo2_resampled_vals = resample_signal(spo2_df['value'].values, fs_spo2, fs_resp)

    # Build a time array for the resampled SpO2 (evenly spaced)
    spo2_start_t = spo2_df['time'].iloc[0]
    spo2_end_t = spo2_df['time'].iloc[-1]
    spo2_resampled_times = np.linspace(spo2_start_t, spo2_end_t, len(spo2_resampled_vals))

    # Interpolate resampled SpO2 onto nasal time grid
    spo2_vals = np.interp(nasal_times, spo2_resampled_times, spo2_resampled_vals)

    # ── Apply filters ─────────────────────────────────────────────────────────
    print("  Applying bandpass filter to Nasal Airflow and Thoracic Movement...")
    nasal_filtered = bandpass_filter(nasal_vals, fs=fs_resp, low_hz=0.17, high_hz=0.4)
    thoracic_filtered = bandpass_filter(thoracic_vals, fs=fs_resp, low_hz=0.17, high_hz=0.4)

    print("  Applying low-pass filter to SpO2...")
    spo2_filtered = lowpass_filter(spo2_vals, fs=fs_resp, high_hz=0.5)

    # ── Create sliding windows ────────────────────────────────────────────────
    print(f"  Creating {window_sec}s windows with {int(overlap*100)}% overlap...")
    windows = create_windows(
        nasal_filtered, thoracic_filtered, spo2_filtered,
        nasal_times, window_sec=window_sec, overlap=overlap, fs=fs_resp
    )
    print(f"  → {len(windows)} windows created")

    # ── Label each window ─────────────────────────────────────────────────────
    print("  Labeling windows...")
    rows = []
    label_counts = {}

    for w in windows:
        label = get_window_label(w['start_time'], w['end_time'], events_df)
        label_counts[label] = label_counts.get(label, 0) + 1

        feats = extract_features(w)
        row = {
            'participant_id': participant_id,
            'start_time': w['start_time'],
            'end_time': w['end_time'],
            'label': label,
            **feats,
            # Save raw signal as flattened arrays (for CNN)
            # We save as comma-separated strings within the CSV
            'nasal_raw': ','.join(f'{v:.5f}' for v in w['nasal']),
            'thoracic_raw': ','.join(f'{v:.5f}' for v in w['thoracic']),
            'spo2_raw': ','.join(f'{v:.5f}' for v in w['spo2']),
        }
        rows.append(row)

    print(f"  Label distribution: {label_counts}")
    return rows


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(in_dir, out_dir, window_sec=30, overlap=0.5):
    """
    Process all participant folders in in_dir and save the combined dataset.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Find all participant folders (any folder starting with AP or containing signal files)
    participant_folders = sorted([
        d for d in glob.glob(os.path.join(in_dir, '*'))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'nasal_airflow.txt'))
    ])

    if not participant_folders:
        print(f"[ERROR] No participant folders found in '{in_dir}'")
        print("Expected structure: Data/AP01/nasal_airflow.txt, etc.")
        return

    print(f"Found {len(participant_folders)} participant(s): {[os.path.basename(p) for p in participant_folders]}")

    all_rows = []
    for folder in participant_folders:
        try:
            rows = process_participant(folder, window_sec=window_sec, overlap=overlap)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [WARNING] Failed to process {folder}: {e}")
            import traceback; traceback.print_exc()

    if not all_rows:
        print("[ERROR] No data processed. Check your Data directory.")
        return

    # ── Save main dataset ─────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    out_path = os.path.join(out_dir, 'breathing_dataset.csv')
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved dataset ({len(df)} windows) → {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Dataset Summary ──────────────────────────────────")
    print(f"  Total windows : {len(df)}")
    print(f"  Participants  : {df['participant_id'].nunique()}")
    print("\n  Label distribution (all participants combined):")
    print(df['label'].value_counts().to_string())
    print("\n  Per-participant label counts:")
    print(df.groupby(['participant_id', 'label']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess sleep signals and create a labeled windowed dataset.'
    )
    parser.add_argument('-in_dir', type=str, required=True,
                        help='Input directory containing participant folders (e.g., "Data")')
    parser.add_argument('-out_dir', type=str, required=True,
                        help='Output directory for the dataset (e.g., "Dataset")')
    parser.add_argument('--window_sec', type=int, default=30,
                        help='Window size in seconds (default: 30)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Window overlap fraction (default: 0.5 = 50%%)')
    args = parser.parse_args()

    main(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        window_sec=args.window_sec,
        overlap=args.overlap
    )