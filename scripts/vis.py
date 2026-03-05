"""
vis.py - Sleep Data Visualization Script
=========================================
This script reads physiological signals for one participant and generates
a multi-panel PDF visualization with annotated breathing events.

Usage:
    python vis.py -name "Data/AP20"

What this script does:
1. Loads Nasal Airflow, Thoracic Movement, SpO2 signals
2. Loads the flow_events (breathing event annotations)
3. Loads sleep_profile (sleep stages)
4. Creates a 4-panel figure showing all signals over 8 hours
5. Overlays colored shaded regions for apnea/hypopnea events
6. Saves the result as a PDF in the Visualizations/ directory
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


# ─────────────────────────────────────────────
# Helper: find a file regardless of extension
# ─────────────────────────────────────────────
def find_file(folder, basename):
    """
    Look for a file by its base name (without extension) inside a folder.
    Tries common extensions: .txt, .csv, .tsv, .dat in that order.
    Uses os.path.normpath to handle both Windows and Unix path separators.
    """
    # Normalize path so forward slashes work on Windows too
    folder = os.path.normpath(folder)

    # Try the basename as-is first (maybe it already includes extension)
    direct = os.path.join(folder, basename)
    if os.path.exists(direct):
        return direct

    # Strip any extension from basename and try all common extensions
    base_no_ext = os.path.splitext(basename)[0]
    for ext in ['.txt', '.csv', '.tsv', '.dat']:
        candidate = os.path.join(folder, base_no_ext + ext)
        if os.path.exists(candidate):
            return candidate

    # Print what we looked for to help debug
    print(f"  [DEBUG] Looked in: {os.path.abspath(folder)}")
    print(f"  [DEBUG] Files found there: {os.listdir(folder) if os.path.isdir(folder) else 'FOLDER NOT FOUND'}")
    raise FileNotFoundError(
        f"Could not find '{basename}' (or .txt/.csv/.tsv/.dat variant) in '{folder}'"
    )


# ─────────────────────────────────────────────
# Timestamp parser for the specific file format
# ─────────────────────────────────────────────
def parse_timestamp(ts_str, recording_start):
    """
    Parse a timestamp string like '30.05.2024 20:59:00,031' into
    seconds elapsed since recording_start.

    The milliseconds use a comma separator (European format): HH:MM:SS,mmm
    We replace the comma with a dot to make it parseable, then compute
    the difference from the recording start in seconds.
    """
    ts_str = ts_str.strip().replace(',', '.')
    dt = pd.to_datetime(ts_str, format='%d.%m.%Y %H:%M:%S.%f')
    return (dt - recording_start).total_seconds()


# ─────────────────────────────────────────────
# Helper: load a signal text file
# ─────────────────────────────────────────────
def load_signal(filepath):
    """
    Load a signal file in the format:
        Signal Type: Flow_TH_Type
        Start Time: 5/30/2024 8:59:00 PM
        Sample Rate: 32
        Length: 875184
        Unit:
        <blank line>
        Data:
        30.05.2024 20:59:00,000; 120
        30.05.2024 20:59:00,031; 120
        ...

    Returns DataFrame with columns:
        time  — seconds since recording start (float)
        value — signal amplitude (float)
    """
    lines = open(filepath, encoding='utf-8', errors='replace').readlines()

    # Parse recording start time from header (line 1: "Start Time: 5/30/2024 8:59:00 PM")
    recording_start = None
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('Start Time:'):
            ts = line.split(':', 1)[1].strip()
            recording_start = pd.to_datetime(ts)
        if line.strip() == 'Data:':
            data_start_idx = i + 1
            break

    if recording_start is None:
        raise ValueError(f"Could not find 'Start Time' in header of {filepath}")

    # Parse data lines: "30.05.2024 20:59:00,031; 120"
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
        raise ValueError(f"No data rows parsed from {filepath}")

    return pd.DataFrame({'time': times, 'value': values})


# ─────────────────────────────────────────────
# Helper: load flow events CSV
# ─────────────────────────────────────────────
def load_events(filepath):
    """
    Load breathing event annotations in the format:
        Signal ID: FlowD\\flow
        Start Time: 5/30/2024 8:59:00 PM
        Unit: s
        Signal Type: Impuls
        <blank line>
        30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1

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
            # Format: "30.05.2024 23:48:45,119-23:49:01,408; 16; Hypopnea; N1"
            parts = [p.strip() for p in line.split(';')]
            time_part = parts[0]   # "30.05.2024 23:48:45,119-23:49:01,408"
            label = parts[2].strip() if len(parts) > 2 else 'Event'

            date_str   = time_part[:10]   # "30.05.2024"
            time_range = time_part[11:]   # "23:48:45,119-23:49:01,408"
            t_start_str, t_end_str = time_range.split('-')

            t_start = parse_timestamp(f"{date_str} {t_start_str}", recording_start)
            t_end   = parse_timestamp(f"{date_str} {t_end_str}",   recording_start)

            starts.append(t_start)
            ends.append(t_end)
            labels.append(label)
        except Exception:
            continue

    if not starts:
        raise ValueError(f"No event rows parsed from {filepath}")

    return pd.DataFrame({'start': starts, 'end': ends, 'label': labels})
def load_sleep_profile(filepath):
    """
    Load sleep profile in the format:
        Signal ID: SchlafProfil\\profil
        Start Time: 5/30/2024 8:59:00 PM
        ...
        Rate: 30 s
        <blank line>
        30.05.2024 20:59:00,000; Wake
        30.05.2024 20:59:30,000; Wake
        ...

    Returns DataFrame with columns: time (s), stage (str)
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

    times, stages = [], []
    for line in lines[data_start_idx:]:
        line = line.strip()
        if not line:
            continue
        try:
            parts = [p.strip() for p in line.split(';')]
            t = parse_timestamp(parts[0], recording_start)
            stage = parts[1].strip() if len(parts) > 1 else 'Unknown'
            times.append(t)
            stages.append(stage)
        except Exception:
            continue

    return pd.DataFrame({'time': times, 'stage': stages})

# ─────────────────────────────────────────────
# Color coding for events
# ─────────────────────────────────────────────
EVENT_COLORS = {
    'apnea': '#e74c3c',           # red
    'obstructive apnea': '#e74c3c',
    'central apnea': '#c0392b',
    'mixed apnea': '#e67e22',
    'hypopnea': '#f39c12',        # orange
    'obstructive hypopnea': '#f39c12',
    'desaturation': '#9b59b6',    # purple
    'arousal': '#3498db',         # blue
}

def get_event_color(label):
    label_lower = str(label).lower()
    for key, color in EVENT_COLORS.items():
        if key in label_lower:
            return color
    return '#95a5a6'  # grey fallback


# ─────────────────────────────────────────────
# Main visualization function
# ─────────────────────────────────────────────
def plot_page(axes, nasal, thoracic, spo2, events, sleep_profile,
              t_start_h, t_end_h, stage_map, stage_labels,
              participant_id, total_hours, page_num, total_pages):
    """
    Draw one 5-minute window page onto the provided axes array.
    """
    for ax in axes:
        ax.cla()

    window_label = f"{t_start_h*60:.0f}–{t_end_h*60:.0f} min  ({t_start_h:.3f}h–{t_end_h:.3f}h)"

    # ── Panel 1: Nasal Airflow ────────────────────────────────────────────────
    ax1 = axes[0]
    mask = (nasal['time_h'] >= t_start_h) & (nasal['time_h'] <= t_end_h)
    ax1.plot(nasal.loc[mask, 'time_h'], nasal.loc[mask, 'value'],
             color='#2980b9', linewidth=0.6, alpha=0.9)
    ax1.set_ylabel('Nasal Airflow\n(a.u.)', fontsize=9)
    ax1.set_title('Nasal Airflow', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Thoracic Movement ────────────────────────────────────────────
    ax2 = axes[1]
    mask = (thoracic['time_h'] >= t_start_h) & (thoracic['time_h'] <= t_end_h)
    ax2.plot(thoracic.loc[mask, 'time_h'], thoracic.loc[mask, 'value'],
             color='#27ae60', linewidth=0.6, alpha=0.9)
    ax2.set_ylabel('Thoracic\nMovement (a.u.)', fontsize=9)
    ax2.set_title('Thoracic Movement', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: SpO2 ─────────────────────────────────────────────────────────
    ax3 = axes[2]
    mask = (spo2['time_h'] >= t_start_h) & (spo2['time_h'] <= t_end_h)
    spo2_win = spo2.loc[mask]
    ax3.plot(spo2_win['time_h'], spo2_win['value'],
             color='#8e44ad', linewidth=0.8, marker='.', markersize=2, alpha=0.9)
    ax3.set_ylabel('SpO₂ (%)', fontsize=9)
    ax3.set_title('Oxygen Saturation (SpO₂)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.25)
    if len(spo2_win) > 0:
        spo2_min = max(80, spo2_win['value'].min() - 2)
        spo2_max = min(102, spo2_win['value'].max() + 1)
    else:
        spo2_min, spo2_max = 80, 102
    ax3.set_ylim(spo2_min, spo2_max)

    # ── Panel 4: Sleep Stages ─────────────────────────────────────────────────
    ax4 = axes[3]
    if sleep_profile is not None:
        mask = (sleep_profile['time_h'] >= t_start_h) & (sleep_profile['time_h'] <= t_end_h + 0.02)
        sp_win = sleep_profile.loc[mask]
        if len(sp_win) >= 1:
            # Extend slightly past window so step plot fills to edge
            ax4.step(sp_win['time_h'], sp_win['stage_num'],
                     color='#2c3e50', linewidth=1.4, where='post')
    ax4.set_yticks(list(stage_labels.keys()))
    ax4.set_yticklabels(list(stage_labels.values()), fontsize=8)
    ax4.set_ylabel('Sleep Stage', fontsize=9)
    ax4.set_title('Sleep Stages', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.25)
    ax4.set_ylim(-0.5, 4.5)
    ax4.invert_yaxis()

    # ── Overlay breathing events on ALL panels ────────────────────────────────
    legend_handles = {}
    win_events = events[(events['end_h'] >= t_start_h) & (events['start_h'] <= t_end_h)]
    for _, ev in win_events.iterrows():
        color = get_event_color(ev['label'])
        for ax in axes:
            ax.axvspan(max(ev['start_h'], t_start_h),
                       min(ev['end_h'],   t_end_h),
                       alpha=0.25, color=color, linewidth=0)
        if ev['label'] not in legend_handles:
            legend_handles[ev['label']] = mpatches.Patch(
                color=color, alpha=0.5, label=ev['label'])

    # ── X-axis: fine-grained ticks every 30 seconds within the 5-min window ──
    tick_interval_h = 30 / 3600.0   # 30 seconds in hours
    x_ticks = np.arange(t_start_h, t_end_h + tick_interval_h/2, tick_interval_h)
    for ax in axes:
        ax.set_xlim(t_start_h, t_end_h)
        ax.set_xticks(x_ticks)
    axes[-1].set_xticklabels(
        [f"{int((t*3600)//60):02d}:{int((t*3600)%60):02d}" for t in x_ticks],
        fontsize=7, rotation=45
    )
    for ax in axes[:-1]:
        ax.set_xticklabels([])
    axes[-1].set_xlabel('Time (MM:SS from recording start)', fontsize=9)

    return legend_handles


def visualize_participant(participant_folder, output_dir='Visualizations',
                          window_minutes=5):
    """
    Generate a multi-page PDF for one participant.
    Each page covers `window_minutes` of the recording (default: 5 min).

    Parameters
    ----------
    participant_folder : str
        Path to folder containing signal files (e.g. "Data/AP01")
    output_dir : str
        Directory where the PDF will be saved
    window_minutes : int
        Length of each page window in minutes (default 5)
    """
    participant_id = os.path.basename(participant_folder)
    print(f"[vis.py] Processing participant: {participant_id}")

    # ── Load signals ──────────────────────────────────────────────────────────
    nasal_path    = find_file(participant_folder, 'nasal_airflow')
    thoracic_path = find_file(participant_folder, 'thoracic_movement')
    spo2_path     = find_file(participant_folder, 'spo2')
    events_path   = find_file(participant_folder, 'flow_events')

    print("  Loading nasal airflow...")
    nasal = load_signal(nasal_path)
    print(f"    → {len(nasal)} samples")

    print("  Loading thoracic movement...")
    thoracic = load_signal(thoracic_path)

    print("  Loading SpO2...")
    spo2 = load_signal(spo2_path)

    print("  Loading flow events...")
    events = load_events(events_path)
    print(f"    → {len(events)} events")

    sleep_profile = None
    try:
        sleep_path = find_file(participant_folder, 'sleep_profile')
        print("  Loading sleep profile...")
        sleep_profile = load_sleep_profile(sleep_path)
    except FileNotFoundError:
        print("  No sleep_profile file found — skipping sleep stage panel")

    # ── Convert to hours ──────────────────────────────────────────────────────
    for df in [nasal, thoracic, spo2]:
        df['time_h'] = df['time'] / 3600.0
    events['start_h'] = events['start'] / 3600.0
    events['end_h']   = events['end']   / 3600.0

    stage_map    = {'wake': 0, 'w': 0, 'rem': 1, 'n1': 2, 'n2': 3,
                    'n3': 4, 'n4': 4, 'sws': 4, 'movement': 0}
    stage_labels = {0: 'Wake', 1: 'REM', 2: 'N1', 3: 'N2', 4: 'N3/SWS'}

    if sleep_profile is not None:
        sleep_profile['time_h']   = sleep_profile['time'] / 3600.0
        sleep_profile['stage_num'] = (sleep_profile['stage']
                                        .str.strip().str.lower()
                                        .map(stage_map).fillna(2))

    t_max = max(nasal['time_h'].max(),
                thoracic['time_h'].max(),
                spo2['time_h'].max())

    window_h  = window_minutes / 60.0
    windows   = np.arange(0, t_max, window_h)
    total_pages = len(windows)
    print(f"  Generating {total_pages} pages ({window_minutes}-min windows)...")

    # ── Create figure with 4 panels (reused across pages) ────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=False)
    fig.subplots_adjust(hspace=0.35)

    # ── Save multi-page PDF ───────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{participant_id}_visualization.pdf')

    with PdfPages(output_path) as pdf:
        for page_num, t_start_h in enumerate(windows, start=1):
            t_end_h = min(t_start_h + window_h, t_max)

            fig.suptitle(
                f'Overnight Sleep Recording — Participant {participant_id}\n'
                f'Page {page_num}/{total_pages}  |  '
                f'{t_start_h*60:.1f}–{t_end_h*60:.1f} min  '
                f'(Total duration: {t_max:.2f} h)',
                fontsize=12, fontweight='bold', y=0.99
            )

            legend_handles = plot_page(
                axes, nasal, thoracic, spo2, events, sleep_profile,
                t_start_h, t_end_h, stage_map, stage_labels,
                participant_id, t_max, page_num, total_pages
            )

            # Shared legend at bottom
            if legend_handles:
                fig.legend(
                    handles=list(legend_handles.values()),
                    loc='lower center', ncol=5, fontsize=8,
                    title='Breathing Events', title_fontsize=9,
                    bbox_to_anchor=(0.5, 0.001), framealpha=0.9
                )

            pdf.savefig(fig, bbox_inches='tight')

            if page_num % 10 == 0:
                print(f"    Page {page_num}/{total_pages}...")

        # PDF metadata
        d = pdf.infodict()
        d['Title']   = f'Sleep Recording — {participant_id}'
        d['Author']  = 'vis.py'
        d['Subject'] = 'Overnight PSG — 5-minute paginated view'

    plt.close(fig)
    print(f"\n✓ Saved {total_pages}-page PDF → {output_path}")
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate PDF visualization of overnight sleep data for one participant.'
    )
    parser.add_argument(
        '-name',
        type=str,
        required=True,
        help='Path to participant folder, e.g. "Data/AP20"'
    )
    parser.add_argument(
        '-out_dir',
        type=str,
        default='Visualizations',
        help='Output directory for the PDF (default: Visualizations/)'
    )
    args = parser.parse_args()

    visualize_participant(
        participant_folder=args.name,
        output_dir=args.out_dir
    )