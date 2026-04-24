#!/usr/bin/env python3
"""
augment_53tet.py — 53-TET Transposition Augmentation for SymbTr MIDI files.

Performs Holdrian-comma transpositions in the MIDI domain.
See plans/plan_augmentation.md for full specification.

Usage:
    python augment_53tet.py all     --input data/augmentation/test_set
    python augment_53tet.py stats   --input data/augmentation/test_set
    python augment_53tet.py augment --input data/augmentation/test_set
    python augment_53tet.py verify  --input data/augmentation/test_set
    python augment_53tet.py poststats --input data/symbtr/aug --progress-every 1
"""

import argparse
import copy
import glob
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import mido
import numpy as np

# ─── Project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from llama_recipes.datasets.music_tokenizer import (
    MusicTokenizer,
    canonical_pitch_to_midi_and_pitchbend,
    pitch_to_octave_pitch_class_microtonal,
    pitchbend_to_semitones,
)

# ─── Constants ───────────────────────────────────────────────────────────────
TET53_STEPS = 53
HOLDRIAN_COMMA_CENTS = 1200.0 / TET53_STEPS  # ≈ 22.6415 cents
PITCHBEND_SENSITIVITY = 2.0
MIDI_NOTE_MIN, MIDI_NOTE_MAX = 0, 127
PITCHBEND_MIN, PITCHBEND_MAX = -8192, 8191

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('augment_53tet')


# ═══════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════════

def collect_midi_files(input_dir):
    """Return sorted list of .mid files in input_dir (non-recursive)."""
    files = sorted(glob.glob(os.path.join(input_dir, '*.mid')))
    if not files:
        logger.error(f"No .mid files found in {input_dir}")
    return files


def parse_notes_from_midi(midi_path):
    """Parse a MIDI file and return per-note info with pitchbend state.

    Returns list of dicts with keys:
      - track_idx, msg_idx: location in the MIDI structure
      - channel, midi_note, velocity, time (absolute seconds)
      - pitchbend_at_onset: the pitchbend value active on that channel at note_on
      - canonical_pitch: float semitones (midi_note + bend offset)
      - pitchwheel_msg_idx: index of the last pitchwheel msg before this note_on
        on the same channel (or None)
    Also returns a summary dict with channel/instrument/pitchwheel stats.
    """
    mid = mido.MidiFile(midi_path)
    notes = []
    stats = {
        'channels_with_notes': set(),
        'channels_with_pitchwheel': set(),
        'instruments': defaultdict(int),
        'pitchwheel_count': 0,
        'note_on_count': 0,
        'pitchwheel_note_pairing': [],  # list of (gap_in_messages) for each note_on
        'note0_count': 0,
    }

    for track_idx, track in enumerate(mid.tracks):
        pitchbend_state = defaultdict(int)  # channel -> last pitchbend value
        last_pw_msg_idx = {}  # channel -> msg_idx of last pitchwheel
        instruments = defaultdict(int)
        time = 0.0

        for msg_idx, msg in enumerate(track):
            time += msg.time
            if msg.type == 'program_change':
                instruments[msg.channel] = msg.program
            elif msg.type == 'pitchwheel':
                pitchbend_state[msg.channel] = msg.pitch
                last_pw_msg_idx[msg.channel] = msg_idx
                stats['pitchwheel_count'] += 1
                stats['channels_with_pitchwheel'].add(msg.channel)
            elif msg.type == 'note_on' and msg.velocity > 0:
                ch = msg.channel
                stats['channels_with_notes'].add(ch)
                stats['note_on_count'] += 1

                if msg.note == 0:
                    stats['note0_count'] += 1
                    continue  # skip rest markers

                bend_val = pitchbend_state[ch]
                bend_semi = pitchbend_to_semitones(bend_val, PITCHBEND_SENSITIVITY)
                canonical = msg.note + bend_semi

                pw_idx = last_pw_msg_idx.get(ch, None)
                # Measure gap: how many messages between last pitchwheel and this note_on
                if pw_idx is not None:
                    gap = msg_idx - pw_idx
                else:
                    gap = -1  # no preceding pitchwheel
                stats['pitchwheel_note_pairing'].append(gap)

                instr = 128 if ch == 9 else instruments.get(ch, 0)
                stats['instruments'][instr] += 1

                notes.append({
                    'track_idx': track_idx,
                    'msg_idx': msg_idx,
                    'channel': ch,
                    'midi_note': msg.note,
                    'velocity': msg.velocity,
                    'time': time,
                    'pitchbend_at_onset': bend_val,
                    'canonical_pitch': canonical,
                    'pitchwheel_msg_idx': pw_idx,
                    'instrument': instr,
                })

    return notes, stats


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _iter_canonical_pitches(midi_path, sensitivity=PITCHBEND_SENSITIVITY):
    """Yield canonical pitches (in semitones) for note_on events in a MIDI file.

    This is a lightweight streaming parser used by poststats to avoid retaining
    large per-note structures in memory.
    """
    mid = mido.MidiFile(midi_path)
    for track in mid.tracks:
        pitchbend_state = defaultdict(int)
        for msg in track:
            if msg.type == 'pitchwheel':
                pitchbend_state[msg.channel] = msg.pitch
            elif msg.type == 'note_on' and msg.velocity > 0:
                if msg.note == 0:
                    continue
                bend_semi = pitchbend_to_semitones(pitchbend_state[msg.channel], sensitivity)
                yield msg.note + bend_semi


def _accumulate_pitch_stats(file_paths, label, progress_every=100):
    """Accumulate pitch statistics in a streaming, memory-efficient way."""
    pc_hist = np.zeros(1200, dtype=np.int64)
    full_hist = np.zeros(1280, dtype=np.int64)  # 0..12800 cents in 10-cent bins

    total_notes = 0
    min_pitch = None
    max_pitch = None
    errors = []
    processed = 0

    for idx, fpath in enumerate(file_paths, start=1):
        try:
            for canonical in _iter_canonical_pitches(fpath):
                total_notes += 1

                if min_pitch is None or canonical < min_pitch:
                    min_pitch = canonical
                if max_pitch is None or canonical > max_pitch:
                    max_pitch = canonical

                pc_bin = int(round((canonical % 12) * 100))
                if pc_bin > 1199:
                    pc_bin = 1199
                elif pc_bin < 0:
                    pc_bin = 0
                pc_hist[pc_bin] += 1

                cent_val = canonical * 100
                # Match previous plot domain: bins over [0, 12800) with width 10 cents.
                full_bin = int(cent_val // 10)
                if 0 <= full_bin < 1280:
                    full_hist[full_bin] += 1

            processed += 1
            if progress_every > 0 and (idx % progress_every == 0 or idx == len(file_paths)):
                print(f"  [{label}] processed {idx}/{len(file_paths)} files")
        except Exception as e:
            errors.append(f"{os.path.basename(fpath)}\t{e}")

    return {
        'processed_files': processed,
        'failed_files': len(errors),
        'errors': errors,
        'total_notes': total_notes,
        'min_pitch': min_pitch,
        'max_pitch': max_pitch,
        'pc_hist': pc_hist,
        'full_hist': full_hist,
        'unique_bins': int(np.count_nonzero(pc_hist)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Dataset Statistics & Exploration
# ═══════════════════════════════════════════════════════════════════════════════

def run_stats(input_dir):
    """Phase 1: Collect and plot statistics for all MIDI files in input_dir."""
    midi_files = collect_midi_files(input_dir)
    if not midi_files:
        return

    aug_dir = os.path.join(input_dir, 'aug')
    stats_dir = os.path.join(aug_dir, 'stats')
    ensure_dir(stats_dir)

    all_canonical_pitches = []
    all_pitchbend_values = []
    all_midi_notes = []
    file_reports = []

    print(f"\n{'='*70}")
    print(f"PHASE 1: Dataset Statistics — {len(midi_files)} files")
    print(f"{'='*70}\n")

    for fpath in midi_files:
        fname = os.path.basename(fpath)
        notes, file_stats = parse_notes_from_midi(fpath)

        canonical_pitches = [n['canonical_pitch'] for n in notes]
        pitchbend_values = [n['pitchbend_at_onset'] for n in notes]
        midi_notes_list = [n['midi_note'] for n in notes]

        all_canonical_pitches.extend(canonical_pitches)
        all_pitchbend_values.extend(pitchbend_values)
        all_midi_notes.extend(midi_notes_list)

        # Pitchwheel-per-note pairing
        pairing = file_stats['pitchwheel_note_pairing']
        pairing_1to1 = sum(1 for g in pairing if g == 1)
        pairing_other = sum(1 for g in pairing if g != 1)

        report = {
            'file': fname,
            'note_count': len(notes),
            'note0_count': file_stats['note0_count'],
            'pitchwheel_events': file_stats['pitchwheel_count'],
            'note_on_events': file_stats['note_on_count'],
            'midi_note_min': min(midi_notes_list) if midi_notes_list else None,
            'midi_note_max': max(midi_notes_list) if midi_notes_list else None,
            'canonical_min': min(canonical_pitches) if canonical_pitches else None,
            'canonical_max': max(canonical_pitches) if canonical_pitches else None,
            'channels_notes': sorted(file_stats['channels_with_notes']),
            'channels_pitchwheel': sorted(file_stats['channels_with_pitchwheel']),
            'instruments': dict(file_stats['instruments']),
            'pw_note_1to1': pairing_1to1,
            'pw_note_other': pairing_other,
        }
        file_reports.append(report)

        print(f"  {fname}")
        print(f"    Notes: {report['note_count']}, Note-0 rests: {report['note0_count']}")
        print(f"    MIDI note range: [{report['midi_note_min']}, {report['midi_note_max']}]")
        print(f"    Canonical pitch range: [{report['canonical_min']:.2f}, {report['canonical_max']:.2f}] semitones")
        print(f"    Pitchwheel events: {report['pitchwheel_events']}, Note-on events: {report['note_on_events']}")
        print(f"    PW→note_on gap=1 (1:1): {pairing_1to1}, other: {pairing_other}")
        print(f"    Channels (notes): {report['channels_notes']}, Channels (PW): {report['channels_pitchwheel']}")
        print(f"    Instruments: {report['instruments']}")
        print()

    # ── Summary ──
    print(f"\n{'─'*70}")
    print(f"SUMMARY across {len(midi_files)} files:")
    total_notes = sum(r['note_count'] for r in file_reports)
    print(f"  Total notes (excl. note-0): {total_notes}")
    print(f"  Total note-0 rests filtered: {sum(r['note0_count'] for r in file_reports)}")
    if all_midi_notes:
        print(f"  Global MIDI note range: [{min(all_midi_notes)}, {max(all_midi_notes)}]")
        print(f"  Global canonical pitch range: [{min(all_canonical_pitches):.2f}, {max(all_canonical_pitches):.2f}] semitones")

    total_1to1 = sum(r['pw_note_1to1'] for r in file_reports)
    total_other = sum(r['pw_note_other'] for r in file_reports)
    print(f"  Pitchwheel→note_on pairing: 1:1={total_1to1}, other={total_other}")
    if total_other > 0:
        print(f"    ⚠ Not all pitchwheel events are 1:1 with note_on — using robust approach.")
    else:
        print(f"    ✓ All pitchwheel events are 1:1 with note_on.")

    # ── Predict out-of-range transpositions ──
    print(f"\n  Transposition range analysis:")
    if all_midi_notes:
        min_note = min(all_midi_notes)
        max_note = max(all_midi_notes)
        # Max shift at k=52: 52 * 22.6415 / 100 semitones ≈ 11.77 semitones
        max_shift_semi = (TET53_STEPS - 1) * HOLDRIAN_COMMA_CENTS / 100
        print(f"    Lowest MIDI note: {min_note}, Highest: {max_note}")
        print(f"    Max transposition shift: {max_shift_semi:.2f} semitones (k=52)")
        print(f"    Highest note after max shift: {max_note + max_shift_semi:.2f} → MIDI {int(round(max_note + max_shift_semi))}")
        if max_note + max_shift_semi > 127:
            first_problem_k = None
            for k in range(TET53_STEPS):
                shift = k * HOLDRIAN_COMMA_CENTS / 100
                if max_note + shift > 127.5:
                    first_problem_k = k
                    break
            print(f"    ⚠ Some transpositions (k≥{first_problem_k}) may push notes out of range.")
        else:
            print(f"    ✓ All transpositions should be in range.")

    # ── Plots ──
    if all_canonical_pitches:
        # Canonical pitch histogram (in cents)
        cents = [p * 100 for p in all_canonical_pitches]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.hist(cents, bins=range(0, 12800, 10), edgecolor='none', alpha=0.8)
        ax.set_xlabel('Canonical Pitch (cents)')
        ax.set_ylabel('Count')
        ax.set_title(f'Canonical Pitch Distribution — {len(midi_files)} Original Files ({total_notes} notes)')
        plt.tight_layout()
        fig.savefig(os.path.join(stats_dir, 'pitch_distribution_original.png'), dpi=150)
        plt.close(fig)
        print(f"\n  Saved: pitch_distribution_original.png")

        # Pitch class cents histogram (mod 1200)
        pc_cents = [int(round((p % 12) * 100)) for p in all_canonical_pitches]
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.hist(pc_cents, bins=range(0, 1201, 1), edgecolor='none', alpha=0.8, color='orange')
        ax.set_xlabel('Pitch Class (cents, 0–1199)')
        ax.set_ylabel('Count')
        ax.set_title(f'Pitch Class Distribution (mod 1200 cents) — Original')
        plt.tight_layout()
        fig.savefig(os.path.join(stats_dir, 'pitch_class_distribution_original.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved: pitch_class_distribution_original.png")

        unique_cent_bins = len(set(pc_cents))
        print(f"  Unique cent bins occupied: {unique_cent_bins} / 1200")

    if all_pitchbend_values:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(all_pitchbend_values, bins=200, edgecolor='none', alpha=0.8, color='green')
        ax.set_xlabel('Pitchbend Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Pitchbend Distribution — Original')
        plt.tight_layout()
        fig.savefig(os.path.join(stats_dir, 'pitchbend_distribution_original.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved: pitchbend_distribution_original.png")

    # Save raw stats as JSON
    stats_json = {
        'files': file_reports,
        'total_notes': total_notes,
        'unique_cent_bins': len(set(int(round((p % 12) * 100)) for p in all_canonical_pitches)) if all_canonical_pitches else 0,
    }
    with open(os.path.join(stats_dir, 'phase1_stats.json'), 'w') as f:
        json.dump(stats_json, f, indent=2, default=str)
    print(f"  Saved: phase1_stats.json")

    print(f"\n{'='*70}\n")
    return stats_json


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: 53-TET Transposition
# ═══════════════════════════════════════════════════════════════════════════════

def transpose_midi_53tet(midi_path, k, sensitivity=PITCHBEND_SENSITIVITY):
    """Transpose a MIDI file by k Holdrian commas in 53-TET.

    Uses the robust approach: re-emit pitchwheel before each note_on.
    Removes original pitchwheel events and inserts new ones at each note_on.

    Returns:
        (new_mid, None) on success,
        (None, reason_str) if any note goes out of range.
    """
    shift_semitones = k * HOLDRIAN_COMMA_CENTS / 100.0
    mid = mido.MidiFile(midi_path)
    new_mid = copy.deepcopy(mid)

    # Validate first: check all notes
    for track_idx, track in enumerate(mid.tracks):
        pitchbend_state = defaultdict(int)
        for msg in track:
            if msg.type == 'pitchwheel':
                pitchbend_state[msg.channel] = msg.pitch
            elif msg.type == 'note_on' and msg.velocity > 0:
                if msg.note == 0:
                    continue  # rest marker
                bend_semi = pitchbend_to_semitones(pitchbend_state[msg.channel], sensitivity)
                canonical = msg.note + bend_semi
                canonical_new = canonical + shift_semitones

                midi_note_new = int(round(canonical_new))
                if midi_note_new < MIDI_NOTE_MIN or midi_note_new > MIDI_NOTE_MAX:
                    return None, (f"note out of range: track={track_idx}, "
                                  f"orig_note={msg.note}, canonical={canonical:.3f}, "
                                  f"new_canonical={canonical_new:.3f}, "
                                  f"new_midi_note={midi_note_new}")

                residual = canonical_new - midi_note_new
                pitchbend_new = int(round((residual / sensitivity) * 8192))
                if pitchbend_new < PITCHBEND_MIN or pitchbend_new > PITCHBEND_MAX:
                    return None, (f"pitchbend out of range: track={track_idx}, "
                                  f"orig_note={msg.note}, "
                                  f"new_pitchbend={pitchbend_new}")

    # Now apply the transposition
    for track_idx, track in enumerate(new_mid.tracks):
        pitchbend_state = defaultdict(int)  # channel -> current raw pitchbend
        new_track = []
        for msg in track:
            if msg.type == 'pitchwheel':
                # Track the state but DON'T emit the original pitchwheel.
                # We will record the delta time because we can't lose it.
                pitchbend_state[msg.channel] = msg.pitch
                # Transfer this message's delta time to the next message.
                # We do this by carrying over the time.
                # Mark for removal (we'll handle timing below).
                new_track.append(('REMOVE_PW', msg))
            elif msg.type in ('note_on', 'note_off'):
                ch = msg.channel
                if msg.note == 0:
                    # Rest marker — keep as-is
                    new_track.append(('KEEP', msg))
                    continue

                bend_semi = pitchbend_to_semitones(pitchbend_state[ch], sensitivity)
                canonical = msg.note + bend_semi
                canonical_new = canonical + shift_semitones

                midi_note_new = int(round(canonical_new))
                residual = canonical_new - midi_note_new
                pitchbend_new = int(round((residual / sensitivity) * 8192))
                pitchbend_new = max(PITCHBEND_MIN, min(PITCHBEND_MAX, pitchbend_new))
                midi_note_new = max(MIDI_NOTE_MIN, min(MIDI_NOTE_MAX, midi_note_new))

                msg.note = midi_note_new

                if msg.type == 'note_on' and msg.velocity > 0:
                    # Insert pitchwheel before note_on
                    pw_msg = mido.Message('pitchwheel', channel=ch, pitch=pitchbend_new, time=0)
                    new_track.append(('INSERT_PW', pw_msg, msg))
                else:
                    new_track.append(('KEEP', msg))
            else:
                new_track.append(('KEEP', msg))

        # Rebuild the track, fixing delta times
        rebuilt = mido.MidiTrack()
        accumulated_time = 0
        for item in new_track:
            if item[0] == 'REMOVE_PW':
                # Carry the time from the removed pitchwheel to the next event
                accumulated_time += item[1].time
            elif item[0] == 'INSERT_PW':
                pw_msg, note_msg = item[1], item[2]
                # The note_msg's original delta time + any accumulated time from removed PW
                pw_msg.time = note_msg.time + accumulated_time
                note_msg.time = 0  # follows immediately after pitchwheel
                accumulated_time = 0
                rebuilt.append(pw_msg)
                rebuilt.append(note_msg)
            elif item[0] == 'KEEP':
                msg = item[1]
                msg.time += accumulated_time
                accumulated_time = 0
                rebuilt.append(msg)

        new_mid.tracks[track_idx] = rebuilt

    return new_mid, None


def run_augment(input_dir):
    """Phase 2: Generate 53-TET transpositions for all MIDI files."""
    midi_files = collect_midi_files(input_dir)
    if not midi_files:
        return

    aug_dir = os.path.join(input_dir, 'aug')
    ensure_dir(aug_dir)

    skipped_log_path = os.path.join(aug_dir, 'skipped.log')
    summary_log_path = os.path.join(aug_dir, 'summary.log')

    total_generated = 0
    total_skipped = 0
    skipped_entries = []

    print(f"\n{'='*70}")
    print(f"PHASE 2: 53-TET Transposition — {len(midi_files)} files × {TET53_STEPS} rotations")
    print(f"{'='*70}\n")

    for fpath in midi_files:
        stem = Path(fpath).stem
        fname = os.path.basename(fpath)
        file_gen = 0
        file_skip = 0

        for k in range(TET53_STEPS):
            out_name = f"{stem}_rot53_{k:02d}.mid"
            out_path = os.path.join(aug_dir, out_name)

            new_mid, reason = transpose_midi_53tet(fpath, k)
            if new_mid is not None:
                new_mid.save(out_path)
                file_gen += 1
                total_generated += 1
            else:
                file_skip += 1
                total_skipped += 1
                skipped_entries.append(f"{fname}\tk={k}\t{reason}")

        print(f"  {fname}: {file_gen} generated, {file_skip} skipped")

    # Write logs
    with open(skipped_log_path, 'w') as f:
        for entry in skipped_entries:
            f.write(entry + '\n')

    summary_text = (
        f"Total files processed: {len(midi_files)}\n"
        f"Total transpositions generated: {total_generated}\n"
        f"Total transpositions skipped: {total_skipped}\n"
        f"Expected maximum: {len(midi_files) * TET53_STEPS}\n"
    )
    with open(summary_log_path, 'w') as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"Skipped log: {skipped_log_path}")
    print(f"Summary log: {summary_log_path}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Integrity Verification
# ═══════════════════════════════════════════════════════════════════════════════

def run_verify(input_dir):
    """Phase 3 + 4: Integrity verification and pitchbend correctness."""
    aug_dir = os.path.join(input_dir, 'aug')
    stats_dir = os.path.join(aug_dir, 'stats')
    ensure_dir(stats_dir)

    aug_files = sorted(glob.glob(os.path.join(aug_dir, '*_rot53_*.mid')))
    if not aug_files:
        logger.error(f"No augmented files found in {aug_dir}")
        return

    orig_files = collect_midi_files(input_dir)

    print(f"\n{'='*70}")
    print(f"PHASE 3: Integrity Verification — {len(aug_files)} augmented files")
    print(f"{'='*70}\n")

    # ── Phase 3: Basic integrity ──
    tokenizer = MusicTokenizer(
        timeshift_vocab_size=1026,
        dur_vocab_size=1026,
        octave_vocab_size=13,
        pitch_class_vocab_size=1202,
        instrument_vocab_size=131,
        velocity_vocab_size=130,
        microtonal=True,
        pitchbend_sensitivity=PITCHBEND_SENSITIVITY,
    )

    failed = []
    pass_count = 0

    for fpath in aug_files:
        fname = os.path.basename(fpath)
        errors = []

        # 1. MIDI parse check
        try:
            mid = mido.MidiFile(fpath)
        except Exception as e:
            errors.append(f"MIDI parse error: {e}")

        # 2. Tokenization check
        try:
            compounds = tokenizer.midi_to_compound(fpath)
            if not compounds:
                errors.append("Tokenization returned empty compounds")
        except Exception as e:
            errors.append(f"Tokenization error: {e}")

        # 3. Range check (octave in [0,10], pitch_class_cents in [0,1199])
        if not errors:
            for idx, c in enumerate(compounds):
                octave, pc = c[2], c[3]
                if octave < 0 or octave > 10:
                    errors.append(f"Octave out of range: note {idx}, octave={octave}")
                    break
                if pc < 0 or pc > 1199:
                    errors.append(f"Pitch class out of range: note {idx}, pc={pc}")
                    break

        if errors:
            failed.append((fname, errors))
        else:
            pass_count += 1

    failed_log_path = os.path.join(aug_dir, 'failed_integrity.log')
    with open(failed_log_path, 'w') as f:
        for fname, errs in failed:
            for e in errs:
                f.write(f"{fname}\t{e}\n")

    print(f"  Integrity: {pass_count} passed, {len(failed)} failed")
    if failed:
        print(f"  Failed files logged to: {failed_log_path}")
        for fname, errs in failed[:5]:
            print(f"    {fname}: {errs[0]}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")
    else:
        print(f"  ✓ All {pass_count} files passed integrity checks.")

    # ── Phase 4: Pitchbend correctness ──
    print(f"\n{'='*70}")
    print(f"PHASE 4: Pitchbend Correctness Testing")
    print(f"{'='*70}\n")

    # Select a subset for detailed testing: up to 3 files × 5 transpositions
    test_ks = [1, 7, 13, 26, 52]
    test_orig_files = orig_files[:3]  # first 3 original files
    test_results = []

    for orig_path in test_orig_files:
        orig_stem = Path(orig_path).stem
        orig_notes, _ = parse_notes_from_midi(orig_path)
        if not orig_notes:
            continue

        orig_canonical = [n['canonical_pitch'] for n in orig_notes]

        for k in test_ks:
            aug_name = f"{orig_stem}_rot53_{k:02d}.mid"
            aug_path = os.path.join(aug_dir, aug_name)
            if not os.path.exists(aug_path):
                test_results.append({
                    'file': orig_stem, 'k': k, 'status': 'SKIP',
                    'reason': 'augmented file not found (likely out-of-range skip)',
                })
                continue

            aug_notes, _ = parse_notes_from_midi(aug_path)
            aug_canonical = [n['canonical_pitch'] for n in aug_notes]

            expected_shift = k * HOLDRIAN_COMMA_CENTS / 100.0
            result = {'file': orig_stem, 'k': k, 'expected_shift': expected_shift}

            # Check note count
            if len(orig_canonical) != len(aug_canonical):
                result['status'] = 'FAIL'
                result['reason'] = f"Note count mismatch: orig={len(orig_canonical)}, aug={len(aug_canonical)}"
                test_results.append(result)
                continue

            # Check canonical pitch shift
            max_pitch_error = 0.0
            for i, (o, a) in enumerate(zip(orig_canonical, aug_canonical)):
                error = abs(a - (o + expected_shift))
                max_pitch_error = max(max_pitch_error, error)

            # Check interval preservation
            max_interval_error = 0.0
            if len(orig_canonical) > 1:
                orig_intervals = [orig_canonical[i+1] - orig_canonical[i] for i in range(len(orig_canonical)-1)]
                aug_intervals = [aug_canonical[i+1] - aug_canonical[i] for i in range(len(aug_canonical)-1)]
                for oi, ai in zip(orig_intervals, aug_intervals):
                    max_interval_error = max(max_interval_error, abs(ai - oi))

            result['max_pitch_error_cents'] = max_pitch_error * 100
            result['max_interval_error_cents'] = max_interval_error * 100

            # Tolerance: 0.5 cents for pitch, 0.01 semitones (1 cent) for intervals
            pitch_ok = max_pitch_error * 100 < 0.5
            interval_ok = max_interval_error < 0.01

            if pitch_ok and interval_ok:
                result['status'] = 'PASS'
            else:
                result['status'] = 'FAIL'
                result['reason'] = (f"pitch_err={max_pitch_error*100:.4f}¢, "
                                    f"interval_err={max_interval_error*100:.4f}¢")

            test_results.append(result)

    # Print results
    pass_count_p4 = sum(1 for r in test_results if r['status'] == 'PASS')
    fail_count_p4 = sum(1 for r in test_results if r['status'] == 'FAIL')
    skip_count_p4 = sum(1 for r in test_results if r['status'] == 'SKIP')

    for r in test_results:
        status_str = {'PASS': '✓', 'FAIL': '✗', 'SKIP': '⊘'}[r['status']]
        line = f"  {status_str} {r['file']} k={r['k']}"
        if 'max_pitch_error_cents' in r:
            line += f"  pitch_err={r['max_pitch_error_cents']:.4f}¢  interval_err={r['max_interval_error_cents']:.4f}¢"
        if 'reason' in r:
            line += f"  ({r['reason']})"
        print(line)

    print(f"\n  Summary: {pass_count_p4} PASS, {fail_count_p4} FAIL, {skip_count_p4} SKIP")

    # ── Phase 4 Plot: Original vs. Transposed pitchbend values ──
    # Plot for one file, one k
    if test_orig_files:
        sample_orig = test_orig_files[0]
        sample_stem = Path(sample_orig).stem
        sample_k = test_ks[0]
        sample_aug_path = os.path.join(aug_dir, f"{sample_stem}_rot53_{sample_k:02d}.mid")

        if os.path.exists(sample_aug_path):
            orig_notes_sample, _ = parse_notes_from_midi(sample_orig)
            aug_notes_sample, _ = parse_notes_from_midi(sample_aug_path)

            if orig_notes_sample and aug_notes_sample:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Pitchbend comparison
                orig_pb = [n['pitchbend_at_onset'] for n in orig_notes_sample]
                aug_pb = [n['pitchbend_at_onset'] for n in aug_notes_sample]
                axes[0].hist(orig_pb, bins=100, alpha=0.6, label='Original')
                axes[0].hist(aug_pb, bins=100, alpha=0.6, label=f'Transposed k={sample_k}')
                axes[0].set_xlabel('Pitchbend Value')
                axes[0].set_ylabel('Count')
                axes[0].set_title('Pitchbend Distribution')
                axes[0].legend()

                # Canonical pitch comparison
                orig_cp = [n['canonical_pitch'] * 100 for n in orig_notes_sample]
                aug_cp = [n['canonical_pitch'] * 100 for n in aug_notes_sample]
                axes[1].hist(orig_cp, bins=100, alpha=0.6, label='Original')
                axes[1].hist(aug_cp, bins=100, alpha=0.6, label=f'Transposed k={sample_k}')
                axes[1].set_xlabel('Canonical Pitch (cents)')
                axes[1].set_ylabel('Count')
                axes[1].set_title('Canonical Pitch Distribution')
                axes[1].legend()

                plt.tight_layout()
                fig.savefig(os.path.join(stats_dir, 'pitchbend_correctness_test.png'), dpi=150)
                plt.close(fig)
                print(f"\n  Saved: pitchbend_correctness_test.png")

    # ── Rounding verification: k=0..52 should cycle back to near original ──
    print(f"\n  Rounding cycle check (k=0 to k=52):")
    if test_orig_files:
        sample_orig = test_orig_files[0]
        sample_stem = Path(sample_orig).stem
        orig_notes_check, _ = parse_notes_from_midi(sample_orig)
        if orig_notes_check:
            orig_canonical_check = [n['canonical_pitch'] for n in orig_notes_check]
            # Check k=0 identity
            k0_path = os.path.join(aug_dir, f"{sample_stem}_rot53_00.mid")
            if os.path.exists(k0_path):
                k0_notes, _ = parse_notes_from_midi(k0_path)
                k0_canonical = [n['canonical_pitch'] for n in k0_notes]
                if len(orig_canonical_check) == len(k0_canonical):
                    max_err = max(abs(a - b) for a, b in zip(orig_canonical_check, k0_canonical))
                    print(f"    k=0 (identity) max error: {max_err*100:.4f} cents — {'✓ PASS' if max_err*100 < 0.5 else '✗ FAIL'}")

    # Save test results as JSON
    with open(os.path.join(stats_dir, 'phase4_results.json'), 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"  Saved: phase4_results.json")

    print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Post-Augmentation Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def run_poststats(input_dir, max_files=None, progress_every=100):
    """Phase 5: Compute statistics on the augmented set and compare.

    Accepts either:
      - dataset root containing original .mid files and an ./aug subdir, or
      - the aug directory itself.
    """
    input_dir = os.path.abspath(input_dir)
    if os.path.basename(input_dir.rstrip(os.sep)) == 'aug':
        aug_dir = input_dir
        orig_dir = os.path.dirname(input_dir)
    else:
        orig_dir = input_dir
        aug_dir = os.path.join(input_dir, 'aug')

    stats_dir = os.path.join(aug_dir, 'stats')
    ensure_dir(stats_dir)

    orig_files = collect_midi_files(orig_dir)
    aug_files = sorted(glob.glob(os.path.join(aug_dir, '*_rot53_*.mid')))

    if not aug_files:
        logger.error(f"No augmented files found in {aug_dir}")
        return

    if max_files is not None and max_files > 0:
        orig_files = orig_files[:max_files]
        aug_files = aug_files[:max_files]

    if not orig_files:
        logger.warning(f"No original .mid files found in {orig_dir}; original-side metrics will be empty.")

    print(f"\n{'='*70}")
    print(f"PHASE 5: Post-Augmentation Statistics")
    print(f"{'='*70}\n")

    print(f"  Collecting original statistics from {len(orig_files)} files...")
    orig_stats = _accumulate_pitch_stats(orig_files, label='orig', progress_every=progress_every)

    print(f"  Collecting augmented statistics from {len(aug_files)} files...")
    aug_stats = _accumulate_pitch_stats(aug_files, label='aug', progress_every=progress_every)

    orig_unique_bins = orig_stats['unique_bins']
    aug_unique_bins = aug_stats['unique_bins']

    print(f"  {'Metric':<35} {'Original':>15} {'Augmented':>15}")
    print(f"  {'─'*65}")
    print(f"  {'Total files':<35} {orig_stats['processed_files']:>15} {aug_stats['processed_files']:>15}")
    print(f"  {'Total notes':<35} {orig_stats['total_notes']:>15} {aug_stats['total_notes']:>15}")
    print(f"  {'Unique cent bins (mod 1200)':<35} {orig_unique_bins:>15} {aug_unique_bins:>15}")
    if orig_stats['min_pitch'] is not None:
        print(f"  {'Pitch range (cents)':<35} {orig_stats['min_pitch']*100:>12.1f}–{orig_stats['max_pitch']*100:.1f}")
    if aug_stats['min_pitch'] is not None:
        print(f"  {'Pitch range augmented (cents)':<35} {aug_stats['min_pitch']*100:>12.1f}–{aug_stats['max_pitch']*100:.1f}")

    if orig_stats['failed_files'] or aug_stats['failed_files']:
        error_log_path = os.path.join(stats_dir, 'poststats_errors.log')
        with open(error_log_path, 'w') as f:
            for entry in orig_stats['errors']:
                f.write(f"orig\t{entry}\n")
            for entry in aug_stats['errors']:
                f.write(f"aug\t{entry}\n")
        print(f"  File parse errors: {orig_stats['failed_files']} original, {aug_stats['failed_files']} augmented")
        print(f"  Error log: {error_log_path}")

    # ── Side-by-side histograms ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    pc_x = np.arange(1200)
    axes[0].bar(pc_x, orig_stats['pc_hist'], width=1.0, edgecolor='none', alpha=0.8, color='steelblue')
    axes[0].set_xlabel('Pitch Class (cents, 0–1199)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f"Original — {orig_stats['processed_files']} files, {orig_stats['total_notes']} notes, {orig_unique_bins} unique bins")

    axes[1].bar(pc_x, aug_stats['pc_hist'], width=1.0, edgecolor='none', alpha=0.8, color='coral')
    axes[1].set_xlabel('Pitch Class (cents, 0–1199)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f"Augmented — {aug_stats['processed_files']} files, {aug_stats['total_notes']} notes, {aug_unique_bins} unique bins")

    plt.tight_layout()
    fig.savefig(os.path.join(stats_dir, 'pitch_distribution_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"\n  Saved: pitch_distribution_comparison.png")

    # Augmented full pitch histogram
    full_x = np.arange(1280) * 10
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(full_x, aug_stats['full_hist'], width=10.0, edgecolor='none', alpha=0.8, color='coral', align='edge')
    ax.set_xlabel('Canonical Pitch (cents)')
    ax.set_ylabel('Count')
    ax.set_title(f"Canonical Pitch Distribution — Augmented ({aug_stats['processed_files']} files)")
    plt.tight_layout()
    fig.savefig(os.path.join(stats_dir, 'pitch_distribution_augmented.png'), dpi=150)
    plt.close(fig)
    print(f"  Saved: pitch_distribution_augmented.png")

    # ── Bin-filling analysis ──
    print(f"\n  Bin-filling improvement: {orig_unique_bins} → {aug_unique_bins} unique cent bins")
    if orig_unique_bins > 0:
        improvement = (aug_unique_bins - orig_unique_bins) / orig_unique_bins * 100
        print(f"  Improvement: +{improvement:.1f}%")
        print(f"  Coverage: {aug_unique_bins}/1200 = {aug_unique_bins/1200*100:.1f}%")

    # Files per transposition k
    from collections import Counter
    k_counts = Counter()
    for f in aug_files:
        # extract k from filename: ..._rot53_KK.mid
        base = os.path.basename(f)
        try:
            k_str = base.rsplit('_rot53_', 1)[1].replace('.mid', '')
            k_counts[int(k_str)] += 1
        except (IndexError, ValueError):
            pass

    print(f"\n  Files per transposition k:")
    for k in sorted(k_counts.keys()):
        bar = '█' * k_counts[k]
        print(f"    k={k:2d}: {k_counts[k]:3d} {bar}")

    # Save phase 5 stats
    phase5_stats = {
        'original_files': orig_stats['processed_files'],
        'augmented_files': aug_stats['processed_files'],
        'original_notes': orig_stats['total_notes'],
        'augmented_notes': aug_stats['total_notes'],
        'original_unique_bins': orig_unique_bins,
        'augmented_unique_bins': aug_unique_bins,
        'files_per_k': dict(k_counts),
        'failed_original_files': orig_stats['failed_files'],
        'failed_augmented_files': aug_stats['failed_files'],
    }
    with open(os.path.join(stats_dir, 'phase5_stats.json'), 'w') as f:
        json.dump(phase5_stats, f, indent=2)
    print(f"\n  Saved: phase5_stats.json")

    print(f"\n{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='53-TET Transposition Augmentation for SymbTr MIDI files.')
    parser.add_argument('command', choices=['stats', 'augment', 'verify', 'poststats', 'all'],
                        help='Phase to run: stats (Phase 1), augment (Phase 2), '
                             'verify (Phase 3+4), poststats (Phase 5), all (1→5).')
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory containing .mid files.')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Optional cap on number of files to process (useful for quick poststats dry-runs).')
    parser.add_argument('--progress-every', type=int, default=100,
                        help='Print poststats progress every N files (default: 100).')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if args.command == 'stats':
        run_stats(input_dir)
    elif args.command == 'augment':
        run_augment(input_dir)
    elif args.command == 'verify':
        run_verify(input_dir)
    elif args.command == 'poststats':
        run_poststats(input_dir, max_files=args.max_files, progress_every=args.progress_every)
    elif args.command == 'all':
        run_stats(input_dir)
        run_augment(input_dir)
        run_verify(input_dir)
        run_poststats(input_dir, max_files=args.max_files, progress_every=args.progress_every)


if __name__ == '__main__':
    main()
