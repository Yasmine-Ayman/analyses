# Import libraries
from mouse_imaging import *
import numpy as np
import argparse
import json


def add_significant_spatial_peaks_h(
    adata,
    n_shuffles=1000,
    present_percentile=95,
    highconf_percentile=99,
    absent_percentile=95,
    min_consecutive_bins=3,
    layer='dF_sig',
    conditions=None,
    y_col='y',
    h_col='h',
    trial_col='trial',
    iti_col='inITI',
    correct_col='correct',
    world_col='world',
    track_min=0.0,
    track_max=300.0,
    num_bins=60,
    random_seed=42,
    stem_middle_min=30.0,
    stem_middle_max=270.0,
):
    """
    Identify significant spatial peaks by circularly shifting behavior relative
    to neural activity, following Pettit et al.

    This is the '_h' variant: it excludes trials in which the heading |h| exceeds
    pi/2 anywhere in the middle 80% of the stem (y in [30, 270]) before
    computing peaks. The absolute value is used so leftward (negative h) and
    rightward (positive h) over-rotations are caught symmetrically.
    Results are written to '_h'-suffixed columns in adata.var, leaving the
    non-'_h' columns (if present) untouched.

    A cell is marked as having a peak in a condition if:
      actual bin activity strictly > shuffle bin activity in > present_percentile%
      of *valid* shuffles (shuffles where that bin was occupied) for at least
      min_consecutive_bins consecutive bins.

    Parameters
    ----------
    conditions : dict or None
        E.g. {'white_left': 'white_left', 'black_right': 'black_right'}.
        If None, pools all correct non-ITI task data into one condition 'all'.
    layer : str
        Default 'dF_sig'. Change to 'dcnv_norm' if using deconvolved signal.

    Adds to adata.var (all h5ad-serializable)
    -----------------
    significant_peak_h             : bool   — has any sig peak in any condition
    significant_peak_bins_h        : str    — JSON list of bin indices (present threshold)
    peak_conditions_h              : str    — JSON list of conditions that passed
    significant_peak_hiconf_h      : bool   — has peak at high-confidence threshold
    significant_peak_bins_hiconf_h : str    — JSON list of bin indices (hiconf threshold)
    absent_peak_bins_95_h          : str    — JSON list of bins below absent threshold
    """

    # ----------------------------
    # initialise columns on the ORIGINAL adata
    # (cells from excluded trials will simply keep these defaults)
    # ----------------------------
    adata.var['significant_peak_h']             = False
    adata.var['significant_peak_bins_h']        = '[]'
    adata.var['peak_conditions_h']              = '[]'
    adata.var['significant_peak_hiconf_h']      = False
    adata.var['significant_peak_bins_hiconf_h'] = '[]'
    adata.var['absent_peak_bins_95_h']          = '[]'

    # ----------------------------
    # exclude trials where h > pi/2 in middle 80% of stem
    # stem = 300 units, so middle 80% = y in [30, 270]
    # (mirrors add_modulation_selectivity_to_var_dfsig_h)
    # ----------------------------
    if all(col in adata.obs.columns for col in [trial_col, y_col, h_col]):
        middle_stem_mask = (
            (adata.obs[y_col] >= stem_middle_min) &
            (adata.obs[y_col] <= stem_middle_max)
        )
        bad_heading_mask = middle_stem_mask & (np.abs(adata.obs[h_col]) > np.pi / 2)
        if bad_heading_mask.any():
            bad_trials = adata.obs.loc[bad_heading_mask, trial_col].unique()
            adata_use = adata[~adata.obs[trial_col].isin(bad_trials)].copy()
            print(
                f"  Excluded {len(bad_trials)} trials with |h| > pi/2 in middle 80% of stem "
                f"({adata_use.n_obs} / {adata.n_obs} frames remain)"
            )
        else:
            adata_use = adata.copy()
            print("  No trials with |h| > pi/2 in middle 80% of stem — using all trials")
    else:
        adata_use = adata.copy()
        print(
            f"  WARNING: one of {trial_col}/{y_col}/{h_col} missing from adata.obs; "
            f"no heading-based exclusion applied"
        )

    if conditions is None:
        conditions = {'all': None}

    # ----------------------------
    # activity matrix  (frames x cells)
    # ----------------------------
    activity = adata_use.layers[layer]
    if hasattr(activity, 'toarray'):
        activity = activity.toarray()
    else:
        activity = np.asarray(activity, dtype=float)

    if activity.shape[0] != adata_use.n_obs and activity.shape[1] == adata_use.n_obs:
        activity = activity.T

    # ----------------------------
    # base task mask
    # ----------------------------
    task_mask = (
        (~adata_use.obs[iti_col].astype(bool).values) &
        (adata_use.obs[correct_col].astype(bool).values) &
        (adata_use.obs[y_col].values > track_min) &
        (adata_use.obs[y_col].values < track_max)
    )

    bin_edges = np.linspace(track_min, track_max, num_bins + 1)
    rng = np.random.default_rng(random_seed)

    present_thresh = int(n_shuffles * present_percentile / 100)
    hiconf_thresh  = int(n_shuffles * highconf_percentile / 100)
    absent_thresh  = int(np.floor(n_shuffles * absent_percentile / 100))

    # ----------------------------
    # helper: find runs of >= min_consecutive_bins
    # ----------------------------
    def consecutive_runs(indices, min_len):
        if len(indices) == 0:
            return []
        runs = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        return [r for r in runs if len(r) >= min_len]

    # ----------------------------
    # vectorised binned-mean helper
    # ----------------------------
    def fast_binned_means(neural, bin_ids, n_bins):
        n_cells = neural.shape[1]
        counts  = np.bincount(bin_ids, minlength=n_bins)
        sums    = np.zeros((n_bins, n_cells), dtype=float)
        np.add.at(sums, bin_ids, neural)
        out     = np.full((n_bins, n_cells), np.nan, dtype=float)
        valid   = counts > 0
        out[valid] = sums[valid] / counts[valid, None]
        return out

    # ----------------------------
    # process each condition
    # ----------------------------
    for cond_name, world_label in conditions.items():

        if world_label is None:
            cond_mask = task_mask.copy()
        else:
            cond_mask = task_mask & (adata_use.obs[world_col].values == world_label)

        if cond_mask.sum() < 20:
            continue

        y        = adata_use.obs.loc[cond_mask, y_col].to_numpy()
        neural   = activity[cond_mask, :]
        n_frames = len(y)

        if n_frames < 20:
            continue

        bin_ids      = np.clip(np.digitize(y, bin_edges) - 1, 0, num_bins - 1)
        actual_means = fast_binned_means(neural, bin_ids, num_bins)

        # null distribution (n_shuffles, n_bins, n_cells)
        null = np.full((n_shuffles, num_bins, neural.shape[1]), np.nan, dtype=float)
        for i in range(n_shuffles):
            shift     = rng.integers(1, n_frames)
            s_bin_ids = np.clip(
                np.digitize(np.roll(y, shift), bin_edges) - 1,
                0, num_bins - 1
            )
            null[i] = fast_binned_means(neural, s_bin_ids, num_bins)

        # ----------------------------
        # per-cell evaluation
        # ----------------------------
        for cell in range(neural.shape[1]):
            cell_name = adata_use.var.index[cell]
            actual    = actual_means[:, cell]
            null_cell = null[:, :, cell]

            real_valid = np.isfinite(actual)
            if real_valid.sum() < min_consecutive_bins:
                continue

            null_valid    = np.isfinite(null_cell)
            exceed        = (actual[None, :] > null_cell) & null_valid
            exceed_counts = exceed.sum(axis=0)

            is_present = (exceed_counts > present_thresh) & real_valid
            is_hiconf  = (exceed_counts > hiconf_thresh)  & real_valid
            is_absent  = (exceed_counts < absent_thresh)  & real_valid

            present_idx = np.where(is_present)[0]
            hiconf_idx  = np.where(is_hiconf)[0]

            present_runs = consecutive_runs(present_idx, min_consecutive_bins)
            hiconf_runs  = consecutive_runs(hiconf_idx,  min_consecutive_bins)

            if present_runs:
                all_present_bins = np.concatenate(present_runs).tolist()
                adata.var.at[cell_name, 'significant_peak_h']      = True
                adata.var.at[cell_name, 'significant_peak_bins_h'] = json.dumps(all_present_bins)
                existing = json.loads(adata.var.at[cell_name, 'peak_conditions_h'])
                existing.append(cond_name)
                adata.var.at[cell_name, 'peak_conditions_h'] = json.dumps(existing)

            if hiconf_runs:
                all_hiconf_bins = np.concatenate(hiconf_runs).tolist()
                adata.var.at[cell_name, 'significant_peak_hiconf_h']      = True
                adata.var.at[cell_name, 'significant_peak_bins_hiconf_h'] = json.dumps(all_hiconf_bins)

            adata.var.at[cell_name, 'absent_peak_bins_95_h'] = json.dumps(
                np.where(is_absent)[0].tolist()
            )


def main(mouse, date, session, ops, layer, n_shuffles):
    adata = sess.load_imaging_sessions(mouse, dates=[{'date': date, 'session': session}], ops=ops)[0]
    print(f"Computing significant spatial peaks (_h, heading-filtered) for {mouse}, {date}, session {session}")
    print(f"  {adata.n_obs} frames x {adata.n_vars} cells")
    print(f"  layer={layer}, n_shuffles={n_shuffles}")

    add_significant_spatial_peaks_h(adata, layer=layer, n_shuffles=n_shuffles)

    n_sig = adata.var['significant_peak_h'].sum()
    print(f"  Significant peak cells (_h): {n_sig} / {adata.n_vars} ({100*n_sig/adata.n_vars:.1f}%)")

    print(f"  Saving adata...")
    sess.save_adata(adata, adata_filekey='adata_h5ad')
    print(f"  Done.")


if __name__ == '__main__':
    import options

    parser = argparse.ArgumentParser(
        description="Compute significant spatial peaks with heading-based trial exclusion "
                    "(|h| > pi/2 in middle 80% of stem) and save to adata.var (_h columns)"
    )
    parser.add_argument('--mouse',      required=True,  type=str, help='Mouse ID (e.g., YRA086)')
    parser.add_argument('--date',       required=True,  type=str, help='Date string (e.g., 260309)')
    parser.add_argument('--session',    required=True,  type=str, help='Session (e.g., session_1)')
    parser.add_argument('--ops',        required=True,  type=str, help='Options function name (e.g., jRGECO_ops_newS2P)')
    parser.add_argument('--layer',      default='dF_sig', type=str, help='Layer to use (default: dF_sig)')
    parser.add_argument('--n_shuffles', default=1000,   type=int, help='Number of shuffles (default: 1000)')

    args = vars(parser.parse_args())
    args['ops'] = getattr(options, args['ops'])()
    main(**args)
