# Import libraries
from mouse_imaging import *
import numpy as np
import pandas as pd
import itertools
import argparse
import pickle
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_preferred_world(cell_var_row, all_worlds, peak_conditions_col='peak_conditions'):
    """
    Return the single preferred world label for one cell.

    Priority:
      1. peak_conditions column (from add_significant_spatial_peaks):
         use the condition with the highest mean_activity_{cond} value.
         Falls back to first listed peak condition if those columns are absent.
      2. left_selective / right_selective columns.
      3. None — cell will be skipped.
    """
    peak_conds = cell_var_row.get(peak_conditions_col, pd.Series([None])).iloc[0]

    if peak_conds is not None and not (isinstance(peak_conds, float) and np.isnan(peak_conds)):
        if isinstance(peak_conds, str):
            peak_conds = [peak_conds]
        valid = [c for c in peak_conds if c in all_worlds]
        if len(valid) == 1:
            return valid[0]
        if len(valid) > 1:
            best, best_val = valid[0], -np.inf
            for cond in valid:
                col = f'mean_activity_{cond}'
                if col in cell_var_row.columns:
                    val = float(cell_var_row[col].iloc[0])
                    if np.isfinite(val) and val > best_val:
                        best_val = val
                        best = cond
            return best

    left_sel  = bool(cell_var_row.get('left_selective',  pd.Series([False])).iloc[0])
    right_sel = bool(cell_var_row.get('right_selective', pd.Series([False])).iloc[0])
    if left_sel and not right_sel and 'black_left' in all_worlds:
        return 'black_left'
    if right_sel and not left_sel and 'white_right' in all_worlds:
        return 'white_right'

    return None


def _build_tuning_curves(mean_act_world, trial_ids, ci):
    """
    Build a dict {real_trial_id: tuning_curve_array} for one cell in one world.

    mean_act_world : subset of mean_act for one world, indexed by real trial IDs
    trial_ids      : sorted list of real trial IDs to include
    ci             : integer cell column index

    Each tuning curve is indexed by y_bin. Bins not visited on a trial are NaN.
    All curves share the same bin index — no cumcount re-labelling.
    """
    all_bins = sorted(mean_act_world['y_bins'].unique())
    bin_to_pos = {b: i for i, b in enumerate(all_bins)}
    n_bins = len(all_bins)

    curves = {}
    for tid in trial_ids:
        trial_rows = mean_act_world[mean_act_world['trial'] == tid]
        curve = np.full(n_bins, np.nan)
        for _, row in trial_rows.iterrows():
            pos = bin_to_pos[row['y_bins']]
            curve[pos] = row[ci]
        curves[tid] = curve

    return curves


def _reliability_from_curves(curves, trial_ids):
    """
    Compute mean pairwise Pearson r across all trial pairs.

    Correlations are computed only over bins both trials visited (both finite).
    Returns (mean_r, corr_matrix) where corr_matrix is indexed by position
    in trial_ids.
    """
    n = len(trial_ids)
    corr_mat = np.full((n, n), np.nan)

    tid_to_idx = {tid: i for i, tid in enumerate(trial_ids)}

    for t1, t2 in itertools.combinations(trial_ids, 2):
        c1 = curves[t1]
        c2 = curves[t2]
        valid = np.isfinite(c1) & np.isfinite(c2)
        if valid.sum() < 2:
            continue
        if np.std(c1[valid]) == 0 or np.std(c2[valid]) == 0:
            continue
        r = pearsonr(c1[valid], c2[valid]).statistic
        i1, i2 = tid_to_idx[t1], tid_to_idx[t2]
        corr_mat[i1, i2] = r
        corr_mat[i2, i1] = r

    mean_r = np.nanmean(corr_mat)
    return mean_r, corr_mat


# ---------------------------------------------------------------------------
# main reliability function
# ---------------------------------------------------------------------------

def compute_mean_reliability_all_cells(
    adata,
    layer='dF_sig',
    sig_peak_col='significant_peak',
    selective_only=False,
    left_sel_col='left_selective',
    right_sel_col='right_selective',
    peak_conditions_col='peak_conditions',
    preferred_condition='peak_only',
    y_bin_size=5,
):
    """
    Compute trial-by-trial reliability for each cell, restricted to its
    preferred trial type (world).

    Returns
    -------
    per_cell_reliability : list, length n_vars; NaN for excluded cells
    corr_matrix_dict     : dict {cell_idx: (n_trials, n_trials) array or None}
                           matrix is for the preferred world only
    """
    y_bins = np.arange(0, 300 + y_bin_size, y_bin_size)

    task_mask = (
        (adata.obs['y'] > 0) & (adata.obs['y'] < 300) &
        (~adata.obs['inITI'].astype(bool)) &
        (adata.obs['correct'].astype(bool))
    )
    adata_filt = adata[task_mask].copy()

    var = adata.var.copy()
    if sig_peak_col not in var.columns:
        raise ValueError(
            f"'{sig_peak_col}' not found in adata.var. "
            "Run add_significant_spatial_peaks first."
        )
    sig_mask = var[sig_peak_col].astype(bool)

    if selective_only:
        sel = pd.Series(False, index=var.index)
        for col in [left_sel_col, right_sel_col]:
            if col in var.columns:
                sel |= var[col].astype(bool)
            else:
                raise ValueError(
                    f"selective_only=True but '{col}' not found in adata.var."
                )
        sig_mask = sig_mask & sel

    all_worlds = list(adata_filt.obs['world'].unique())

    X = adata_filt.layers[layer]
    if hasattr(X, 'toarray'): X = X.toarray()
    X = np.asarray(X, dtype=float)

    act_df = pd.DataFrame(X)
    act_df['y_bins'] = np.digitize(adata_filt.obs['y'].values, y_bins)
    act_df['trial']  = adata_filt.obs['trial'].values
    act_df['world']  = adata_filt.obs['world'].values

    cell_cols = list(range(adata.n_vars))

    mean_act = (
        act_df
        .groupby(['world', 'trial', 'y_bins'], observed=True)[cell_cols]
        .mean()
        .reset_index()
    )

    per_cell_reliability = [np.nan] * adata.n_vars
    corr_matrix_dict     = {ci: None for ci in range(adata.n_vars)}

    for ci in range(adata.n_vars):
        if not sig_mask.iloc[ci]:
            continue

        cell_var_row = var.iloc[[ci]]

        if preferred_condition == 'peak_only':
            pref_world = _get_preferred_world(
                cell_var_row, all_worlds, peak_conditions_col
            )
            if pref_world is None:
                continue
            worlds_to_use = [pref_world]
        else:
            worlds_to_use = all_worlds

        world_mean_rs = []

        for world in worlds_to_use:
            w_data    = mean_act[mean_act['world'] == world]
            trial_ids = sorted(w_data['trial'].unique())

            if len(trial_ids) < 2:
                continue

            curves = _build_tuning_curves(w_data, trial_ids, ci)
            mean_r, corr_mat = _reliability_from_curves(curves, trial_ids)

            if np.isfinite(mean_r):
                world_mean_rs.append(mean_r)

            if preferred_condition == 'peak_only':
                corr_matrix_dict[ci] = corr_mat

        if len(world_mean_rs) == 0:
            continue

        per_cell_reliability[ci] = float(np.mean(world_mean_rs))

    # store under layer-specific column name
    var_col = 'reliability_dF_sig' if layer == 'dF_sig' else 'reliability_dcnv'
    adata.var[var_col] = per_cell_reliability

    return per_cell_reliability, corr_matrix_dict


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(mouse, date, session, ops, layer, selective_only, preferred_condition, y_bin_size):
    adata = sess.load_imaging_sessions(
        mouse, dates=[{'date': date, 'session': session}], ops=ops
    )[0]

    var_col = 'reliability_dF_sig' if layer == 'dF_sig' else 'reliability_dcnv'
    print(f"Computing reliability ({layer}) for {mouse}, {date}, session {session}")
    print(f"  {adata.n_obs} frames x {adata.n_vars} cells")
    print(f"  selective_only={selective_only}, preferred_condition={preferred_condition}, y_bin_size={y_bin_size}")

    reliabilities, corr_matrices = compute_mean_reliability_all_cells(
        adata,
        layer=layer,
        selective_only=selective_only,
        preferred_condition=preferred_condition,
        y_bin_size=y_bin_size,
    )

    n_computed = sum(np.isfinite(r) for r in reliabilities)
    print(f"  Cells with reliability computed: {n_computed} / {adata.n_vars}")

    print(f"  Saving adata...")
    sess.save_adata(adata, adata_filekey='adata_h5ad')

    out_path = (
        f"/n/data2/hms/neurobio/harvey/yasmine/data/imaging/"
        f"{mouse}/{date}/{session}/reliability_{layer}.pkl"
    )
    with open(out_path, 'wb') as f:
        pickle.dump({"reliabilities": reliabilities, "correlations": corr_matrices}, f)
    print(f"  Saved pkl to {out_path}")


if __name__ == '__main__':
    import options

    parser = argparse.ArgumentParser(
        description="Compute trial-by-trial reliability and save to adata.var"
    )
    parser.add_argument('--mouse',                required=True,                type=str)
    parser.add_argument('--date',                 required=True,                type=str)
    parser.add_argument('--session',              required=True,                type=str)
    parser.add_argument('--ops',                  required=True,                type=str)
    parser.add_argument('--layer',                default='dF_sig',             type=str,
                        help='Layer to use: dF_sig or dcnv_norm')
    parser.add_argument('--selective_only',       default=False,
                        type=lambda x: x.lower() == 'true',
                        help='Only analyse left/right selective cells (default: False)')
    parser.add_argument('--preferred_condition',  default='peak_only',          type=str,
                        help='peak_only or all (default: peak_only)')
    parser.add_argument('--y_bin_size',           default=5,                    type=int,
                        help='Spatial bin size in cm (default: 5)')

    args = vars(parser.parse_args())
    args['ops'] = getattr(options, args['ops'])()
    main(**args)
