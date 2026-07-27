from scipy.stats import ttest_ind
import numpy as np


def add_modulation_selectivity_to_var_dfsig_h(
    adata,
    threshold=0.25,
    min_time=1.3,
    p_threshold=0.01,
):
    """
    Task-modulation and left/right selectivity on dF_sig, excluding trials with
    over-rotated heading in the middle 80% of the stem.

    A trial is excluded if |h| > pi/2 at any frame with y in [30, 270].
    Using |h| (rather than h directly) catches both leftward (negative h) and
    rightward (positive h) over-rotations symmetrically.

    Writes:
        task_modulated_dfsig_h : bool
        right_selective_dfsig_h : bool
        left_selective_dfsig_h  : bool
        selectivity_dfsig_h     : float (absolute selectivity index)
    """
    adata.var['task_modulated_dfsig_h']  = False
    adata.var['right_selective_dfsig_h'] = False
    adata.var['left_selective_dfsig_h']  = False
    adata.var['selectivity_dfsig_h']     = 0.0

    # --------------------------------------------------
    # exclude trials where |h| > pi/2 in middle 80% of stem
    # stem = 300 units, so middle 80% = y in [30, 270]
    # --------------------------------------------------
    adata_use = adata
    if all(col in adata.obs.columns for col in ['trial', 'y', 'h']):
        middle_stem_mask = (adata.obs['y'] >= 30) & (adata.obs['y'] <= 270)
        bad_heading_mask = middle_stem_mask & (np.abs(adata.obs['h']) > np.pi / 2)
        if bad_heading_mask.any():
            bad_trials = adata.obs.loc[bad_heading_mask, 'trial'].unique()
            adata_use = adata[~adata.obs['trial'].isin(bad_trials)].copy()
            print(f"Excluded {len(bad_trials)} trials with |h| > pi/2 in middle 80% of stem")
        else:
            adata_use = adata.copy()
    else:
        adata_use = adata.copy()

    min_consecutive = int(min_time / adata_use.obs.dt.mean())
    activity = adata_use.layers['dF_sig']

    adata_non_iti = adata_use[adata_use.obs['inITI'] == False]
    adata_left = adata_non_iti[
        adata_non_iti.obs['correct'] &
        (adata_non_iti.obs['world'] == 'black_left')
    ]
    adata_right = adata_non_iti[
        adata_non_iti.obs['correct'] &
        (adata_non_iti.obs['world'] == 'white_right')
    ]

    adata_use.obs['y_gt_200_h'] = adata_use.obs['y'] > 200
    trig = analysis.trigger_inds(adata_use.obs, trigger_name='y_gt_200_h', t_range=(-5, 5))
    try:
        threshold_values = threshold * activity[trig['idyx'], :].mean(axis=0).max(axis=0)
    except Exception:
        threshold_values = threshold * activity[trig['idyx'][:-1, :], :].mean(axis=0).max(axis=0)

    for col, cell in enumerate(adata_use.var_names):
        periods = np.where(activity[:, col] > threshold_values[col])[0]
        consecutive_periods = np.split(periods, np.where(np.diff(periods) != 1)[0] + 1)
        prolonged_activity = [period for period in consecutive_periods if len(period) >= min_consecutive]

        if prolonged_activity:
            inactive_indices = np.setdiff1d(np.arange(activity.shape[0]), np.concatenate(prolonged_activity))
            for prolonged_activity_period in prolonged_activity:
                mean_active   = activity[prolonged_activity_period, col].mean()
                mean_inactive = activity[inactive_indices, col].mean()
                if mean_active > 3 * mean_inactive:
                    adata.var.loc[cell, 'task_modulated_dfsig_h'] = True
                    dF_left  = adata_left.layers['dF_sig'][:, col].mean()
                    dF_right = adata_right.layers['dF_sig'][:, col].mean()
                    t_stat, p_value = ttest_ind(
                        adata_left.layers['dF_sig'][:, col],
                        adata_right.layers['dF_sig'][:, col],
                        equal_var=False,
                    )
                    if p_value < p_threshold:
                        denom = dF_right + dF_left
                        if denom != 0:
                            selectivity_index = (dF_right - dF_left) / denom
                        else:
                            selectivity_index = 0.0
                        if selectivity_index > 0:
                            adata.var.loc[cell, 'right_selective_dfsig_h'] = True
                        elif selectivity_index < 0:
                            adata.var.loc[cell, 'left_selective_dfsig_h']  = True
                        adata.var.loc[cell, 'selectivity_dfsig_h'] = abs(selectivity_index)
                    break
