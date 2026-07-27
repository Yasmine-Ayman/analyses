# import things 
# Import libraries
from mouse_imaging import *
import numpy as np
import pandas as pd
import itertools
import argparse
import pickle
from scipy.stats import pearsonr


def compute_mean_reliability_all_cells(adata):
    # Fixed y-bin size to avoid unstable binning
    y_bins = np.arange(0, 305, 5)

    # Filter trials
    adata_filtered = adata[
        (adata.obs['y'] > 0) & 
        (adata.obs['y'] < 300) & 
        (~adata.obs['inITI']) & 
        (adata.obs['correct'])
    ].copy()

    # Set up activity dataframe
    act = pd.DataFrame(adata_filtered.layers['dcnv_norm'])
    act['y_bins'] = np.digitize(adata_filtered.obs['y'], y_bins)
    act['trial'] = adata_filtered.obs['trial'].values
    act['world'] = adata_filtered.obs['world'].values

    # Mean activity per world/trial/y_bin
    mean_act = act.groupby(['world', 'trial', 'y_bins'], observed=True).mean().reset_index().dropna(subset=[0])
    mean_act['trial'] = mean_act.groupby(['world', 'y_bins'], observed=True).cumcount()

    # Create empty correlation matrices
    corr_matrix_dict = {
        cell: {world: np.full((len(mean_act[mean_act.world == world].trial.unique()), 
                               len(mean_act[mean_act.world == world].trial.unique())), np.nan) 
               for world in mean_act.world.unique()} 
        for cell in range(adata.shape[1])
    }

    # Compute correlations
    for world in mean_act.world.unique():
        trial_ids = mean_act[mean_act.world == world].trial.unique()
        for trial_1, trial_2 in itertools.combinations(trial_ids, 2):
            for cell in range(adata.shape[1]):
                trial_1_data = mean_act.loc[(mean_act.trial == trial_1) & (mean_act.world == world), cell]
                trial_2_data = mean_act.loc[(mean_act.trial == trial_2) & (mean_act.world == world), cell]

                if (
                    len(trial_1_data) == len(trial_2_data) and
                    len(trial_1_data) > 1 and
                    not np.all(trial_1_data == trial_1_data.iloc[0]) and
                    not np.all(trial_2_data == trial_2_data.iloc[0])
                ):
                    r = pearsonr(trial_1_data, trial_2_data).statistic
                    corr_matrix_dict[cell][world][trial_1-1, trial_2-1] = r
                    corr_matrix_dict[cell][world][trial_2-1, trial_1-1] = r

    # Compute per-cell mean reliability
    per_cell_reliability = []
    for cell, cell_dict in corr_matrix_dict.items():
        reliabilities = [np.nanmean(matrix) for matrix in cell_dict.values()]
        per_cell_reliability.append(np.nanmean(reliabilities))

    # Store in adata
    adata.var['reliability'] = per_cell_reliability

    return per_cell_reliability, corr_matrix_dict



def main(mouse, date, session, ops):
    adata = sess.load_imaging_sessions(mouse, dates= [{'date':date, 'session':session}], ops=ops)[0]
    print(f"Computing reliability for {mouse}, {date}, session {session}")
    reliabilities, corr_matrices = compute_mean_reliability_all_cells(adata)
    
    sess.save_adata(adata, adata_filekey='adata_h5ad')
    out_path = f"/n/data2/hms/neurobio/harvey/yasmine/data/imaging/{mouse}/{date}/{session}/reliability.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump({"reliabilities": reliabilities, "correlations": corr_matrices}, f)

    print(f"Saved results to {out_path}")

if __name__ == '__main__':
    import argparse
    import options  # Make sure this exists and is imported

    parser = argparse.ArgumentParser(description="Compute trial-by-trial reliability for calcium imaging data")
    parser.add_argument('--mouse', required=True, type=str, help='Mouse ID (e.g., yra054)')
    parser.add_argument('--date', required=True, type=str, help='Date string (e.g., 250115)')
    parser.add_argument('--session', required=True, type=str)
    parser.add_argument('--ops', required=True, type=str)

    args = vars(parser.parse_args())  # Convert Namespace to dict
    args['ops'] = getattr(options, args['ops'])()  # Now dict syntax works
    main(**args)