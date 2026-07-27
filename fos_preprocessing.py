# FOS preprocessing script
# Runs process_fos_data_norm and add_modulation_selectivity_to_var on imaging data

from mouse_imaging import *
import numpy as np
import pandas as pd
import argparse
import os
from scipy.io import loadmat
from scipy.ndimage import median_filter
from scipy.stats import ttest_ind


def process_fos_data_norm(adata_allsources, adata, mouse_id, date, session_num=1, plane_num=0):
    """
    Process fos data for a given mouse, date, and session, with background subtraction.
    
    Parameters:
    adata_allsources (AnnData): Full dataset containing cell information.
    adata (AnnData): Subset of the data specific to this session.
    mouse_id (str): Mouse ID (e.g., 'YRA049')
    date (str): Date in format 'DDMMYY' (e.g., '250119')
    session_num (int): Session number (default: 1)
    plane_num (int): Plane number (default: 0)
    
    Returns:
    pd.DataFrame: DataFrame containing fos data with classifications
    """
    
    # Construct paths
    base_path = '/n/data2/hms/neurobio/harvey'
    fos_stack_directory = f'/n/scratch/users/y/yra642/Behavior_Imaging_Data/2P/{mouse_id}/{date}/session_{session_num}/stacks/stack2/'
    cell_mask_directory = f'{base_path}/yasmine/data/imaging/{mouse_id}/{date}/session_{session_num}/slice_1/plane{plane_num}'
    
    # Load fos data
    os.chdir(fos_stack_directory)
    fos_pre = loadmat('registered_fos_pre.mat')['registered_fos_pre_nonrigid']
    fos_post = loadmat('registered_fos_post.mat')['registered_fos_post_nonrigid']
    
    # Apply background subtraction using median filtering
    fos_pre_normalized = fos_pre / median_filter(fos_pre, size=(60, 60), mode='reflect')
    fos_post_normalized = fos_post / median_filter(fos_post, size=(60, 60), mode='reflect')
    
    # Load cell masks
    os.chdir(cell_mask_directory)
    stat = np.load('stat.npy', allow_pickle=True)
    
    # Create masks
    ca_masks = np.zeros((512, 512))
    for row in range(len(stat)):
        cell_num = row + 1
        xpix, ypix = stat[row]['xpix'], stat[row]['ypix']
        for x, y in zip(xpix, ypix):
            ca_masks[y, x] = cell_num
            
    # Get unique cell masks
    masks = np.unique(ca_masks)[1:]  # Exclude background
    
    # Calculate fluorescence
    fluor_pre, fluor_post = [], []
    for cell_index in masks:
        cell_indices = np.where(ca_masks == cell_index)
        pre_values = fos_pre_normalized[cell_indices]
        post_values = fos_post_normalized[cell_indices]
        
        fluor_pre.append(np.mean(pre_values))
        fluor_post.append(np.mean(post_values))
    
    # Create DataFrame
    df = pd.DataFrame({
        'fos_pre': fluor_pre,
        'fos_post': fluor_post,
        'fold_change': np.array(fluor_post) / np.array(fluor_pre)
    })
    
    # Make a copy of df
    df_copy = df.copy()
    
    # Set index to match adata_var
    indices = adata_allsources.var.index
    df_copy.index = indices
    
    # Align with adata.var
    new_df = pd.concat([adata.var, df_copy.loc[adata.var.index]], axis=1)
    
    # Add percentile-based classification
    high_threshold = new_df['fold_change'].quantile(0.8)
    low_threshold = new_df['fold_change'].quantile(0.2)

    new_df['fosHigh+'] = new_df['fold_change'] >= high_threshold
    new_df['fosLow+'] = new_df['fold_change'] <= low_threshold
    
    return new_df


def add_modulation_selectivity_to_var(adata, threshold=0.25, min_time=1.3, p_threshold=0.01):
    """
    Add task modulation and selectivity metrics to adata.var.
    """
    adata.var['task_modulated'] = False
    adata.var['right_selective'] = False
    adata.var['left_selective'] = False
    adata.var['selectivity'] = 0
    min_consecutive = int(min_time / adata.obs.dt.mean())
    activity = adata.layers['dF']
    adata_non_iti = adata[adata.obs['inITI'] == False]
    adata_left = adata_non_iti[adata_non_iti.obs['correct'] & (adata_non_iti.obs['world'] == 'black_left')]
    adata_right = adata_non_iti[adata_non_iti.obs['correct'] & (adata_non_iti.obs['world'] == 'white_right')]
    adata.obs['y>200'] = adata.obs['y'] > 200
    trig = analysis.trigger_inds(adata.obs, trigger_name='y>200', t_range=(-5, 5))
    try:
        threshold_values = threshold * activity[trig['idyx'], :].mean(axis=0).max(axis=0)
    except:
        threshold_values = threshold * activity[trig['idyx'][:-1, :], :].mean(axis=0).max(axis=0)
    
    for col, cell in enumerate(adata.var_names):
        periods = np.where(activity[:, col] > threshold_values[col])[0]
        consecutive_periods = np.split(periods, np.where(np.diff(periods) != 1)[0] + 1)
        prolonged_activity = [period for period in consecutive_periods if len(period) >= min_consecutive]
        if prolonged_activity:
            inactive_indices = np.setdiff1d(np.arange(activity.shape[0]), np.concatenate(prolonged_activity))
            for prolonged_activity_period in prolonged_activity:
                mean_active = activity[prolonged_activity_period, col].mean()
                mean_inactive = activity[inactive_indices, col].mean()
                if mean_active > 3 * mean_inactive:
                    adata.var.loc[cell, 'task_modulated'] = True
                    dF_left = adata_left.layers['dF'][:, col].mean()
                    dF_right = adata_right.layers['dF'][:, col].mean()
                    t_stat, p_value = ttest_ind(
                        adata_left.layers['dF'][:, col],
                        adata_right.layers['dF'][:, col],
                        equal_var=False)
                    if p_value < p_threshold:
                        selectivity_index = (dF_right - dF_left) / (dF_right + dF_left)
                        if selectivity_index > 0:
                            adata.var.loc[cell, 'right_selective'] = True
                        elif selectivity_index < 0:
                            adata.var.loc[cell, 'left_selective'] = True
                        adata.var.loc[cell, 'selectivity'] = abs(selectivity_index)
                    break


def main(mouse, date, session, ops):
    """Main function to run FOS preprocessing."""
    print(f"Processing FOS data for {mouse}, {date}, {session}")
    
    # Extract session number from session string (e.g., "session_1" -> 1)
    session_num = int(session.split('_')[1])
    
    # Load adata and adata_allsources
    adata = sess.load_imaging_sessions(
        mouse, 
        dates=[{'date': date, 'session': session}], 
        ops=ops
    )[0]
    
    adata_allsources = sess.load_imaging_sessions(
        mouse, 
        dates=[{'date': date, 'session': session}], 
        adata_filekey='adata_allsources_h5ad'
    )[0]
    
    print(f"Loaded adata with {adata.shape[1]} cells")
    
    # Check if registered_fos_pre.mat exists
    fos_stack_directory = f'/n/scratch/users/y/yra642/Behavior_Imaging_Data/2P/{mouse}/{date}/session_{session_num}/stacks/stack2/'
    if not os.path.exists(os.path.join(fos_stack_directory, 'registered_fos_pre.mat')):
        raise FileNotFoundError(f"No 'registered_fos_pre.mat' found at {fos_stack_directory}")
    
    # Process FOS data
    print("Running process_fos_data_norm...")
    df_fos = process_fos_data_norm(
        adata_allsources=adata_allsources,
        adata=adata,
        mouse_id=mouse,
        date=date,
        session_num=session_num
    )
    adata.var = df_fos
    
    # Add modulation and selectivity metrics
    print("Running add_modulation_selectivity_to_var...")
    add_modulation_selectivity_to_var(adata, threshold=0.25, min_time=1.3, p_threshold=0.01)
    
    # Print summary statistics
    print(f"Task-modulated cells: {np.sum(adata.var['task_modulated'])}")
    print(f"Left-selective cells: {np.sum(adata.var['left_selective'])}")
    print(f"Right-selective cells: {np.sum(adata.var['right_selective'])}")
    print(f"Mean selectivity index: {np.mean(adata.var['selectivity']):.4f}")
    print(f"FosHigh+ cells: {np.sum(adata.var['fosHigh+'])}")
    print(f"FosLow+ cells: {np.sum(adata.var['fosLow+'])}")
    
    # Save the updated adata
    print("Saving adata...")
    sess.save_adata(adata, adata_filekey='adata_h5ad')
    print(f"Successfully saved adata for {mouse} {date} {session}")


if __name__ == '__main__':
    import options

    parser = argparse.ArgumentParser(description="Run FOS preprocessing on imaging data")
    parser.add_argument('--mouse', required=True, type=str, help='Mouse ID (e.g., YRA084)')
    parser.add_argument('--date', required=True, type=str, help='Date string (e.g., 260329)')
    parser.add_argument('--session', required=True, type=str, help='Session (e.g., session_1)')
    parser.add_argument('--ops', required=True, type=str, help='Options function name (e.g., jRGECO_ops_newS2P)')

    args = vars(parser.parse_args())
    args['ops'] = getattr(options, args['ops'])()
    main(**args)
