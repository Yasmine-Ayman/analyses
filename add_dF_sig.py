# Import libraries
from mouse_imaging import *
import numpy as np
import argparse


def detect_significant_transients(dF):
    """
    Detect significant transients in dF traces (frames x cells) using a statistical thresholding method.

    Parameters:
    - dF: numpy array of shape (n_frames, n_cells), containing dF/F traces.

    Returns:
    - significant_traces: numpy array of same shape as dF with non-significant frames set to zero.
    """
    dF = dF.copy()
    n_frames, n_cells = dF.shape

    # 1. Standardize each column (z-score using median and std)
    median = np.median(dF, axis=0)
    std = np.std(dF, axis=0)
    dF_z = (dF - median) / std

    # 2. Define thresholds
    thresholds = np.arange(1, 4.2, 0.2)  # 1.0 to 4.0 in 0.2 increments

    # 3. Initialize boolean mask for significant frames
    significant_mask = np.zeros_like(dF, dtype=bool)

    for t in thresholds:
        above_t = dF_z > t
        below_t = dF_z < -t

        for cell in range(n_cells):
            pos = above_t[:, cell]
            neg = below_t[:, cell]

            # Get lengths of all positive transients
            pos_diff = np.diff(np.concatenate([[False], pos, [False]]).astype(int))
            pos_starts = np.where(pos_diff == 1)[0]
            pos_ends = np.where(pos_diff == -1)[0]
            pos_lengths = pos_ends - pos_starts

            # Same for negative transients
            neg_diff = np.diff(np.concatenate([[False], neg, [False]]).astype(int))
            neg_starts = np.where(neg_diff == 1)[0]
            neg_ends = np.where(neg_diff == -1)[0]
            neg_lengths = neg_ends - neg_starts

            for i, length in enumerate(pos_lengths):
                false_positives = np.sum(neg_lengths >= length)
                true_positives = np.sum(pos_lengths >= length)
                fpr = false_positives / max(true_positives, 1)

                if fpr < 0.001:
                    significant_mask[pos_starts[i]:pos_ends[i], cell] = True

    # Union across thresholds
    merged_mask = significant_mask.copy()

    # 4. Merge transients < 2 frames apart and remove < 2 frame transients
    for cell in range(n_cells):
        mask = merged_mask[:, cell]
        diff = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Merge transients < 2 frames apart
        merged_starts = []
        merged_ends = []
        i = 0
        while i < len(starts):
            start = starts[i]
            end = ends[i]
            while i + 1 < len(starts) and starts[i + 1] - end < 2:
                i += 1
                end = ends[i]
            if end - start >= 2:
                merged_starts.append(start)
                merged_ends.append(end)
            i += 1

        mask[:] = False
        for s, e in zip(merged_starts, merged_ends):
            mask[s:e] = True

        merged_mask[:, cell] = mask

    # 5. Final output: zero-out non-significant parts in original dF
    significant_traces = np.where(merged_mask, dF, 0)
    significant_traces = np.maximum(significant_traces, 0)
    return significant_traces


def main(mouse, date, session, ops):
    adata = sess.load_imaging_sessions(mouse, dates=[{'date': date, 'session': session}], ops=ops)[0]
    print(f"Computing significant dF for {mouse}, {date}, session {session}")
    print(f"  {adata.n_obs} frames x {adata.n_vars} cells")

    dF = adata.layers['dF']
    adata.layers['dF_sig'] = detect_significant_transients(dF)

    print(f"  Saving adata...")
    sess.save_adata(adata, adata_filekey='adata_h5ad')
    print(f"  Done.")


if __name__ == '__main__':
    import options

    parser = argparse.ArgumentParser(description="Compute significant dF transients and add as adata layer")
    parser.add_argument('--mouse', required=True, type=str, help='Mouse ID (e.g., YRA086)')
    parser.add_argument('--date', required=True, type=str, help='Date string (e.g., 260309)')
    parser.add_argument('--session', required=True, type=str, help='Session (e.g., session_1)')
    parser.add_argument('--ops', required=True, type=str, help='Options function name (e.g., jRGECO_ops_newS2P)')

    args = vars(parser.parse_args())
    args['ops'] = getattr(options, args['ops'])()
    main(**args)
