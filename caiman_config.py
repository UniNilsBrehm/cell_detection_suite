CONFIG = {
    'data':
        {
            'fr': 3,  # Frame rate (frames per second)
            'decay_time': 2  # Decay time of calcium indicator
        },
    'init':
        {
            'K': 20,  # Expected number of neurons per patch
            'gSig': [5, 5],  # Expected half-size of neurons in pixels (height, width)
            'method_init': 'greedy_roi',  # Initialization method: 'greedy_roi' or 'corr_pnr'
            'min_corr': 0.7,  # Minimum local correlation for seed
            'min_pnr': 10,  # Minimum peak-to-noise ratio for seed
            'ssub': 2,   # Spatial subsampling (every 2nd pixel)
            'tsub': 2,  # Temporal subsampling (average every 2 frames)
            'nb': 2,  # Global background order
            'normalize_init': True  # Z-score normalization (do not use with CNMF-E Ring Model)
        },
    'online':
        {
            'ring_CNN': False},  # Use CNMF-E Ring Background Model (False = global low-rank background)
    'patch':  # Patching is not supported so far, so rf and stride have to be "null"!
        {
            'n_processes': None,  # Number of parallel processes (null = auto)
            'rf': None,  # Half size of each patch (≥ 2× gSig)
            'stride': None},  # Overlap between patches (typically rf/2)
    'merging':
        {
            'merge_thr': 0.8  # Maximum correlation for merging components
        },
    'temporal':
        {
            'p': 1   # Order of the autoregressive model
        },
    'quality':
        {
            'SNR_lowest': 0.5,  # Minimum trace SNR to be considered
            'min_SNR': 2.0,  # Peak SNR for accepting a component
            'rval_lowest': 0.2,  # Minimum spatial correlation for consideration
            'rval_thr': 0.6,  # Spatial correlation threshold for acceptance
            'use_cnn': True,  # Use CNN-based component classification
            'min_cnn_thr': 0.8,  # Minimum CNN score for accepting a component
            'cnn_lowest': 0.1  # Components below this score are ignored
        }
}
