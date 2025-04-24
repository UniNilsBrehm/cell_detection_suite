import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tifffile as tiff
import time
import yaml

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# CAIMAN IMPORTS
from caiman import load, load_memmap, stop_server, load_movie_chain, concatenate
from caiman.cluster import setup_cluster
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline
# from IPython import embed


def load_tiff_recording(file_name, flatten=False):
    all_frames = []

    with tiff.TiffFile(file_name) as tif:
        for i, series in enumerate(tif.series):
            # print(f"Series {i} shape: {series.shape}")
            data = series.asarray()
            all_frames.append(data)

    # Flatten the list if needed
    if flatten:
        frames = np.concatenate(all_frames, axis=0).view(np.uint16)
    else:
        frames = np.array(all_frames)

    return frames


def source_extraction(file_name, save_dir, params_dict=None, visual_check=True, parallel=True, corr_map=False):
    # start a cluster
    if parallel:
        c, dview, n_processes = setup_cluster(
            backend='multiprocessing', n_processes=None, single_thread=False
        )
    else:
        dview = None
        n_processes = 1

    # File Name
    f_names = [file_name]
    if len(f_names) <= 0:
        print('\n==== ERROR: No tiff file recording found =====\n')
        return

    cmap = 'gray'
    # cmap = 'hot'

    if params_dict is None:
        # SETTINGS
        params_dict = {
            'data': {
                'fnames': f_names,
                'fr': 3,
                'decay_time': 5
            },
            'init': {
                'K': 100,  # expected # of neurons per patch
                'gSig': [2, 2],  # expected half size of neurons in px
                # 'method_init': 'corr_pnr',   # correlation-based initialization, patching should be avoided here
                'method_init': 'greedy_roi',
                'min_corr': 0.8,  # min local correlation for seed
                'min_pnr': 10,  # min peak-to-noise ratio for seed
                'ssub': 2,  # spatial subsampling during initialization (use every 2nd pixel → half resolution)
                'tsub': 2,   # temporal subsampling during initialization (average every 2 frames)
                'nb': 2,  # global background order
                'normalize_init': True,  # z score data, do not use with CNMF-E Background Ring Model
            },
            'online': {
                'ring_CNN': False  # CNMF-E Background Ring Model, if False, use global low-rank background modeling
            },
            'patch': {
                'n_processes': None,
                'rf': None,  # half size of each patch (should be ≥ 2× gSig)
                'stride': None  # overlap between patches ( 1- stride/rf), (typically 50% of rf)
                # 'rf': 16,  # half size of each patch (should be ≥ 2× gSig)
                # 'stride': 8  # overlap between patches ( 1- stride/rf), (typically 50% of rf)
            },
            'merging': {
                'merge_thr': 0.8  # merging threshold, max correlation allowed
            },
            'temporal': {
                'p': 1,  # order of the autoregressive system
            },
            'quality': {
                'SNR_lowest': 1.0,  # minimum required trace SNR. Traces with SNR below this will get rejected
                'min_SNR': 2.0,  # peak SNR for accepted components (if above this, accept)
                'rval_lowest': 0.2,  # minimum required space correlation. Components with correlation below this will get rejected
                'rval_thr': 0.8,  # spatial footprint consistency: space correlation threshold (if above this, accept)
                'use_cnn': True,  # use the CNN classifier (prob. of component being a neuron)
                'min_cnn_thr': 0.8,   # Only components with CNN scores ≥ thr are accepted as likely real neurons.
                'cnn_lowest': 0.1  # Components scoring < lowest are considered garbage and won’t be touched even during manual curation or re-evaluation.
            },
        }
    else:
        params_dict['data']['fnames'] = f_names

    opts = params.CNMFParams(params_dict=params_dict)

    # 2. Run full CNMF pipeline
    print('\n==== RUN CNMF =====\n')
    cnm = cnmf.CNMF(n_processes=n_processes, params=opts, dview=dview)
    cnm = cnm.fit_file()

    # 3. Evaluate components
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier (this will pick up only neurons
    #           and filter out active processes)
    # A component has to exceed ALL low thresholds as well as ONE high threshold to be accepted.

    print('\n==== EVALUATING COMPONENTS =====\n')
    if cnm.estimates.A.shape[-1] <= 0:
        print('\n==== WARNING: NO COMPONENTS FOUND =====\n')
        exit()

    # load memory mapped file
    Yr, dims, T = load_memmap(cnm.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    mean_image = np.mean(images, axis=0)
    # sd_image = np.std(images, axis=0)

    print("Dims:", dims, "Frames:", T)
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    # 4. Visualize components
    plt.ioff()  # Turn off interactive mode
    if visual_check:
        if corr_map:
            print('\n==== VISUAL VALIDATION =====\n')
            # Plotting contours
            # Compute Pixel Correlation Matrix (px and its 8 neighbors)
            lc = local_correlations_movie_offline(
                f_names[0],
                remove_baseline=True,
                swap_dim=False,
                window=500,
                stride=8,
                winSize_baseline=200,
                quantil_min_baseline=10,
                dview=dview
            )
            Cn = lc.max(axis=0)

        # Get ROI centers
        component_centers = cnm.estimates.center
        good_idx = cnm.estimates.idx_components

        # Compare Accepted and Rejected Components
        save_contour_plot(cnm, mean_image, f'{save_dir}/caiman_mean_img_contour_plot.jpg', cmap=cmap)
        save_roi_centers_plot(component_centers[good_idx], mean_image, file_dir=f'{save_dir}/caiman_mean_img_rois_center.jpg', marker_size=30, cmap=cmap)

        # View components
        # cnm.estimates.view_components(images, idx=cnm.estimates.idx_components, img=Cn)

        if corr_map:
            save_contour_plot(cnm, Cn, f'{save_dir}/corr_contour_plot.jpg', cmap=cmap)
            save_roi_centers_plot(component_centers[good_idx], Cn, file_dir=f'{save_dir}/caiman_corr_rois_center.jpg', marker_size=30, cmap=cmap)

        # # play movie with results (original, reconstructed (A·C + b), amplified residual)
        # v = cnm.estimates.play_movie(
        #     images,
        #     q_min=1,
        #     q_max=99,
        #     include_bck=True,
        #     magnification=3,
        #     gain_res=1,
        #     gain_color=1,
        #     thr=0,
        #     use_color=False,
        #     display=False,
        # )
        # v.save(f'{save_dir}/caiman_movie.avi')
        # print('Movie Stored to Disk')
        # write_video_from_array(v, f'{save_dir}/movie.mp4', fps=10)

    # 5. Save results
    print('\n==== SAVING RESULTS =====\n')
    cnm.save(f'{save_dir}/cnmf_full_pipeline_results.hdf5')
    # Save manual curation
    # cnm.save(f'{save_dir}/cnmf_curated.hdf5')
    # Save the movie overlay
    if corr_map:
        np.save(f'{save_dir}/caiman_local_correlation_map.npy', Cn)

    # 6. Save ROI Traces
    # Export de-noised traces (C)
    # raw_traces = cnm.estimates.A.T @ Yr
    C = cnm.estimates.C  # shape (n_neurons, n_frames)
    pd.DataFrame(C.T).to_csv(f"{save_dir}/caiman_ca_traces.csv", index=False)

    # Export de-convolved spikes (S)
    # S = cnm.estimates.S
    # pd.DataFrame(S.T).to_csv(os.path.join(output_dir, "S_traces_spikes.csv"), index=False)

    # Export component centers (x, y)
    component_centers = cnm.estimates.center
    df_centers = pd.DataFrame(component_centers, columns=["y", "x"])  # CaImAn uses (row, col)
    df_centers.to_csv(f'{save_dir}/caiman_roi_centers.csv', index=False)

    print('\n==== CAIMAN FINISHED =====\n')
    if parallel:
        # Stop the cluster
        stop_server(dview=dview)


def save_roi_centers_plot(centers, bg_image, file_dir, marker_size=20, cmap='gray'):
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    plt.imshow(bg_image, cmap=cmap, vmin=np.min(bg_image), vmax=np.max(bg_image)/1.5)
    plt.scatter(centers[:, 1], centers[:, 0], color='red', marker='o', s=marker_size, alpha=0.5)
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    plt.savefig(file_dir, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_contour_plot(cnm, bg_image, file_dir, cmap):
    cnm.estimates.plot_contours(img=bg_image, idx=cnm.estimates.idx_components, cmap=cmap)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    plt.savefig(file_dir, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def motion_correction(file_name, pw_rigid, output_path, display_images=False):
    # First setup some parameters for data and motion correction
    # dataset dependent parameters
    fnames = [file_name]
    fr = 3  # imaging rate in frames per second
    decay_time = 5.0  # length of a typical transient in seconds
    dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
    max_shift_um = (12., 12.)  # maximum shift in um
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um

    # motion correction parameters
    # pw_rigid = True  # flag to select rigid vs pw_rigid motion correction

    # maximum allowed rigid shift in pixels
    max_shifts = [int(a / b) for a, b in zip(max_shift_um, dxy)]

    # start a new patch for pw-rigid motion correction every x pixels
    strides = tuple([int(a / b) for a, b in zip(patch_motion_um, dxy)])

    # overlap between patches (size of patch in pixels: strides+overlaps)
    overlaps = (24, 24)

    # maximum deviation allowed for patch with respect to rigid shifts
    max_deviation_rigid = 3

    # size of filter, change this one if algorithm does not work
    # gSig_filt = (3, 3)

    params_dict = {
        'data': {
            'fnames': fnames,
            'fr': fr,
            'decay_time': decay_time,
            'dxy': dxy
        },
        'motion': {
            'pw_rigid': pw_rigid,
            'max_shifts': max_shifts,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': 'copy'
        },
    }

    opts = params.CNMFParams(params_dict=params_dict)
    # play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the video press q

    if display_images:
        m_orig = load_movie_chain(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=60, magnification=2)

    # start a cluster for parallel processing
    c, dview, n_processes = setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # note that the file is not loaded in memory

    # Run motion correction using NoRMCorre
    mc.motion_correct(save_movie=True)

    # compare with original movie
    if display_images:
        m_orig = load_movie_chain(fnames)
        m_els = load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = concatenate([m_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                                      m_els.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit

    # Store corrected video to disk
    # Load the corrected memory-mapped file
    corrected_movie = load(mc.mmap_file)

    # Normalize and convert to uint16 (if needed)
    corrected_array = corrected_movie - corrected_movie.min()
    corrected_array /= corrected_array.max()
    corrected_array = (corrected_array * 65535).astype(np.uint16)

    # Save as TIFF stack
    # output_path = f'{file_name[:-4]}_motion_corrected.tif'
    tiff.imwrite(
        output_path,
        corrected_array,
        dtype=np.uint16,
        # bigtiff=False,
        imagej=True,
        # metadata=None,
        photometric='minisblack',
        # planarconfig=None,
        # compress=None,
        # tile=None,
        # resolution=None,
        # description=None
    )

    stop_server(dview=dview)


def c_viewer(cnm, tif_rec, mean_image):
    cnm.estimates.view_components(tif_rec, idx=cnm.estimates.idx_components, img=mean_image)


def main():
    # selected_file = ""
    file_label = None  # Declare it here so it’s visible in nested functions
    status_label = None

    def freeze_gui(freeze, status_text):
        if freeze:
            # Disable buttons and show status
            # open_button.config(state="disabled")
            run_motion_correction_button.config(state="disabled")
            run_source_extraction_button.config(state="disabled")
            status_label.config(text=status_text)
        else:
            # Re-enable and show complete
            # open_button.config(state="normal")
            run_motion_correction_button.config(state="normal")
            run_source_extraction_button.config(state="normal")
            status_label.config(text=status_text)
            root.update_idletasks()  # Ensure it's redrawn
            messagebox.showinfo("Analysis", f"Finished!")


    def open_file(text, f_types, change_label=True):
        # f_types = [("Text files", "*.txt"), ("All files", "*.*")]
        f = filedialog.askopenfilename(title=text, filetypes=f_types)
        if f:
            if change_label:
                file_label.config(text=f"{f}")
            return f
        else:
            return None

    def run_motion_correction():
        selected_file = open_file(text='Select Recording File', f_types=[("tif files", "*.tif *.tiff *.TIF *.TIFF")])
        if not selected_file:
            messagebox.showwarning("No file", "Please select a file first.")
            return

        # Disable buttons and show status
        freeze_gui(freeze=True, status_text='')
        root.update_idletasks()  # <- Force update to apply the GUI changes immediately
        time.sleep(0.1)  # Just for testing

        freeze_gui(freeze=True, status_text='Running Motion Correction, please wait ....')
        root.update_idletasks()  # <- Force update to apply the GUI changes immediately
        time.sleep(0.1)  # Just for testing

        # messagebox.showinfo("Analysis", f"Analysis started on:\n{selected_file}")
        output_path = f'{os.path.split(selected_file)[0]}/motion_corrected_{os.path.split(selected_file)[1]}'
        motion_correction(selected_file, pw_rigid=True, output_path=output_path, display_images=False)

        # Re-enable and show complete
        freeze_gui(freeze=False, status_text='Finished Motion Correction and stored new file to disk!')

    def run_source_extraction():
        selected_file = open_file(text='Select Recording File', f_types=[("tif files", "*.tif *.tiff *.TIF *.TIFF")])
        if not selected_file:
            messagebox.showwarning("No file", "Please select a file first.")
            return

        # Get Settings File
        # settings_dir = open_file(text='Select Settings File', f_types=[("settings file", "*.yaml")], change_label=False)
        # if not selected_file:
        #     messagebox.showwarning("No file", "Please select a file first.")
        #     return

        settings_dir = 'caiman_settings.yaml'
        try:
            with open(settings_dir, 'r') as f:
                params_dict = yaml.safe_load(f)
        except FileNotFoundError:
            print('ERROR COULD NOT FIND SETTINGS FILE')
            messagebox.showwarning("No Settings File", "ERROR: COULD NOT FIND SETTINGS FILE")
            return

        # Disable buttons and show status
        freeze_gui(freeze=True, status_text='Running Motion Correction, please wait ....')
        root.update_idletasks()  # <- Force update to apply the GUI changes immediately
        time.sleep(0.1)  # Just for testing

        # messagebox.showinfo("Analysis", f"Analysis started on:\n{selected_file}")
        output_folder= f'{os.path.split(selected_file)[0]}/caiman_output'
        os.makedirs(output_folder, exist_ok=True)

        corr_map = checkbox_var.get()
        print(f'Correlation Map: {corr_map}')
        source_extraction(selected_file, output_folder, params_dict=params_dict, visual_check=True, parallel=True, corr_map=corr_map)
        # Re-enable and show complete
        freeze_gui(freeze=False, status_text='Finished Cell Detection and stored data to disk!')

    root = tk.Tk()
    root.title("File Selector and Runner")
    root.geometry("600x400")

    # open_button = tk.Button(root, text="Open File", command=open_file)
    # open_button.pack(pady=10)

    nonlocal_file_label = tk.Label(root, text="No file selected")
    file_label = nonlocal_file_label  # Assign to the outer variable
    file_label.pack(pady=5, expand=True)

    run_motion_correction_button = tk.Button(root, text="Run Motion Correction", command=run_motion_correction)
    run_motion_correction_button.pack(pady=10, expand=True)

    # Frame to hold checkbox and button side by side
    row_frame = tk.Frame(root)
    row_frame.pack(pady=20, expand=True)

    run_source_extraction_button = tk.Button(row_frame, text="Run Cell Detection", command=run_source_extraction)
    run_source_extraction_button.pack(pady=10, expand=True)

    # Checkbox
    checkbox_var = tk.BooleanVar()  # Tracks checkbox state
    checkbox = tk.Checkbutton(row_frame, text="Correlation Map", variable=checkbox_var)
    checkbox.pack(side="left", padx=10, expand=True)

    status_label = tk.Label(root, text="READY", fg="blue", font=("Helvetica", 14))
    status_label.pack(pady=20, expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()

