import matplotlib
matplotlib.use('Agg')  # Disable GUI plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tifffile as tiff
import time
import yaml
import threading
import logging
import traceback

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

# CAIMAN IMPORTS
import multiprocessing
from caiman import load, load_memmap, stop_server, load_movie_chain, concatenate
from caiman.cluster import setup_cluster
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline

# Do some logging
logging.basicConfig(
    filename='caiman_gui_log.txt',
    filemode='w',  # <--- Overwrite the file at each run (otherwise set to "a")
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

def source_extraction(file_name, save_dir, params_dict=None, parallel=True, corr_map=False):
    # start a cluster
    if parallel:
        # Use all but one CPU
        total_cpus = multiprocessing.cpu_count()
        n_cpus = total_cpus - 1

        # Ensure at least one process is used
        n_cpus = max(1, n_cpus)
        logging.info(f'FOUND: {total_cpus} CPUs')
        logging.info(f'USING: {n_cpus} CPUs')

        c, dview, n_processes = setup_cluster(
            backend='multiprocessing', n_processes=n_cpus, single_thread=True
        )
    else:
        dview = None
        n_processes = 1

    # File Name
    f_names = [file_name]
    if len(f_names) <= 0:
        print('\n==== ERROR: No tiff file recording found =====\n')
        return

    if params_dict is None:
        print('ERROR: COULD NOT FIND PARAMETER SETTINGS')
        return
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

    print("Dims:", dims, "Frames:", T)
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    # 5. Save results
    print('\n==== SAVING RESULTS =====\n')
    cnm.save(f'{save_dir}/cnmf_full_pipeline_results.hdf5')

    # 6. Save ROI Traces
    # Export de-noised traces (C)
    # raw_traces = cnm.estimates.A.T @ Yr
    C = cnm.estimates.C  # shape (n_neurons, n_frames)

    # Subset traces
    accepted_idx = cnm.estimates.idx_components
    rejected_idx = cnm.estimates.idx_components_bad
    C_accepted = cnm.estimates.C[accepted_idx, :]  # shape: (n_accepted, n_frames)
    C_rejected = cnm.estimates.C[rejected_idx, :]

    # Generate column names: cell_1, cell_2, ..., cell_n
    column_names = [f"cell_{i + 1}" for i in range(len(C_accepted))]
    column_names_rejected = [f"cell_{i + 1}" for i in range(len(C_rejected))]

    pd.DataFrame(C_accepted.T, columns=column_names).to_csv(f"{save_dir}/caiman_accepted_ca_traces.csv", index=False)
    pd.DataFrame(C_rejected.T, columns=column_names_rejected).to_csv(f"{save_dir}/caiman_rejected_ca_traces.csv", index=False)

    # Export component centers (x, y)
    component_centers = cnm.estimates.center
    df_centers_accepted = pd.DataFrame(component_centers[accepted_idx, :], columns=["y", "x"])  # CaImAn uses (row, col)
    df_centers_rejected = pd.DataFrame(component_centers[rejected_idx, :], columns=["y", "x"])  # CaImAn uses (row, col)

    df_centers_accepted.to_csv(f'{save_dir}/caiman_accepted_roi_centers.csv', index=False)
    df_centers_rejected.to_csv(f'{save_dir}/caiman_rejected_roi_centers.csv', index=False)

    if corr_map:
        print('\n==== COMPUTING CORRELATION MAP, THIS CAN TAKE SOME TIME .... =====\n')
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
        np.save(f'{save_dir}/caiman_local_correlation_map.npy', Cn)

    print('\n==== CAIMAN FINISHED =====\n')

    if parallel:
        # Stop the cluster
        stop_server(dview=dview)

def save_roi_centers_plot(centers, bg_image, file_dir, marker_size=20, cmap='gray'):
    vmin = np.percentile(bg_image, 1)  # lower 1st percentile
    vmax = np.percentile(bg_image, 99)  # upper 99th percentile
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    plt.imshow(bg_image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.scatter(centers[:, 1], centers[:, 0], color='red', marker='o', s=marker_size, alpha=0.5)
    plt.axis('off')  # remove axes and ticks
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding/margins
    plt.savefig(file_dir, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_contour_plot(cnm, bg_image, file_dir, cmap):
    vmin = np.percentile(bg_image, 1)  # lower 1st percentile
    vmax = np.percentile(bg_image, 99)  # upper 99th percentile
    # vmax = np.max(bg_image)  # upper 99th percentile

    # Plot with idx (generates subplot layout with 2 axes)
    cnm.estimates.plot_contours(img=bg_image, idx=cnm.estimates.idx_components, cmap=cmap)

    # Access current figure and all axes
    fig = plt.gcf()
    axes = fig.get_axes()

    # Loop through each subplot and update its image contrast
    for ax in axes:
        for im in ax.get_images():  # should be just one image per subplot
            im.set_clim(vmin=vmin, vmax=vmax)

    # Formatting
    fig.set_size_inches(15, 10)
    for ax in axes:
        ax.axis('off')
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(file_dir, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def motion_correction(file_name, pw_rigid, output_path, display_images=False):
    print('\n==== RUN MOTION CORRECTION =====\n')
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
    print('\n==== STORING MOTION CORRECTED FILE TO DISK =====\n')
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
    print('\n==== FINISHED MOTION CORRECTION =====\n')

def c_viewer(cnm, tif_rec, mean_image):
    cnm.estimates.view_components(tif_rec, idx=cnm.estimates.idx_components, img=mean_image)

def detect_neuropil_rois(file_dir, tif_file, neuropil_roi_dir, output_dir):
    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    import read_roi
    from matplotlib.path import Path

    # file_dir = f'{sw_dir}/caiman_output/cnmf_full_pipeline_results.hdf5'
    # tif_file_dir = f'{sw_dir}/rec/'
    # tif_file = f'{tif_file_dir}/{os.listdir(tif_file_dir)[0]}'
    # neuropil_roi_dir = f'{sw_dir}/neuropil.roi'

    # Load Data
    cnm = load_CNMF(file_dir)
    tif_rec = load_tiff_recording(tif_file, flatten=True)
    try:
        neuropil_roi = read_roi.read_roi_file(neuropil_roi_dir)
    except FileNotFoundError:
        print(f'ERROR in {tif_file}')
        print('COULD NOT FINED IMAGEJ ROI FILE')
        return

    mean_image = np.mean(tif_rec, axis=0)

    idx_components = cnm.estimates.idx_components
    component_centers = cnm.estimates.center[idx_components]
    df_centers = pd.DataFrame(component_centers, columns=["y", "x"])  # CaImAn uses (row, col)

    # Neuropil detection
    # --- 1. Load and parse neuropil ROI polygon ---
    neuropil = neuropil_roi['neuropil']
    polygon_x = np.array(neuropil['x'])
    polygon_y = np.array(neuropil['y'])
    polygon_vertices = np.column_stack((polygon_x, polygon_y))
    neuropil_path = Path(polygon_vertices)

    # --- 2. Convert CaImAn (y, x) component centers to (x, y) for polygon check ---
    component_xy = df_centers[["x", "y"]].values  # Now in (x, y)

    # --- 3. Check which components are inside the neuropil polygon ---
    inside_mask = neuropil_path.contains_points(component_xy)

    # --- 4. Create a DataFrame with the result ---
    df_centers["inside_neuropil"] = inside_mask

    # Optional: Filter just inside or outside
    df_inside = df_centers[df_centers["inside_neuropil"]]
    df_outside = df_centers[~df_centers["inside_neuropil"]]

    # --- Plot setup ---
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mean_image, cmap='gray', origin='upper')  # Show the mean image

    # --- Plot neuropil ROI polygon outline ---
    ax.plot(neuropil['x'] + [neuropil['x'][0]],  # x becomes col
            neuropil['y'] + [neuropil['y'][0]],  # y becomes row
            linestyle='--', color='cyan', linewidth=2, label='Neuropil ROI')

    # --- Plot component centers ---
    # Flip x and y back for plotting (since image is row=y, col=x)
    ax.scatter(df_inside["x"], df_inside["y"], c='lime', s=20, label='Inside Neuropil', alpha=0.7)
    ax.scatter(df_outside["x"], df_outside["y"], c='red', s=20, label='Outside Neuropil', alpha=0.7)

    # --- Labels and legend ---
    ax.set_title("Component Centers Overlaid on Mean Image", fontsize=14)
    ax.legend(loc='upper right')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/neuropil_detection.jpg', dpi=300)
    plt.close(fig)

    # Get accepted components
    # A_accepted = cnm.estimates.A[:, idx_components]
    C_accepted = cnm.estimates.C[idx_components]

    # Separate traces
    C_inside = C_accepted[inside_mask, :]  # Traces for components inside neuropil
    C_outside = C_accepted[~inside_mask, :]  # Traces for components outside neuropil

    # pd.DataFrame(C_accepted.T).to_csv(f"{sw_dir}/caiman_output/accepted_caiman_ca_traces.csv", index=False)
    pd.DataFrame(C_inside.T).to_csv(f"{output_dir}/neuropil_caiman_ca_traces.csv", index=False)
    pd.DataFrame(C_outside.T).to_csv(f"{output_dir}/cells_caiman_ca_traces.csv", index=False)

    # Export component centers (x, y)
    df_centers_inside = pd.DataFrame(component_centers[inside_mask, :], columns=["y", "x"])
    df_centers_outside = pd.DataFrame(component_centers[~inside_mask, :], columns=["y", "x"])
    df_centers_inside.to_csv(f'{output_dir}/neuropil_caiman_roi_centers.csv', index=False)
    df_centers_outside .to_csv(f'{output_dir}/cells_caiman_roi_centers.csv', index=False)

def create_figures(cnmf_path, save_dir, corr_map):
    from caiman.source_extraction.cnmf.cnmf import load_CNMF
    plt.ioff()  # Turn off interactive mode
    # Load the CNMF object
    cnm = load_CNMF(cnmf_path)
    cmap = 'viridis'

    # load memory mapped file
    Yr, dims, T = load_memmap(cnm.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # bg_image = np.mean(images, axis=0)
    bg_image = np.std(images, axis=0)

    # Get ROI centers
    component_centers = cnm.estimates.center
    good_idx = cnm.estimates.idx_components

    # Compare Accepted and Rejected Components
    save_contour_plot(cnm, bg_image, f'{save_dir}/caiman_mean_img_contour_plot.jpg', cmap=cmap)
    save_roi_centers_plot(component_centers, bg_image,
                          file_dir=f'{save_dir}/caiman_mean_img_rois_center.jpg', marker_size=30, cmap=cmap)
    save_roi_centers_plot(component_centers[good_idx], bg_image,
                          file_dir=f'{save_dir}/caiman_accepted_mean_img_rois_center.jpg', marker_size=30, cmap=cmap)

    if corr_map:
        # Load the local correlation map
        Cn = np.load(f'{save_dir}/caiman_local_correlation_map.npy')
        save_contour_plot(cnm, Cn, f'{save_dir}/corr_contour_plot.jpg', cmap=cmap)
        save_roi_centers_plot(component_centers[good_idx], Cn, file_dir=f'{save_dir}/caiman_corr_rois_center.jpg',
                              marker_size=30, cmap=cmap)

def main():
    file_label = None
    spinner = None

    def check_tiff_file(tif_file_dir):
        with tiff.TiffFile(tif_file_dir) as tif_file:
            dtype = tif_file.pages[0].dtype  # e.g., dtype('uint16')

            if dtype == 'int16':
                log_to_gui('Tiff File is "int16" ... will convert it to "unit16" to match CAIMAN')
                logging.info('Converting Tiff File to uint16')
                data = tif_file.asarray()  # This loads the image into memory
                if data.ndim <= 2:
                    # Tifffile did only load one frame
                    # New Dimensions: ZYX
                    data = np.stack([page.asarray() for page in tif_file.pages])


                # Convert to uint16 by offsetting
                data_uint16 = (data.astype(np.int32) + 32768).astype(np.uint16)
                # Save as new TIFF
                new_file = f'{os.path.split(tif_file_dir)[0]}/uint16_{os.path.split(tif_file_dir)[1]}'

                # Save with ImageJ compatibility
                tiff.imwrite(
                    new_file,
                    data_uint16,
                    dtype='uint16',
                    imagej=True,
                )

                return new_file
            return tif_file_dir

    def freeze_gui(freeze):
        if freeze:
            run_motion_correction_button.config(state="disabled")
            run_source_extraction_button.config(state="disabled")
            neuropil_button.config(state="disabled")
            checkbox_corr_map.config(state="disabled")
            checkbox_parallel.config(state="disabled")
            # status_label.config(text=status_text)
            spinner.start()
        else:
            run_motion_correction_button.config(state="normal")
            run_source_extraction_button.config(state="normal")
            neuropil_button.config(state="normal")
            checkbox_corr_map.config(state="normal")
            checkbox_parallel.config(state="normal")
            # status_label.config(text=status_text)
            spinner.stop()
            root.update_idletasks()
            messagebox.showinfo("Analysis", f"Finished!")

    def open_file(text, f_types, change_label=True):
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
            # messagebox.showwarning("No file", "Please select a file first.")
            log_to_gui("No file, Please select a file first.")
            logging.info("No file found.")
            return

        log_to_gui("Running motion correction...")
        logging.info("Motion correction started.")
        freeze_gui(freeze=True)
        output_path = f'{os.path.split(selected_file)[0]}/motion_corrected_{os.path.split(selected_file)[1]}'
        motion_correction(selected_file, pw_rigid=True, output_path=output_path, display_images=False)
        log_to_gui("Motion Correction Finished!")
        logging.info("Motion Correction Finished.")
        freeze_gui(freeze=False)

    def run_source_extraction():
        selected_file = open_file(text='Select Recording File', f_types=[("tif files", "*.tif *.tiff *.TIF *.TIFF")])
        if not selected_file:
            # messagebox.showwarning("No file", "Please select a file first.")
            log_to_gui("No file, Please select a file first.")
            logging.info("No file found.")
            return

        settings_dir = 'caiman_settings.yaml'
        try:
            with open(settings_dir, 'r') as f:
                params_dict = yaml.safe_load(f)
        except FileNotFoundError:
            log_to_gui("ERROR: Could not find settings file.")
            logging.info("ERROR: Could not find settings file.")
            # messagebox.showwarning("No Settings File", "ERROR: COULD NOT FIND SETTINGS FILE")
            return

        freeze_gui(freeze=True)

        # Check Tiff File (has to match CAIMAN)
        selected_file = check_tiff_file(selected_file)

        output_folder = f'{os.path.split(selected_file)[0]}/caiman_output'
        os.makedirs(output_folder, exist_ok=True)
        corr_map = checkbox_corr_map_var.get()
        parallel_processing = checkbox_parallel_var.get()

        if parallel_processing:
            # print('\n==== MODE: PARALLEL PROCESSING ==== \n')
            log_to_gui("MODE: PARALLEL PROCESSING")
            logging.info("MODE: PARALLEL PROCESSING")
        else:
            # print('\n==== MODE: NON-PARALLEL PROCESSING ==== \n')
            log_to_gui("MODE: NON-PARALLEL PROCESSING")
            logging.info("MODE: NON-PARALLEL PROCESSING")

        if corr_map:
            # print('\n==== CORRELATION MAP SELECTED, THIS CAN EXTEND PROCESSING TIME EXTREMELY.... ==== \n')
            # print('\n==== .... MAKE SURE YOUR COMPUTER CAN HANDLE THIS ! ==== \n')
            log_to_gui("CORRELATION MAP: TRUE - This can be computationally demanding!")
            logging.info("CORRELATION MAP: TRUE")

        else:
            # print('\n==== NO CORRELATION MAP (DEFAULT) ==== \n')
            log_to_gui("CORRELATION MAP: FALSE")
            logging.info("CORRELATION MAP: FALSE")

        log_to_gui("Running Source Extraction (Cell Detection)...")
        logging.info("Source Extraction started.")
        log_to_gui("Parameters:")
        log_to_gui(yaml.dump(params_dict, sort_keys=False))
        logging.info("Parameters:")
        logging.info(yaml.dump(params_dict, sort_keys=False))

        # Run Source Extraction (Cell Detection)
        source_extraction(selected_file, output_folder, params_dict=params_dict, parallel=parallel_processing, corr_map=corr_map)

        # Create Figures
        log_to_gui("Creating Validation Figures...")
        logging.info("Creating Validation Figures.")
        cnmf_dir = f'{output_folder}/cnmf_full_pipeline_results.hdf5'
        create_figures(cnmf_dir, output_folder, corr_map)

        # Create Figures
        log_to_gui("Source Extraction Finished!")
        logging.info("Source Extraction Finished.")
        freeze_gui(freeze=False)

    def run_neuropil():
        tif_file = open_file(text='Select Recording File', f_types=[("tif files", "*.tif *.tiff *.TIF *.TIFF")])
        if not tif_file:
            # messagebox.showwarning("No file", "Please select a file first.")
            log_to_gui("ERROR: Could not find settings file.")
            logging.info("ERROR: Could not find settings file.")
            return

        caiman_file = open_file(text='Select CAIMAN File', f_types=[("caiman files", "*.hdf5")])
        if not caiman_file:
            log_to_gui("ERROR: Could not find CAIMAN results file.")
            logging.info("ERROR: Could not find CAIMAN results file.")
            # messagebox.showwarning("No file", "Please select a file first.")
            return

        roi_file = open_file(text='Select ROI File', f_types=[("roi files", "*.roi")])
        if not roi_file:
            # messagebox.showwarning("No file", "Please select a file first.")
            log_to_gui("ERROR: Could not find ROI file.")
            logging.info("ERROR: Could not find ROI file.")
            return

        log_to_gui("Starting Neuropil Detection ...")
        logging.info("Starting Neuropil Detection.")
        freeze_gui(freeze=True)
        output_folder = f'{os.path.split(tif_file)[0]}/caiman_output'
        detect_neuropil_rois(file_dir=caiman_file, tif_file=tif_file, neuropil_roi_dir=roi_file, output_dir=output_folder)
        freeze_gui(freeze=False)
        log_to_gui("Finished Neuropil Detection.")
        logging.info("Finished Neuropil Detection.")

    def safe_run(func):
        def wrapper():
            try:
                func()
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(tb)  # Log to file
                root.after(0, lambda: log_to_gui(f"[ERROR]\n{tb}"))
                root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=wrapper).start()

    root = tk.Tk()
    blank_icon = tk.PhotoImage(width=1, height=1)  # black 1x1 px icon
    root.iconphoto(True, blank_icon)
    root.title("File Selector and Runner")
    root.geometry("600x450")

    nonlocal_file_label = tk.Label(root, text="No file selected")
    file_label = nonlocal_file_label
    file_label.pack(pady=5, expand=True)

    run_motion_correction_button = tk.Button(root, text="Run Motion Correction", command=lambda: safe_run(run_motion_correction))
    run_motion_correction_button.pack(pady=10, expand=True)

    row_frame = tk.Frame(root)
    row_frame.pack(pady=20, expand=True)

    run_source_extraction_button = tk.Button(row_frame, text="Run Cell Detection", command=lambda: safe_run(run_source_extraction))
    run_source_extraction_button.pack(pady=10, expand=True)

    checkbox_corr_map_var = tk.BooleanVar()
    checkbox_corr_map = tk.Checkbutton(row_frame, text="Correlation Map", variable=checkbox_corr_map_var)
    checkbox_corr_map.pack(side="left", padx=10, expand=True)

    checkbox_parallel_var = tk.BooleanVar()
    checkbox_parallel = tk.Checkbutton(row_frame, text="Parallel", variable=checkbox_parallel_var)
    checkbox_parallel.pack(side="left", padx=10, expand=True)

    neuropil_button = tk.Button(root, text="Run Neuropil Detection", command=lambda: safe_run(run_neuropil))
    neuropil_button.pack(pady=10, expand=True)

    spinner = ttk.Progressbar(root, mode='indeterminate')
    spinner.pack(pady=10, fill='x')

    # === LOGGING TEXT BOX ===
    log_text = tk.Text(root, height=8, wrap='word', state='disabled', bg='#f0f0f0')
    log_text.pack(pady=10, fill='both', expand=True)

    def log_to_gui(message):
        log_text.config(state='normal')
        log_text.insert(tk.END, message + '\n')
        log_text.see(tk.END)
        log_text.config(state='disabled')

    def clear_log():
        log_text.config(state='normal')
        log_text.delete('1.0', tk.END)
        log_text.config(state='disabled')

    clear_log_button = tk.Button(root, text="Clear Log", command=clear_log)
    clear_log_button.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
