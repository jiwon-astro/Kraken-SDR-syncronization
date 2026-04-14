import sys
import time
import numpy as np                  
from pathlib import Path
import threading
from collections import deque # Double-Ended Queue
from tqdm.auto import tqdm

import cupy as cp
from cupyx import empty_pinned

# Kraken synchronization
from .kraken_sdr import N_daq
from .buffer import create_buffers, close_buffers, unlink_buffers
from .bridge import launch_worker, stop_worker, aligned_input_blocks
from .gpu import coarse_delay_correction_gpu, gcc_phat_gpu, cfo_gpu, clear_fft_caches

def pad_nan(seq, target_len):
    return np.pad(seq, (max(target_len - len(seq),0), 0), constant_values=np.nan)

def discard_initial_samples(samples, window = 100, Nmax = 3000, 
                            discard_idx = 2048, plot = False):
    N = len(samples)
    if (N < window) or (N < Nmax): 
        print(f'Data length is too short')
        return None

    rms = [np.std(samples[i:i+window]) for i in range(0,Nmax,window)] # rms value of given time windows
    diffs = np.diff(rms) # difference

    # jump point detection
    start_i = 1
    jump_idx = np.argmax(np.abs(diffs)) # maximum difference
    if jump_idx <= 1: std_crit = np.std(rms) * 0.5     # 전체 표준편차로 대체
    else: 
        std_crit = np.std(rms[0:jump_idx])*0.5
        start_i += jump_idx
        
    #stable point detection
    # if no detection, discard_idx set to default value
    for i in range(start_i, Nmax//window):
        snippet = rms[i:i+5] # within 5 windows
        curr_std = np.std(snippet)
        if curr_std < std_crit:
            idx = i*window
            discard_idx = 1 << idx.bit_length() # 2^N
            break
    return discard_idx

def acquisition_monitor(buffers, channels, pbar,
                         stop_event, acquisition_done, poll_interval = 0.05):
    last = 0
    while not stop_event.is_set():
        acq_seq = min(int(buffers[idx].write_seq[0]) for idx in channels)
        acq_seq = min(acq_seq, n_frames)

        if acq_seq > last:
            pbar.update(acq_seq - last)
            last = acq_seq

        if last >= n_frames:
            acquisition_done.set() # notice event
            break

        time.sleep(poll_interval)

# --- alignments ---
def process_aligned_stream(proc, buffers, channels, 
                           std_ch = 0, N_corr=2**15, overlap=0.5, 
                           discard=True, verbose=True):
    # channels : channel list
    # n_frames : the number of frames, N_sample/N_daq
    stride = int((1-overlap) * N_corr)
    N_steps = int(np.ceil((N_daq-N_corr)/stride)) + 1
       
    lags_list = []; dlags_list = []; phi_list = []
    tau0s = {idx: 0 for idx in channels} # tau0 for each channel (relative to std_ch)
    cut_ind = 0
    aligned_output = None; wp = None

    # outer progress bar
    acq_pbar   = tqdm(total=n_frames, desc = "Acquisition", unit="frame",
                    dynamic_ncols = True, position = 0, leave = True,
                    file = sys.stderr, disable = not verbose)
   
    align_pbar = tqdm(total = n_frames, desc = "Alignments", unit = "frame",
                       dynamic_ncols = True, position = 1, leave = True,
                       file = sys.stderr, disable = not verbose)
    
    # acquisition monitor
    monitor_stop = threading.Event()
    acquisition_done = threading.Event()
    monitor = threading.Thread(target=acquisition_monitor, 
                               args=(buffers, channels, acq_pbar, monitor_stop, acquisition_done), daemon=True)
    monitor.start()

    t0 = time.time()
    frame_series = aligned_input_blocks(buffers, channels, stop_after=n_frames)
    try:
        for fidx, samples in frame_series:
            if fidx>=n_frames: break
            if fidx == 0:
                if discard:
                    # discard initial samples during stabilization
                    cut_ind = np.max([discard_initial_samples(sample) for sample in samples.values()])
                    for idx in channels:
                        samples[idx] = samples[idx][cut_ind:]

                # memory preallocation for aligned output (fast GPU -> CPU copies)
                N_total = (n_frames - 1) * N_daq + (N_daq - cut_ind)
                if verbose: print(f"Discard initial {cut_ind} samples from first frame")

                aligned_output = {ch: cp.empty((N_total,), dtype=cp.complex64) for ch in channels}
                wp = {ch: 0 for ch in channels} # writing position

            # ========== Time Delay Correction ============
            # -----------------------
            # Coarse delay correction
            # ------------------------
            shifted, tau_curr = coarse_delay_correction_gpu(samples, tau0s = tau0s)
            # Residual correction
            x_std = shifted[std_ch]
            
            m = int(x_std.size) # memory size
            aligned_output[std_ch][wp[std_ch]:wp[std_ch]+m] = x_std.copy()
            wp[std_ch] += m
            
            # ----------------------------------
            # Fractional delay correction + CFO
            # ----------------------------------
            batch_dlag = {}; batch_phi = {}
            for idx in channels:
                if std_ch == idx: continue
                x_i = shifted[idx] 
                dtaus_i, x_i_corr = gcc_phat_gpu(x_std, x_i, 
                                                N_corr = N_corr, overlap = overlap,
                                                norm = True, to_host = False) 
                if len(x_i_corr) != len(x_i):
                    print(len(x_i_corr),len(x_i),len(x_std))
                #peak_idx = taus_i + N_corr//2
                phi_i = cfo_gpu(x_std, x_i, N_corr = N_corr, 
                                overlap = overlap, to_host = True)#, peak_idx = peak_idx)
                phi_i -= phi_i[0] # relative phase shift

                # gpu -> cpu
                dtaus_i = cp.asnumpy(dtaus_i)
                # padding for compensate discarded samples at initial frames
                dtaus_i = pad_nan(dtaus_i, N_steps)
                phi_i   = pad_nan(phi_i, N_steps)

                # copy aligned block to pinned host and free GPU immediately
                m_i = int(x_i_corr.size)
                aligned_output[idx][wp[idx]:wp[idx]+m_i] = x_i_corr.copy()
                wp[idx] += m_i
                del x_i_corr, x_i # clear memory
                
                batch_dlag[idx] = dtaus_i.copy()
                batch_phi[idx] = phi_i.copy()

                # update tau0 
                tau0s[idx] += tau_curr[idx] #last block's delay (starting point of next frame)

            del x_std
            lags_list.append(tau_curr)
            dlags_list.append(batch_dlag)
            phi_list.append(batch_phi)
            
            align_pbar.update(1)
    
    finally:
        monitor_stop.set()
        monitor.join(timeout=1.0)
        acq_pbar.close()
        align_pbar.close()

    if aligned_output is None: raise RuntimeError("No frames received from worker")

    #t_corr_delta = time.time() - t_corr_init
    # reorganize
    aligned_output = {k: v.get() for k, v in aligned_output.items()} # GPU -> CPU
    taus     = np.vstack([list(lags.values()) for lags in lags_list])
    dtaus    = np.vstack([list(dlags.values()) for dlags in dlags_list])
    cfo_phi  = np.vstack([list(phis.values()) for phis in phi_list])

    if verbose: 
        elapsed = time.time() - t0
        print("-----------------------------------------------")
        print("Synchronization done")
        print(f"Elapsed = {elapsed:.2f} s")
        print('-----------------------------------------------')
    return aligned_output, taus, dtaus, cfo_phi

# =========================================
# Main function
# =========================================
def multi_sdr_acquisition(channels, N_sample, 
                          fs=2.56e6, fc=1420e6, gain=50, N_slots = 64, 
                          N_corr = 2**15, overlap = 0.5, std_ch=0, mode='data', 
                          shm_prefix="kraken", use_async=False, discard=True, verbose = False):    
    global t0, n_frames
    channels = list(channels)
    if not channels: raise ValueError("channels must not be empty")

    N_ch = len(channels)
    print('-----------------------------------------------')
    print(f'Channels = {channels} ({N_ch} ch)')
    VERBOSE = verbose
    # acquisition
    n_frames = int(np.ceil(N_sample / N_daq))
    t_exp = N_sample / fs
    print('-----------------------------------------------')
    print(f'Exposure time = {t_exp:.2f}s / total {N_sample} samples')
    
    t0 = time.time()
    # ring buffer
    buffers, buffer_info = create_buffers(channels=channels, n_daq=N_daq, n_slots=N_slots, prefix=shm_prefix)
    # launch processes
    proc = None
    try:
        proc = launch_worker(device_ids=channels, sample_rate=fs, center_freq=fc, gain=gain,
                             n_daq=N_daq, n_slots = N_slots, mode=mode, prefix=shm_prefix, use_async=use_async)
        # synchronized with ch 0 
        aligned, taus, dtaus, cfo_phi = process_aligned_stream(proc, buffers, channels, N_corr = N_corr, overlap = overlap, 
                                                               discard=discard, std_ch = std_ch, verbose=verbose)
    finally:
        if proc is not None: stop_worker(proc)
        # clean buffers
        close_buffers(buffers)
        unlink_buffers(buffers)
        # clean gpu memory / cache
        cp.get_default_memory_pool().free_all_blocks() # release memory
        #cp.get_default_pinned_memory_pool().free_all_blocks()
        clear_fft_caches()
        
    return aligned, taus, dtaus, cfo_phi
