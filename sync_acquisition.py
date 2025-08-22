import sys
import time
import numpy as np                  
from pathlib import Path
from threading import Thread, Lock, Condition
from collections import deque # Double-Ended Queue
from tqdm.auto import tqdm

import cupy as cp
from cupyx import empty_pinned

# Kraken synchronization
from .kraken_sdr import SDR, N_daq
from .gpu import coarse_delay_correction_gpu, gcc_phat_gpu, cfo_gpu, clear_fft_caches

# global variables
buffer = {} # data buffer
counts = {} # count
threads = {}
active_channels = set() # active channels
threads_lock = Lock() # thread lock
sdr_lock = Lock() # ensuring handle SDR object
cond = Condition() # Thread conditions

# =========================================
# simultaneous acquisition of multi-channel
# =========================================
def multi_sdr_acquisition(sdrs, N_sample, 
                          N_corr = 2**15, overlap = 0.5,
                         verbose = False):    
    global t0, n_frames, channels, N_ch, VERBOSE
    channels = list(sdrs.keys())
    N_ch = len(channels)
    init_buffer(channels)
    print('-----------------------------------------------')
    print(f'Channels = {channels} ({N_ch} ch)')
    VERBOSE = verbose
    
    # open check
    for idx in channels:
        sdr = sdrs[idx]
        if not sdr.device_opened:
            print(f"Opening Device {idx}")
            sdr.reconnect()
    
    # acquisition
    fs = sdrs[channels[0]].sample_rate
    n_frames = int(np.ceil(N_sample / N_daq))
    t_exp = N_sample / fs
    print('-----------------------------------------------')
    print(f'Exposure time = {t_exp:.2f}s / total {N_sample} samples')
    
    t0 = time.time()
    sdr_read_threads(sdrs, N_sample)
    
    # monitor
    monitor = Thread(target = sdr_monitor, args = (sdrs,), daemon = True)
    monitor.start()
    
    # synchronized with ch 0 
    aligned, taus, dtaus, cfo_phi = process_aligned_blocks(N_corr = N_corr, overlap = overlap) 
    
    # clean gpu memory / cache
    cp.get_default_memory_pool().free_all_blocks() # release memory
    #cp.get_default_pinned_memory_pool().free_all_blocks()
    clear_fft_caches()
    
    # clean up threads
    monitor.join()
    with threads_lock:
        for thread in threads.values(): thread.join() # all thread close
        
    with sdr_lock:
        for sdr in sdrs.values(): 
            if not sdr.device_opened: 
                print(f'Device {sdr.device_index} is closed')
                continue
            time.sleep(0.1)
            sdr.reset_buffer()  # all sdr close
                        
    return aligned, taus, dtaus, cfo_phi

# --- Condition checking ---
def preparation_check(next_frame):
    # current cycle is accomplished for all channels
    return all(
        buffer[idx] and buffer[idx][0][0] == next_frame
        for idx in active_channels)
    
def complete_check():
    # acquisition is finished
    return all(counts[idx] >= n_frames for idx in active_channels)

# --- initializers ---
#initialize buffers
def init_buffer(channels):
    global active_channels
    active_channels = set(channels) # all channels begin as activated
    with cond:
        for idx in channels:
            buffer[idx] = deque()
            counts[idx] = 0
    
# --- sdr async read ---
# callback function 
def sdr_callback(samples, sdr):
    # values : acquired samples from read_samples_async
    # if read_samples_async.context = None -> context = SDR instance
    # callback 내에서 cancel_read_async 사용시 내부에서 예상하지 못한 종료 시점으로 오류 발생 가능.
    idx = sdr.device_index
    current_time = time.time() - t0
    with cond:
        if idx not in active_channels: 
            print(f"[Device {idx}] disabled channel")
            return
        # accumulate samples in buffer
        if counts[idx] < n_frames:
            buffer[idx].append((counts[idx], current_time, samples))
            counts[idx]+=1
            cond.notify_all() # condition alarm to other threads - checking any frame available to process
            if VERBOSE:
                print(f"Device {idx} / {counts[idx]} / Elapsed time = {current_time:.6f} s / {len(samples)} samples")

def sdr_disable(sdr):
    idx = sdr.device_index
    try:
        sdr.cancel_read_async()
    except Exception as e:
        print(f"Error cancelling on device {idx}: {e}")
    # join callback
    """
    with threads_lock:
        thread = threads.pop(idx, None)
        if thread is not None:
            thread.join(timeout = 0.1)  # give it a time interval to clean up
    """
    # Remove current channel from active channel list
    with cond:    
        if idx in active_channels:
            print(f"[Device {idx}] disabled")
            active_channels.remove(idx)
            cond.notify_all()

# async
def sdr_async(sdr, N_sample):
    idx = sdr.device_index
    try: 
        sdr.read_samples_async(sdr_callback, min(N_sample, N_daq), context = sdr)
        # context = None? : 내부 참조가 사라질 수도 있음?
    except Exception as e:
        msg = str(e)
        if "LIBUSB_ERROR_NOT_FOUND" in msg: return # normal execution
        # error occured
        print(f"Error in async read on device {idx}: {e}")
        sdr_disable(sdr) # disable, remove from active channel list

def sdr_read_threads(sdrs, N_sample):
    # asynchronous reading of multiple channels with threading
    for idx in channels:
        # daemon : main thread 종료시 함께 종료
        thread = Thread(target = sdr_async, args = (sdrs[idx], N_sample), daemon = True)
        with threads_lock:
            threads[idx] = thread
        thread.start()

# --- status monitoring ---
def sdr_monitor(sdrs):
    with cond:
        cond.wait_for(lambda: complete_check()) 
    print(f"Completely received samples from all {N_ch} channels")
    print('-----------------------------------------------')
    for idx in active_channels:
        try: 
            sdrs[idx].cancel_read_async()
            print(f"[Device {idx}] async read cancled")
        except Exception as e:
            print(f"Error cancelling on device {idx}: {e}")
            
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
    # plot
    if plot:
        plt.plot(rms)
        plt.axvline(jump_idx)
        plt.axvline(i,ls='dotted',color='k')
    return discard_idx

# --- alignments ---
def process_aligned_blocks(std_ch = 0, discard = True,
                           N_corr = 2**15, overlap = 0.5,
                           progress = True):
    # channels : channel list
    # n_frames : the number of frames, N_sample/N_daq
    stride = int((1-overlap) * N_corr)
    N_steps = int(np.ceil((N_daq-N_corr)/stride)) + 1
       
    lags_list = []; dlags_list = []; phi_list = []
    tau0s = {idx: 0 for idx in active_channels} # tau0 for each channel (relative to std_ch)
    
    #t_corr_init = time.time()
    next_frame = 0
    # outer progress bar
    pbar_frames = tqdm(total = n_frames, desc = "Alignments", unit = "frame",
                       dynamic_ncols = True, position = 0, leave = True,
                       file = sys.stderr, disable = not progress)
    while next_frame < n_frames:
        with cond: 
            # waiting for all channels prepared
            cond.wait_for(lambda: preparation_check(next_frame))
            current_channels = [idx for idx in channels if idx in active_channels]
            # ready
            # queue에서 하나 제거, 제거한 원소 반환 
            samples = {idx: buffer[idx].popleft()[2] for idx in current_channels}
            if next_frame ==0:
                # discard initial samples during stabilization
                cut_ind = np.max([discard_initial_samples(sample) for sample in samples.values()])
                for idx in active_channels:
                    samples[idx] = samples[idx].copy()[cut_ind:]
                print(f"Discard intial {cut_ind} samples")
                # memory preallocation for aligned output (fast GPU -> CPU copies)
                N_total = (n_frames - 1) * N_daq + (N_daq - cut_ind)
                #aligned_output = {idx: empty_pinned((N_total,), dtype=np.complex64) for idx in current_channels}
                aligned_output = {idx: cp.empty((N_total,), dtype=cp.complex64) for idx in current_channels}
                wp = {idx: 0 for idx in current_channels} # writing position
                
        if std_ch not in current_channels: 
            print(f'Reference channel [ch {std_ch}] disconnected')
            return
        
        # --------- Time Delay Correction -----------
        # Coarse delay correction
        shifted, tau_curr = coarse_delay_correction_gpu(samples,tau0s = tau0s)
        # Residual correction
        x_std = shifted[std_ch]
        
        m = int(x_std.size) # memory size
        aligned_output[std_ch][wp[std_ch]:wp[std_ch]+m] = x_std
        wp[std_ch] += m
        
        # corr = {}
        batch_dlag = {}; batch_snr = {}; batch_phi = {}
        for idx in current_channels:
            if std_ch == idx: continue
            x_i = shifted[idx] 
            # Fractional delay correction
            dtaus_i, x_i_corr = gcc_phat_gpu(x_std, x_i, 
                                             N_corr = N_corr, overlap = overlap,
                                             norm = True, to_host = False) 
            if len(x_i_corr) != len(x_i):
                print(len(x_i_corr),len(x_i),len(x_std))
            #peak_idx = taus_i + N_corr//2
            phi_i = cfo_gpu(x_std, x_i, N_corr = N_corr, 
                            overlap = overlap, to_host = True)#, peak_idx = peak_idx)
            phi_i -= phi_i[0]
            
            # gpu -> cpu
            dtaus_i = cp.asnumpy(dtaus_i)
            # padding for compensate discarded samples at initial frames
            dtaus_i = pad_nan(dtaus_i, N_steps)
            phi_i   = pad_nan(phi_i, N_steps)
            
            # copy aligned block to pinned host and free GPU immediately
            mi = int(x_i_corr.size)
            aligned_output[idx][wp[idx]:wp[idx]+mi] = x_i_corr
            wp[idx] += mi
            del x_i_corr, x_i
            
            batch_dlag[idx] = dtaus_i.copy()
            batch_phi[idx] = phi_i.copy()

            # update tau0 
            tau0s[idx] += tau_curr[idx] #last block's delay (starting point of next frame)
            
        del x_std
        lags_list.append(tau_curr)
        dlags_list.append(batch_dlag)
        phi_list.append(batch_phi)
        
        next_frame += 1
        if progress: pbar_frames.update(1)
    
    #t_corr_delta = time.time() - t_corr_init
    # reorganize
    aligned_output = {k: v.get() for k, v in aligned_output.items()} # GPU -> CPU
    taus     = np.vstack([list(lags.values()) for lags in lags_list])
    dtaus    = np.vstack([list(dlags.values()) for dlags in dlags_list])
    cfo_phi  = np.vstack([list(phis.values()) for phis in phi_list])
    
    print("Synchronization done")
    print('-----------------------------------------------')
    return aligned_output, taus, dtaus, cfo_phi #t_corr_delta