import numpy as np                   
import cupy as cp # GPU acceleration
from cupyx.scipy.fft import get_fft_plan
from scipy.signal import windows

# Global plan cache
_fft_plans = {} 
_ifft_plans = {}

# ==============================================================
# ==================== GPU FFT/IFFT ============================
# ==============================================================
def _coerce_device(device: int | cp.cuda.Device | None) -> cp.cuda.Device:
    """Return a CuPy Device context from int/Device/None."""
    if device is None:
        return cp.cuda.Device()  # current device
    if isinstance(device, cp.cuda.Device):
        return device
    return cp.cuda.Device(int(device))

def fft_gpu(x, axis = -1, fftshift = False, device = None):
    """
    Compute 1D FFT of x on GPU
    x : numpy array (N,), complex128
    Convert input array data type - CUDA devices supports single precision
    """
    # GPU array
    dev = _coerce_device(device)
    with dev:
        if not isinstance(x, cp.ndarray):
            x_gpu = cp.asarray(x, dtype=cp.complex64)
        else:
            x_gpu = x.astype(cp.complex64, copy=False)
        x_gpu = cp.ascontiguousarray(x_gpu)
        # fft plan
        key = (x_gpu.shape, x_gpu.dtype, axis)
        plan = _fft_plans.get(key)
        if plan is None:
            plan = get_fft_plan(x_gpu, axes = (axis,),
                                value_type = 'C2C')
            _fft_plans[key] = plan
        # perform FFT
        with plan:
            X_gpu = cp.fft.fft(x_gpu, axis=axis)                     
        if fftshift:
            X_gpu = cp.fft.fftshift(X_gpu, axes = axis)
        return X_gpu

def ifft_gpu(X, axis = -1, fftshift = False, device = None):
    """
    Compute 1D inverse FFT of x on GPU
    X : numpy array (N,)
    Convert input array data type - CUDA devices supports 32bit
    """
    # GPU array
    dev = _coerce_device(device)
    with dev:
        if not isinstance(X, cp.ndarray):
            X_gpu = cp.asarray(X, dtype=cp.complex64)
        else:
            X_gpu = X.astype(cp.complex64, copy=False)
        X_gpu = cp.ascontiguousarray(X_gpu)
        # fft plan
        key = (X_gpu.shape, X_gpu.dtype, axis)
        plan = _ifft_plans.get(key)
        if plan is None:
            plan = get_fft_plan(X_gpu, axes = (axis,),
                                value_type = 'C2C')
            _ifft_plans[key] = plan
        # perform IFFT
        with plan:
            x_gpu = cp.fft.ifft(X_gpu, axis=axis)                    
        if fftshift:
            x_gpu = cp.fft.fftshift(x_gpu, axes = axis)
        return x_gpu
        
def clear_fft_caches():
    _fft_plans.clear()
    _ifft_plans.clear()
    
def calc_psd(sample, fs, fc=0, N_fft=1024, overlap=0.0,
             window_func=np.hamming, to_host=True, device=None):
    """
    Welch PSD (GPU)
    - sample: 1D complex IQ (np.ndarray 또는 cp.ndarray)
    - fs: sampling rate [Hz]
    - fc: center frequency [Hz]
    - N_fft: segment 길이
    - overlap: [0,1)
    - window_func: (e.g.) np.hamming, np.hanning, scipy.signal.windows.hann)
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if N_fft <= 0:
        raise ValueError("N_fft must be positive")
    dev = _coerce_device(device)
    with dev:
        x = cp.asarray(sample, dtype=cp.complex64)
        N = int(x.size)
        if N < N_fft:
            freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1.0/(fs/1e6))) + (fc/1e6)
            empty_psd = np.zeros(N_fft, dtype=np.float32)
            return (freq, empty_psd) if to_host else (freq, cp.asarray(empty_psd))

        step = int(N_fft * (1.0 - overlap))
        if step <= 0:
            raise ValueError("step becomes zero; choose smaller overlap or larger N_fft")

        # window function
        win_np = window_func(N_fft).astype(np.float32, copy=False)
        win = cp.asarray(win_np)
        U = cp.mean(win * win) 

        # ---- Segment batch (GPU) ----
        # shape: (n_seg, N_fft)
        xb = cp.lib.stride_tricks.sliding_window_view(x, N_fft)[::step]
        xb = xb * win[None, :]

        # ---- Batch FFT -> Power spectrum ----
        F_seg = fft_gpu(xb, axis=1, fftshift=True, device=dev) / N_fft
        psd_seg = (cp.abs(F_seg) ** 2) / U
        psd = cp.mean(psd_seg, axis=0)

        # ---- Frequency Grid ----
        freq = np.fft.fftshift(np.fft.fftfreq(N_fft, d=1.0/(fs/1e6))) + (fc/1e6)

    if to_host:
        return freq, cp.asnumpy(psd)
    else:
        return freq, psd
    
# ==============================================================
# ========= Delay correction w/ FFT-based correlation ==========
# ==============================================================
def coarse_delay_correction_gpu(samples, tau0s = None, device = None, to_host = False):
    dev = _coerce_device(device)
    # samples : dictionary (idx : data)
    ch_idx = list(samples.keys())
    x_cpu = np.stack([*samples.values()], axis = 0) 
    M, N = x_cpu.shape
    with dev:
        x = cp.asarray(x_cpu, dtype = cp.complex64)
        x = cp.ascontiguousarray(x)
        
         # index setup
        row_idx = cp.arange(M, dtype = cp.int64)[:, None]
        col_idx0 = cp.arange(N, dtype = cp.int64)[None, :]
        
        # apply shift from previous acquisition
        if tau0s is not None:
            tau0 = cp.asarray([tau0s[ch_idx[i]] for i in range(M)], dtype = cp.int64)
            col_idx = (col_idx0 - tau0[:,None]) % N
            x = x[row_idx, col_idx]
        
        # calculate delay
        X = fft_gpu(x, axis = 1, device = dev)
        R = X[0,None]*cp.conj(X[1:])
        corr_b = ifft_gpu(R, axis = 1, fftshift = True, device = dev)

        # sample delay between ch0 and others
        lags = cp.arange(-N//2, N//2, dtype = cp.int64) 
        idx_max = cp.abs(corr_b).argmax(axis = 1)
        dtaus = lags[idx_max]
        tau_samp = cp.concatenate([cp.zeros(1, dtype=cp.int64), dtaus])
        
        # roll index
        col_idx = (col_idx0 - tau_samp[:, None])%N
        x_shifted = x[row_idx, col_idx]

    if to_host:
        shifted = {ch_idx[i]: cp.asnumpy(x_shifted[i]) for i in range(M)}
        tau_coarse = {ch_idx[i]: int(tau_samp[i].item()) for i in range(M)}
    else:
        shifted = {ch_idx[i]: x_shifted[i] for i in range(M)}
        tau_coarse = {ch_idx[i]: int(tau_samp[i].item()) for i in range(M)}
    return shifted, tau_coarse
    
def frac_delay_gpu(freq_sft, phi_b, weight_b = None,
                   dc_reject = 2e3, sigma = 3.0, n_iter = 5,  
                   device = None, to_host = False):
    """
    Fractional (sub-integer) delay
    - Calculate delay by Least-square fitting of phase-frequency curve of cross-spectrum 
    """
    dev = _coerce_device(device)
    with dev:
        N_steps, N_corr = phi_b.shape
        f = freq_sft.astype(cp.float64, copy=False)[None,:]

        # ============ initialization =============
        tau_frac = cp.zeros(N_steps, dtype=cp.float64)
        phi0 = cp.zeros(N_steps, dtype=cp.float64)
        
        if weight_b is None: weight_b = cp.ones_like(phi_b, dtype=cp.float64)
        else: weight_b = weight_b.astype(cp.float64, copy=False)
            
        # value mask
        mask0 = cp.isfinite(weight_b)
        mask0 &= cp.abs(f) > dc_reject
        w_med = cp.nanmedian(weight_b, axis = 1, keepdims=True)
        w_mad = cp.nanmedian(cp.abs(weight_b - w_med), axis = 1, keepdims=True)
        # apply cleaning to spectrum 
        threshold = w_med + 0.5*w_mad
        mask0 &= (weight_b > threshold)
        
        # =============== unwraping ================
        # cleaning shoud be preceded before unwraping phase angle
        y = cp.zeros_like(phi_b, dtype=cp.float64)
        for i in range(N_steps):
            mi = mask0[i]
            if int(mi.sum()) >= 2:
                y[i, mi] = cp.unwrap(phi_b[i, mi], axis = 0)
            else: y[i, mi] = phi_b[i, mi]
                
        # ===== fitting frequency - phase curve =====
        # Build design matrix
        mask = mask0.copy()
        X = cp.stack([cp.ones_like(f), f], axis = 2) # (1, N_corr, 2)
        X = cp.broadcast_to(X, (N_steps, N_corr, 2))
        for _ in range(n_iter):
            Xm = X * mask[:,:,None]
            ym = y[:,:,None] * mask[:,:,None]
            
            Xm_T = Xm.transpose(0,2,1)
            A = cp.matmul(Xm_T, Xm) # X^T X, (N_steps, 2, 2)
            b = cp.matmul(Xm_T, ym) # X^T y, (N_steps, 2, 1)
            # Least squares
            theta = cp.linalg.solve(A, b) # (N_steps, 2, 1) 
            alp = theta[:,0,0] # intercept
            bet = theta[:,1,0] # slope
            
            res = (y - (alp[:,None] +  bet[:,None] * f))
            # sigma clipping
            std = cp.nanstd(cp.where(mask, res, cp.nan), axis = 1, keepdims = True)
            res_mask= cp.abs(res) > sigma * std
            if not bool((res_mask & mask).sum()): break # no updates
            mask &= ~res_mask
            
            tau_frac = - bet / (2*cp.pi) # fractional delay
            phi0     = alp # phase offset
          
        if to_host: return cp.asnumpy(tau_frac)
        else: return tau_frac 

def ewma_timeavg_gpu(A, alpha=0.2):
    # A: (T,F) complex or real, avg along T (time)
    out = cp.empty_like(A)
    acc = cp.zeros_like(A[0])
    for t in range(A.shape[0]):
        acc = alpha * A[t] + (1 - alpha) * acc
        out[t] = acc
    return out

def gcc_phat_gpu(x, y, N_corr = 2**15,
                 overlap = 0.5, window = windows.hann,
                 norm = True, device = None, to_host = False):
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
    - Removing the amplitude information, Phase information only, Beneficial to noise suppression
    - Correcting coarse, integer delay by searching peak of cross-correlation signal in lag domain
    - 
    """
    dev = _coerce_device(device)
    # tau0 : accumulated delay from previous samples
    eps = 1e-6
    N = len(x) # Nx = Ny
    stride = int((1-overlap) * N_corr)
    N_steps = int(np.ceil((N-N_corr)/stride)) + 1
    
    # zero-padding
    N_pad = (N_steps-1) * stride +  N_corr

    with dev:
        x_pad = cp.zeros(N_pad, dtype = cp.complex64)
        y_pad = cp.zeros(N_pad, dtype = cp.complex64)
        if not isinstance(x, cp.ndarray):
            x_pad[:N] = cp.asarray(x, dtype=cp.complex64)
            y_pad[:N] = cp.asarray(y, dtype=cp.complex64)
        else:
            x_pad[:N] = x.astype(cp.complex64, copy=False)
            y_pad[:N] = y.astype(cp.complex64, copy=False)
            
        xb = cp.lib.stride_tricks.sliding_window_view(x_pad, N_corr)[::stride]
        yb = cp.lib.stride_tricks.sliding_window_view(y_pad, N_corr)[::stride]
    
        if norm:
            # should be non-zero mean??
            xb = (xb - xb.mean(axis=1, keepdims=True)) / (xb.std(axis=1, keepdims=True) + eps)
            yb = (yb - yb.mean(axis=1, keepdims=True)) / (yb.std(axis=1, keepdims=True) + eps)

        # blockwise FFT
        Xb = fft_gpu(xb, axis = 1, device = dev)  # envelope - |x|(t)
        Yb = fft_gpu(yb, axis = 1, device = dev)
        freq = cp.fft.fftfreq(N_corr).astype(cp.float32)
        freq_sft = cp.fft.fftshift(freq) 
        
        # ============ calculate correlation ===========
        R_b =Xb * cp.conj(Yb) # Cross-power spectrum
        Sxx = cp.abs(Xb)**2
        Syy = cp.abs(Yb)**2
        
        # Time window averaging for stability
        R_bar  = ewma_timeavg_gpu(R_b,  alpha=0.2)        # ≈ average of ~5 frames
        Sxxbar = ewma_timeavg_gpu(Sxx, alpha=0.2)
        Syybar = ewma_timeavg_gpu(Syy, alpha=0.2)
        
        # PHAT
        R_phat_b = R_bar / (cp.abs(R_bar) + eps) # only phase components
        corr_b = ifft_gpu(R_phat_b, axis = 1, fftshift = True, device = dev)

        # per-block tau
        lags = cp.arange(-N_corr//2, N_corr//2, dtype = cp.int32) 
        idx_max = cp.abs(corr_b).argmax(axis = 1)
        dtaus_int = lags[idx_max].astype(cp.float64)  # lags (in sample units)
        
        # ============ fractional delay ================
        phi_b = cp.fft.fftshift(cp.angle(R_bar), axes = 1) # yet unwraped
        # intial guess
        #phi0 = phi_b[:,N_corr//2]
        #b0   = cp.median(cp.diff(phi_b, axis = 1), axis = 1)*N_corr
        
        msc = cp.abs(R_bar) / cp.sqrt(Sxxbar * Syybar) # magnitude-squared cross-spectrum
        dtaus_frac = frac_delay_gpu(freq_sft, phi_b, weight_b = msc, 
                                    sigma = 3.0, device = dev)
        dtaus = dtaus_int + dtaus_frac
    
        # ============== delay filter ===================
        H = cp.exp(- 2j * cp.pi * freq[None,:] * dtaus[:,None]) # (n_blocks, N_corr)

        if norm:
            yb_original = cp.lib.stride_tricks.sliding_window_view(y_pad, N_corr)[::stride]
            Yb = fft_gpu(yb_original, axis=1, device = dev)

        Yb_delayed = Yb * H
        yb_aligned = ifft_gpu(Yb_delayed, axis = 1, device = dev)
        if N_steps > 1:
            yb_aligned = yb_aligned[:,N_corr - stride:] # valid samples
        
        # ========== overlap-save ===========
        y_aligned = cp.zeros(N_pad, dtype = cp.complex64)
        for i in range(N_steps):
            idx = i*stride
            if N_steps == 1:
                ub = min(N, N_corr)
                y_aligned[:ub] = yb_aligned[0, :ub]
            else:
                ub = min(stride, N - idx)
                y_aligned[idx:idx + ub] = yb_aligned[i, :ub]
        y_aligned = y_aligned[:N]
        
    if to_host:
        # R_cpu = cp.asnumpy(R_b)
        # corr_b_cpu = cp.asnumpy(corr_b)
        dtaus_cpu  = cp.asnumpy(dtaus)
        y_aligned_cpu = cp.asnumpy(y_aligned)
        return dtaus_cpu, y_aligned_cpu
    else:
        return dtaus, y_aligned

def cfo_gpu(x, y, peak_idx = None, 
            N_corr = 2**15, overlap = 0.5, 
            device = None, to_host = False):
    """
    Carrier frequency offset (CFO) correction
    - Measuring phase angle drift of correlation at lag domain 
    """
    dev = _coerce_device(device)
    eps = 1e-6
    N = len(x) # Nx = Ny
    stride = int((1-overlap) * N_corr)
    n_steps = int(np.ceil((N-N_corr)/stride)) + 1
    
    # zero-padding
    N_pad = (n_steps-1) * stride +  N_corr
    with dev:
        x_pad = cp.zeros(N_pad, dtype = cp.complex64)
        y_pad = cp.zeros(N_pad, dtype = cp.complex64)
        if not isinstance(x, cp.ndarray):
            x_pad[:N] = cp.asarray(x, dtype=cp.complex64)
            y_pad[:N] = cp.asarray(y, dtype=cp.complex64)
        else:
            x_pad[:N] = x.astype(cp.complex64, copy=False)
            y_pad[:N] = y.astype(cp.complex64, copy=False)

        xb = cp.lib.stride_tricks.sliding_window_view(x_pad, N_corr)[::stride]
        yb = cp.lib.stride_tricks.sliding_window_view(y_pad, N_corr)[::stride]

        # Blockwise FFT
        Xb = fft_gpu(xb, axis = 1, device = dev)
        Yb = fft_gpu(yb, axis = 1, device = dev)

        # Calculate correlation
        R =Xb * cp.conj(Yb) # Cross-power spectrum
        R_phat = R #/ (cp.abs(R) + eps) # only phase components
        corr_b = ifft_gpu(R_phat, axis = 1, 
                          fftshift = True, device = dev)
        if peak_idx is None: peak_idx = cp.abs(corr_b).argmax(axis = 1)
        else: peak_idx = cp.asarray(peak_idx)

        # phi = 2 * pi * delta f * t
        phi  = cp.unwrap(cp.angle(corr_b[cp.arange(n_steps),peak_idx]))
    
    if to_host: return cp.asnumpy(phi) 
    else: return phi