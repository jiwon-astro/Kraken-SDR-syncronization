# Kraken-SDR-syncronization
Python package for the multi-channel SDR acquisition with real-time synchronization using GPU-based FFT
[<Kraken SDR>](<https://www.crowdsupply.com/krakenrf/krakensdr>)

Developed as a preparation for the radio interferometry experiment (the Undergraduate Radio Lab(in KAIST))

*Working example*
<img width="1329" height="372" alt="image (22)" src="https://github.com/user-attachments/assets/94df26f6-5dbd-4036-bbf2-7d8f3cde076b" />

*Single-channel delay tracking result (~5 min exposure)*
<img width="1460" height="760" alt="image (23)" src="https://github.com/user-attachments/assets/6b762564-447f-46af-a544-e33bb5474468" />

## Pre-requisites
- 'pyrtlsdr[lib]' (modified-version for GPIO control (designated for Kraken SDR internal calibration source))
- 'cupy'
- 'tqdm' (optional)

## Implementation
- Asynchronous reading from each channel (threading) (Maximum 5 channels(Kraken SDR's capability))
- $N_{\rm daq}=2^{18}$ samples at once, $\Delta t\sim0.1\;\rm ms/frame$ (at $f_s=2.56\rm MHz$ sampling (complex I/Q samples))
- Checking the preparation of all channels, completed accumulating a single frame.
- After preparation, performing synchronization
- If all channel completed $N_{\rm cycle}$ times readout, recall 'cancel_async_read()' that stops SDR acquisition. 
<img width="2280" height="1002" alt="image" src="https://github.com/user-attachments/assets/e0bce6f0-e175-46ee-a3fb-a11e309a4c0e" />
<img width="2045" height="965" alt="image" src="https://github.com/user-attachments/assets/ae380be7-df4b-4df9-b38d-a871339957f8" />

## Synchronization scheme
The synchronization conducts cross-correlation implemented with GPU-based FFT.
1. **Coarse delay correction**
   - Calculate cross-correlation for $N_{\rm daq}$ samples (full samples of single frame) between Ch 0 and the others.
   - Applying the circular buffer shift to correct the integer delay.
   - The delay of the $i$-th frame will be accumulated for the next frame.
   
2. **Residual delay correction**
   - Subdividing the samples of single frame into $N_{\rm corr}(<N_{\rm daq})$ with given overlap fraction (default = 0.5)
   - *Residual integer delay*: Calculating the relative delay from Ch 0 and the others at given sub-patches to track minor integer-delay drifts
   - *Fractional delay*: perform least square fitting of a linear function onto the frequency-phase diagram.
      $\phi (f) = \phi_0 -2\pi f\Delta \tau$

3. **Carrier frequency offset (CFO) correction**
      
