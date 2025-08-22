# Kraken-SDR-syncronization
Multi-channel SDR acquisition with real-time synchronization using GPU-based FFT

Developed as a preparation of the radio interferometry experiment (the Undergraduate Radio Lab(in KAIST))

## Implementation
- Asynchronous reading from each channel (threading)
- $N_{\rm daq}=2^{18}$ samples at once, $\Delta t\sim0.1\;\rm ms/frame$ (at $f_s=2.56\rm MHz$ sampling (complex I/Q samples))
<img width="2280" height="1002" alt="image" src="https://github.com/user-attachments/assets/e0bce6f0-e175-46ee-a3fb-a11e309a4c0e" />
<img width="2045" height="965" alt="image" src="https://github.com/user-attachments/assets/ae380be7-df4b-4df9-b38d-a871339957f8" />
