from rtlsdr import RtlSdr # RTL-SDR package, set_bias_tee_gpio included

"""
Basic sdr parameters
"""
fc_def   = 1420e6  # center frequency
fs_def   = 2.56e6  # ADC sampling rate
gain_def = 50      # ADC gain (max 50)

N_daq    = 2**18   # Size of DAQ buffer  

class SDR(RtlSdr):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.init_args = args
        self.init_kwargs = kwargs
        self._saved_settings = {}
        
    # initialize sdr
    def sdr_init(self, idx, center_freq = fc_def, 
                 sample_rate = fs_def, gain = gain_def, 
                 freq_correction = False, bias_tee_enable = False):
        self._saved_settings = {
            'idx': idx,
            'center_freq': center_freq,
            'sample_rate': sample_rate,
            'gain': gain,
            'freq_correction': freq_correction,
            'bias_tee_enable': bias_tee_enable
        }
        self.device_index = idx # device index
        self.set_sample_rate(sample_rate) # Sampling rate (Hz) (=1/2 bandwidths)
        self.set_center_freq(center_freq) # Central frequency
        self.set_gain(gain) # SDR Gain
        if freq_correction:
             self.set_freq_correction(1) # Sampling frequency correction flag
        if bias_tee_enable: # bias tee enable
            result = self.set_bias_tee_gpio(idx+1, 1)
            print(f"Bias-tee enabled = {result}")
                
    # Reconnection  
    def reconnect(self):
        try:
            self.close() # close previous connections
        except Exception as e:
            print(f"Reconnection failed: {e}")
        #time.sleep(0.1) # waiting for USB resource released
        # Reset 
        super().__init__(*self.init_args, **self.init_kwargs)
        # Load previous settings
        if self._saved_settings:
            s = self._saved_settings
            self.sdr_init(
                idx=s['idx'],
                center_freq=s['center_freq'],
                sample_rate=s['sample_rate'],
                gain=s['gain'],
                freq_correction=s['freq_correction'],
                bias_tee_enable=s['bias_tee_enable'])