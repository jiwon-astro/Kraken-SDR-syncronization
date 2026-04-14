# py_sdr_worker.py
import os
import sys
import time
import signal
import struct
import argparse
import threading

import numpy as np
from .buffer import attach_buffers, decode_buffer_info, close_buffers

class DeviceWorker(threading.Thread):
    def __init__(self, channel_id, 
                 sample_rate, center_freq, gain,
                 n_daq, mode, buffer, 
                 stop_event: threading.Event, use_async = False
):
        super().__init__(daemon=True)
        self.channel_id = channel_id
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.gain = gain
        self.n_daq = int(n_daq)

        self.use_async = use_async
        self.mode = mode
        if (mode == 'calib') and (channel_id == 0):
            # enable noise source (only idx=0)
            self.bias_tee_enable=True
        else: self.bias_tee_enable = False

        self.buffer = buffer
        self.stop_event = stop_event

        self.frame_id = 0
        self.sdr = None

    def configure(self):
        from .kraken_sdr import SDR

        self.sdr = SDR(device_index=self.channel_id)
        self.sdr.sdr_init(
            idx=self.channel_id,
            center_freq=self.center_freq,
            sample_rate=self.sample_rate,
            gain=self.gain,
            bias_tee_enable=self.bias_tee_enable,
        )
        self.sdr.reset_buffer()

        print(
            f"device={self.channel_id} opened "
            f"fs={self.sample_rate} fc={self.center_freq} gain={self.gain} n_daq={self.n_daq}",
            file=sys.stderr, flush=True
        )

    def close(self):
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception:
                pass
            self.sdr = None

    def run_sync(self):
        while not self.stop_event.is_set():
            raw_sample = self.sdr.read_bytes(2 * self.n_daq) # ctypes array -> bytes
            self.buffer.write_block(raw_sample)
            self.frame_id += 1

    def run_async(self):
        def _callback(values, context):
            # callback
            if self.stop_event.is_set():
                try: context.cancel_read_async()
                except Exception: pass
                return
            
            raw_sample = bytes(values)
            self.buffer.write_block(raw_sample)
            self.frame_id += 1

        try:
            self.sdr.read_bytes_async(_callback, num_bytes=2 * self.n_daq, context=self.sdr)
        finally:
            try:
                self.sdr.cancel_read_async()
            except Exception:
                pass

    def run(self):
        try:
            self.configure()
            if self.use_async: self.run_async()
            else: self.run_sync()
        except Exception as e:
            print(f"device={self.channel_id} error: {e!r}", 
                  file=sys.stderr, flush=True)
        finally:
            self.close()
            print(f"device={self.channel_id} closed", 
                  file=sys.stderr, flush=True)

def parse_devices(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, required=True, help="e.g. 0,1,2")
    parser.add_argument("--sample-rate", type=float, default=2.56e6)
    parser.add_argument("--center-freq", type=float, default=1420e6)
    parser.add_argument("--gain", type=float, default=50.0)
    parser.add_argument("--mode", type=str, default='data')
    parser.add_argument("--n_daq", type=int, default=2**17)
    parser.add_argument("--n_slots", type=int, default=64)
    # buffer shared memory
    parser.add_argument("--shm_prefix", type=str, required=True)
    parser.add_argument("--use_async", action='store_true') #store_true = flagging
    args = parser.parse_args()

    stop_event = threading.Event()
    def _handle_signal(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    device_ids = parse_devices(args.devices)
    if not device_ids:
        print("No device IDs given", file=sys.stderr, flush=True)
        sys.exit(1)

    buffer_info = decode_buffer_info(device_ids, prefix=args.shm_prefix,
                                     n_daq=args.n_daq, n_slots=args.n_slots)
    buffers = attach_buffers(buffer_info)

    workers = [
        DeviceWorker(channel_id=idx, sample_rate=args.sample_rate,
                     center_freq=args.center_freq, gain=args.gain,
                     n_daq=args.n_daq, mode=args.mode, buffer=buffers[idx], 
                     stop_event=stop_event, use_async=args.use_async)
        for idx in device_ids]

    for w in workers: w.start()

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=2.0)
        close_buffers(buffers)

if __name__ == "__main__":
    main()