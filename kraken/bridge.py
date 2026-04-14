# worker_bridge.py
import sys
import time
import subprocess
import threading
import numpy as np
from pathlib import Path

WORKING_DIR = Path(__file__).resolve().parent.parent # parent of the kraken package folder
WORKER_MODULE = 'kraken.sdr_worker'

def unpack_iq_u8_to_complex64(raw_sample: bytes) -> np.ndarray:
    """
    Convert raw RTL-SDR uint8 interleaved IQ bytes to complex64
    same idea as packed_bytes_to_iq() in rtlsdr.py (vectorized version).
    """
    data = np.frombuffer(raw_sample, dtype=np.uint8).astype(np.float32)
    iq = data.view(np.complex64) # [I0, Q0, I1, Q1, ...]
    iq /= 127.5
    iq -= (1.0 + 1.0j)
    return iq.astype(np.complex64, copy=False)

def aligned_input_blocks(buffers, channels, start_seq=0, stop_after=None, poll_interval=0.001):
    next_seq = start_seq
    while True:
        # exit condition
        if (stop_after is not None) and (next_seq >= stop_after):
            return
        
        min_write_seq = min(int(buffers[idx].write_seq[0]) for idx in channels)
        # wait for next frame
        if next_seq >= min_write_seq:
            time.sleep(poll_interval)
            continue

        samples = {}; flag = True
        for idx in channels:
            raw_sample, ts = buffers[idx].read_block(next_seq)
            if raw_sample is None:
                print(f"[aligned_input_blocks] frame {next_seq} not ready on channel {idx}", 
                      file=sys.stderr)
                flag = False
                break
            samples[idx] = unpack_iq_u8_to_complex64(raw_sample)
        
        if flag: yield next_seq, samples
        next_seq += 1

# ===================================
# SDR I/O Worker
# ===================================
# launch worker processes
def launch_worker(
    worker_module=WORKER_MODULE,
    device_ids=(0, 1),
    sample_rate=2.56e6,
    center_freq=1420e6,
    gain=50,
    n_daq=2**17,
    n_slots=64,
    mode='data',
    prefix="kraken",
    use_async=False
):
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", worker_module,
        "--devices", ",".join(map(str, device_ids)),
        "--sample-rate", str(sample_rate),
        "--center-freq", str(center_freq),
        "--gain", str(gain),
        "--n_daq", str(n_daq),
        "--n_slots", str(n_slots),
        "--mode", str(mode),
        "--shm_prefix", str(prefix),
    ]
    if use_async: cmd.append("--use_async")
    # Popen (Pipe open): create named pipe for IPC
    # create new worker process as subprocess of main function
    # stdout과 stderr 모두 subprocess.PIPE - 이 instance의 I/O를 내가 관리하겠음.
    proc = subprocess.Popen(cmd, cwd=str(WORKING_DIR),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)
    _logger(proc)
    return proc

def stop_worker(proc, timeout=3.0):
    if proc.poll() is not None: return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

def _logger(proc, prefix="[worker]"):
    """
    Read worker stderr on a background thread
    """
    def _reader():
        if proc.stderr is None: return
        for line in iter(proc.stderr.readline,b""):
            if not line: break
            try: text = line.decode("utf-8", errors="replace").rstrip()
            except Exception:
                text = repr(line)
            print(f"{prefix} {text}", file=sys.stderr)

    th = threading.Thread(target=_reader, daemon=True)
    th.start()
    return th


