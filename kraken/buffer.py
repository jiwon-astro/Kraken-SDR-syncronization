import time
import numpy as np
from multiprocessing import shared_memory

# Ring buffer
class SharedRingBuffer:
    def __init__(self, data_shm, hdr_shm, n_slots, slot_bytes, create=False):
        self.n_slots = n_slots
        self.slot_bytes = slot_bytes # 2 * N_daq [bytes]

        # shared memory
        self.data_shm = data_shm 
        self.hdr_shm  = hdr_shm

        self.data = np.ndarray(
            (n_slots, slot_bytes), dtype=np.uint8, 
            buffer=self.data_shm.buf
        )
        buffer_offset = 0
        # write_seq: number of recorded frames
        self.write_seq = np.ndarray((1,), dtype=np.uint64, 
                                    buffer=self.hdr_shm.buf, offset=buffer_offset) # dim = (1,)
        buffer_offset += 8 # uint64 (8bytes)
        # valid_seq: frame number stored in given slot
        self.valid_seq = np.ndarray((n_slots,), dtype=np.uint64, 
                                    buffer=self.hdr_shm.buf, offset=buffer_offset) # dim = (n_slots,)
        buffer_offset += 8 * n_slots
        # timestamps: recorded time
        self.timestamps = np.ndarray((n_slots,), dtype=np.uint64, 
                                     buffer=self.hdr_shm.buf, offset=buffer_offset) # dim = (n_slots,)
        if create:
            self.write_seq[:] = 0
            self.valid_seq[:] = np.uint64(2**64 - 1)  # invalid sentinel
            self.timestamps[:] = 0

    def write_block(self, raw_sample):
        seq = int(self.write_seq[0])
        slot = seq % self.n_slots # slot location

        arr = np.frombuffer(raw_sample, dtype=np.uint8)
        if arr.size != self.slot_bytes:
            raise ValueError('input sample size mismatch: sample size = {arr.size}, slot = {self.slot_bytes}')
        self.data[slot, :arr.size] = arr # (1) store samples
        self.timestamps[slot] = time.monotonic_ns() # (2) timestamp (프로그램이 실행되는 동안 측정한 시간, tracking the slot update)
        self.valid_seq[slot] = seq
        self.write_seq[0] = seq + 1

    def read_block(self, seq):
        slot = seq % self.n_slots
        if int(self.valid_seq[slot])!=seq:
            # not written yet or already overwritten
            return None, None
        raw_sample = self.data[slot].copy()
        ts = int(self.timestamps[slot])
        return raw_sample, ts
    
    def close(self):
        try: self.data_shm.close()
        except Exception: pass
        try: self.hdr_shm.close()
        except Exception: pass

    def unlink(self):
        try: self.data_shm.unlink()
        except Exception: pass
        try: self.hdr_shm.unlink()
        except Exception: pass

def calc_buffer_spec(n_daq, n_slots):
    # buffer memory sizes
    slot_bytes = 2 * n_daq
    data_nbytes = n_slots * slot_bytes
    hdr_nbytes = 8 * (2 * n_slots+1)
    return slot_bytes, data_nbytes, hdr_nbytes

def create_buffers(channels, n_daq, n_slots, prefix="SDR"):
    # create buffers
    slot_bytes, data_nbytes, hdr_nbytes = calc_buffer_spec(n_daq, n_slots)
    
    buffers = {}; buffer_info = {}
    for idx in channels:
        data_name = f'{prefix}{idx}_data'
        hdr_name = f'{prefix}{idx}_hdr'
        # shared memory
        data_shm = shared_memory.SharedMemory(name = data_name, create=True, size=data_nbytes)
        hdr_shm = shared_memory.SharedMemory(name = hdr_name, create=True, size=hdr_nbytes)
        # buffer
        buffers[idx] = SharedRingBuffer(data_shm, hdr_shm, n_slots, slot_bytes, create=True)
        buffer_info[idx] = {"data_mem": data_name, "hdr_mem": hdr_name, 
                            "n_slots": n_slots, "slot_bytes": slot_bytes}
    return buffers, buffer_info

def attach_buffers(buffer_info):
    # load buffers based on memory adresses
    buffers = {}
    for idx, info in buffer_info.items():
        data_shm = shared_memory.SharedMemory(name=info["data_mem"], create=False)
        hdr_shm = shared_memory.SharedMemory(name=info["hdr_mem"], create=False)
        buffers[idx] = SharedRingBuffer(data_shm, hdr_shm, int(info["n_slots"]), int(info["slot_bytes"]), create=False)
    return buffers

def close_buffers(buffers):
    for buf in buffers.values():
        buf.close()

def unlink_buffers(buffers):
    for buf in buffers.values():
        buf.unlink()

def encode_buffer_info(buffer_info, channels):
    # Convert dict to CLI-safe comma-separated strings.
    channels = list(channels)
    data_names = ",".join(buffer_info[ch]["data_mem"] for ch in channels)
    hdr_names = ",".join(buffer_info[ch]["hdr_mem"] for ch in channels)
    return data_names, hdr_names

def decode_buffer_info(channels, prefix, n_daq, n_slots):
    slot_bytes, _, _ = calc_buffer_spec(n_daq, n_slots)
    # Reconstruct buffer_info inside worker from CLI args.
    buffer_info = {}
    for idx in channels:
        data_name = f'{prefix}{idx}_data'
        hdr_name = f'{prefix}{idx}_hdr'
        buffer_info[idx] = {
                "data_mem": data_name,
                "hdr_mem": hdr_name,
                "n_slots": n_slots,
                "slot_bytes": slot_bytes,
        }
    return buffer_info
