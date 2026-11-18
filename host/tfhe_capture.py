#!/usr/bin/env python3
"""Upload all-q-minus-one TFHE LWE ciphertexts and capture CW308 STM32F3.

SimpleSerial 1.1, classic TFHE Torus32, binary LWE key.
Edit experiment settings directly in main().
"""
import json
from pathlib import Path
import secrets
import struct
import time

import numpy as np

MASK32 = (1 << 32) - 1


def reference_phase(a, key, b):
    return (int(b) - sum(int(x) * int(s) for x, s in zip(a, key))) & MASK32


def reference_bit(phase):
    return int(0 < phase < (1 << 31))


def main():
    # Edit these settings directly before running this file.
    firmware_hex = None  # Path("/path/to/firmware.hex") to flash; None to skip.
    key_path = None  # Optional Path to a JSON binary key.
    seed = 123
    b = 0
    num_traces = 10
    samples = 20000
    downsample = 2
    offset = 0
    gain = 25
    no_capture = False
    out = Path("./data/tfhe_all_qminus1.npz")
    timeout_ms = 3000
    if out.exists():
        raise FileExistsError(out)

    # Connect to the scope and SimpleSerial 1.1 firmware.
    import chipwhisperer as cw
    scope = cw.scope()
    target = None
    try:
        scope.default_setup()
        target = cw.target(scope, cw.targets.SimpleSerial)
        if firmware_hex is not None:
            cw.program_target(scope, cw.programmers.STM32FProgrammer, str(firmware_hex))
        scope.io.nrst = "low"
        time.sleep(0.05)
        scope.io.nrst = "high_z"
        time.sleep(0.1)
        target.flush()
        # Query the firmware dimension and packet layout.
        target.simpleserial_write("i", bytearray())
        raw = target.simpleserial_read("r", 8, timeout=timeout_ms, ack=False)
        if raw is None or len(raw) != 8:
            raise RuntimeError("Missing firmware info")
        status = target.simpleserial_wait_ack(timeout=timeout_ms)
        if status != 0:
            raise RuntimeError(f"Target error: {status!r}")
        version, mode, n, chunk_words, word_bits, response_len, _ = struct.unpack("<BBHBBBB", bytes(raw))
        if version != 1 or mode != 0 or word_bits != 32 or chunk_words != 14 or response_len not in (1, 5):
            raise RuntimeError("Unsupported firmware")
        # Generate or load the known binary research key.
        if key_path:
            raw_key = json.loads(key_path.read_text())
            if (not isinstance(raw_key, list) or len(raw_key) != n or
                    any(type(x) is not int or x not in (0, 1) for x in raw_key)):
                raise ValueError(f"Key file must contain exactly {n} binary integers")
            key = np.array(raw_key, dtype=np.int32)
        elif seed is not None:
            key = np.random.default_rng(seed).integers(0, 2, n, dtype=np.int32)
        else:
            key = np.fromiter((secrets.randbits(1) for _ in range(n)),
                              dtype=np.int32, count=n)
        a = np.full(n, MASK32, dtype=np.uint32)
        expected_phase = reference_phase(a, key, b)
        expected_bit = reference_bit(expected_phase)
        # Clear device buffers, then upload key and a in fixed 59-byte packets.
        target.simpleserial_write("x", bytearray())
        status = target.simpleserial_wait_ack(timeout=timeout_ms)
        if status != 0:
            raise RuntimeError(f"Target error: {status!r}")
        for command, vector in (("k", key), ("a", a)):
            for start in range(0, n, chunk_words):
                words = [int(x) for x in vector[start:start + chunk_words]]
                count = len(words)
                words += [0] * (chunk_words - count)
                # Header: uint16 offset, uint8 valid count; payload: 14 uint32.
                packet = struct.pack("<HB14I", start, count, *words)
                target.simpleserial_write(command, bytearray(packet))
                status = target.simpleserial_wait_ack(timeout=timeout_ms)
                if status != 0:
                    raise RuntimeError(f"Upload failed: {command}, {start}, {status!r}")
        # Upload the scalar b separately.
        target.simpleserial_write("b", bytearray(struct.pack("<I", b)))
        status = target.simpleserial_wait_ack(timeout=timeout_ms)
        if status != 0:
            raise RuntimeError(f"Target error: {status!r}")
        print(f"n={n}, q=2**32, a[i]=0xFFFFFFFF, b=0x{b:08X}")
        print(f"Expected phase=0x{expected_phase:08X}, bit={expected_bit}")
        # Configure the ADC capture window; check coverage on your board.
        scope.gain.db = gain
        scope.adc.samples = samples
        scope.adc.downsample = downsample
        scope.adc.offset = offset
        scope.adc.timeout = 5
        waves, bits, phases, trigger_counts = [], [], [], []
        for index in range(num_traces):
            # Arm first; command p runs the triggered phase computation.
            if not no_capture:
                scope.arm()
            target.simpleserial_write("p", bytearray([0]))
            timed_out = scope.capture() if not no_capture else False
            # Read result before consuming its ACK.
            raw = target.simpleserial_read("r", response_len, timeout=timeout_ms, ack=False)
            if raw is None or len(raw) != response_len:
                raise RuntimeError("Missing decryption result")
            status = target.simpleserial_wait_ack(timeout=timeout_ms)
            if status != 0 or timed_out:
                raise RuntimeError(f"Decryption/capture failed: {status!r}, timeout={timed_out}")
            bit = raw[0]
            phase = struct.unpack("<I", bytes(raw[1:]))[0] if response_len == 5 else None
            wave = None if no_capture else np.array(scope.get_last_trace(), copy=True)
            if bit != expected_bit or (phase is not None and phase != expected_phase):
                raise RuntimeError(f"Arithmetic mismatch at trace {index}")
            bits.append(bit)
            phases.append(-1 if phase is None else phase)
            if wave is not None:
                waves.append(wave)
                try:
                    trigger_counts.append(int(scope.adc.trig_count))
                except (AttributeError, ValueError):
                    trigger_counts.append(-1)
            print(f"{index + 1}/{num_traces}: bit={bit}, samples={0 if wave is None else len(wave)}")
        # Save traces, known key/input and acquisition settings.
        metadata = {
            "n": n, "q": 1 << 32, "pattern": "all_q_minus_one",
            "samples": int(scope.adc.samples), "downsample": int(scope.adc.downsample),
            "offset": int(scope.adc.offset), "gain_db": float(scope.gain.db),
            "clkgen_freq": float(scope.clock.clkgen_freq),
            "adc_src": str(scope.clock.adc_src),
            "seed": seed, "phase_exposed": response_len == 5,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental replacement of an existing dataset.
        with out.open("xb") as output:
            np.savez_compressed(
                output, traces=np.asarray(waves, dtype=np.float32), a=a, key=key,
                b=np.uint32(b), bits=np.asarray(bits, dtype=np.uint8),
                phases=np.asarray(phases, dtype=np.int64),
                expected_phase=np.uint32(expected_phase),
                trigger_counts=np.asarray(trigger_counts, dtype=np.int64),
                metadata_json=np.array(json.dumps(metadata)),
            )
        print(f"Saved {out}")
    finally:
        if target is not None:
            target.dis()
        scope.dis()


if __name__ == "__main__":
    main()
