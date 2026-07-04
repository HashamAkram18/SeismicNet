"""Generate a dummy MiniSEED file for inference testing.

Usage:
    python scripts/create_dummy_mseed.py

Creates: data/dummy_waveform.mseed (3-component, 100 Hz, 3000 samples)
"""
from pathlib import Path

import numpy as np
from obspy import Trace, Stream, UTCDateTime


def create_dummy_mseed(output_path: str = "data/dummy_waveform.mseed") -> str:
    """Create a 3-component dummy MiniSEED with a simulated P-arrival.

    Returns:
        Absolute path to the created file.
    """
    npts = 3000
    sampling_rate = 100.0
    dt = 1.0 / sampling_rate

    t = np.arange(npts) * dt

    np.random.seed(42)

    p_sample = 1000
    p_time = p_sample * dt
    p_amplitude = 5.0

    envelope = np.zeros(npts)
    for i in range(p_sample, min(p_sample + 500, npts)):
        t_since_p = (i - p_sample) * dt
        envelope[i] = p_amplitude * np.exp(-t_since_p * 2.0) * np.sin(2 * np.pi * 5.0 * t_since_p)

    noise = np.random.normal(0, 0.1, npts).astype(np.float32)

    components = {}
    for chan in ["BHZ", "BHN", "BHE"]:
        signal = envelope.astype(np.float32) + noise
        if chan == "BHZ":
            signal *= 1.0
        elif chan == "BHN":
            signal *= 0.7
            signal += np.sin(2 * np.pi * 8.0 * t).astype(np.float32) * 0.3
        else:
            signal *= 0.5
            signal += np.sin(2 * np.pi * 12.0 * t).astype(np.float32) * 0.2
        components[chan] = signal

    traces = []
    for i, (chan, data) in enumerate(components.items()):
        header = {
            "network": "SY",
            "station": "TST",
            "location": "00",
            "channel": chan,
            "starttime": UTCDateTime(2024, 1, 1, 0, 0, 0),
            "sampling_rate": sampling_rate,
            "npts": npts,
        }
        tr = Trace(data=data, header=header)
        traces.append(tr)

    stream = Stream(traces)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    stream.write(str(out), format="MSEED")
    return out.resolve()


if __name__ == "__main__":
    path = create_dummy_mseed()
    print(f"Created dummy MiniSEED: {path}")
    print(f"  3 components: BHZ, BHN, BHE")
    print(f"  3000 samples @ 100 Hz (30 seconds)")
    print(f"  P-arrival at sample 1000")
