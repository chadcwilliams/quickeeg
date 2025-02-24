import numpy as np
import mne


def simulate_sine(
    fs: int = 1024,
    duration: float = 10.0,
    frequencies: list[float] = [40],
    plot: bool = False,
) -> mne.io.RawArray:
    """
    Simulates EEG data for testing purposes

    Parameters
    ----------
    fs : int
        The sampling frequency of the EEG data
    f : float
        The frequency of the sine wave
    duration : float
        The duration of the EEG data in seconds
    frequencies : list
        The frequencies to simulate

    Returns
    -------
    raw : mne.io.RawArray
        The simulated EEG data
    """

    # Create sine wave data at 40 Hz
    time = np.arange(0, duration, 1 / fs)
    EEG = np.array([np.sin(2 * np.pi * f_i * time) for f_i in frequencies])

    # Convert into MNE object
    channels = [str(f) for f in frequencies]
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(EEG, info)

    # Add events as annotations where the event is 1 every second
    events = np.array([[i * fs, 0, 1] for i in range(int(duration))])
    annotations = mne.Annotations(onset=events[:, 0] / fs, duration=[0] * len(events), description=[str(e) for e in events[:, 2]])
    raw.set_annotations(annotations)

    if plot:
        raw.plot(
            n_channels=len(frequencies),
            duration=duration,
            scalings={"eeg": 1.1, "stim": 1},
        )

    return raw
