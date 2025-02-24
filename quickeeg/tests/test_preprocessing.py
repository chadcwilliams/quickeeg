import mne

from quickeeg.helpers.preprocessing import Preprocessing
from quickeeg.helpers.simulate import simulate_sine


def test_preprocessing():

    # Create data
    EEG = simulate_sine(fs=1024, duration=10.0, frequencies=[10, 20, 30, 40])

    # Create pipeline
    pipeline = [
        "load_data",
        "filter",
        "notch_filter",
        "epoching",
        "baseline_correction",
        "averaging",
    ]

    params = {
        "pipeline": pipeline,
        "eeg_data": EEG,
        "bp_filter_cutoffs": [1, 50],
        "notch_filter_freq": 60,
        "target_markers": {"1": "1"},
        "epoching_times": [-0.2, 0.8],
        "baseline_times": [-0.2, 0],
    }

    preprocessing = Preprocessing(**params)
    preprocessing.process()

    # Test the pipeline
    assert preprocessing.pipeline == pipeline
    assert isinstance(preprocessing.raw, mne.io.RawArray)
    assert isinstance(preprocessing.epochs, mne.Epochs)
    assert preprocessing.raw.get_data().shape == (4, 10240)
    assert preprocessing.epochs.get_data().shape[1:] == (4, 1024 + 1)
