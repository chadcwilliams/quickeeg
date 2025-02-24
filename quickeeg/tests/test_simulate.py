from quickeeg.helpers.simulate import simulate_sine


def test_simulate_sine():

    # Create data
    fs = 1024
    duration = 10.0
    frequencies = [10, 20, 30, 40]
    EEG = simulate_sine(fs=fs, duration=duration, frequencies=frequencies)
    assert EEG is not None
    assert EEG.get_data().shape == (len(frequencies), fs * duration)
