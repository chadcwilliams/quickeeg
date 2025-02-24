import os
import numpy as np
import mne
import matplotlib.pyplot as plt

from typing import Optional

class Preprocessing:
    """
    Preprocessing pipeline for EEG data
    """

    def __init__(
        self,
        pipeline: list[str],
        file_path: str,
        find_files_by_marker: Optional[str] = None,
        reference_channels: list[str] | str = "average",
        bp_filter_cutoffs: list[float] = [0.1, 50],
        notch_filter_freq: int = 60,
        ica_components: int = 20,
        eog_channel: list[str] = ["1L", "1R"],
        clean_markers: list[str] = ["Stimulus/s"],
        target_markers: dict[str, list[str]] = {
            "11": [f"{i}" for i in range(11, 20)],
            "21": [f"{i}" for i in range(21, 30)],
            "31": [f"{i}" for i in range(31, 40)],
        },
        epoching_times: list[float] = [-0.2, 0.8],
        baseline_times: list[float] = [-0.2, 0],
    ):
        """
        Initialize the preprocessing pipeline

        Parameters
        ----------
        pipeline: list[str]
            The steps in the preprocessing pipeline
                Options: 
                    'load_data'
                    'rereference'
                    'filter'
                    'notch_filter'
                    'ica'
                    'marker_cleaning'
                    'epoching'
                    'baseline_correction'
                    'averaging'
        file_path: str
            The path to the EEG data files
        find_files_by_marker: str
            The marker to use to find the EEG data files, it will find the file that contains this marker
        reference_channels: list[str]
            The channels to use as the new reference, or 'average' for average reference
        bp_filter_cutoffs: list[float]
            The cutoff frequencies for the bandpass filter
        notch_filter_freq: int
            The frequency to notch filter
        ica_components: int
            The number of components to use in the ICA
        eog_channel: list[str]
            The channel to use for EOG detection
        clean_markers: list[str]
            The markers to clean from the EEG data
        target_markers: dict[str, list[str]]
            The markers to keep in the EEG data. The key will become the id, and the values will be the markers to merge
        epoching_times: list[float]
            The start and end times for the epochs
        baseline_times: list[float]
            The start and end times for the baseline correction
        """

        # Parameters
        self.pipeline = pipeline
        self.file_path = file_path
        self.id = os.path.split(file_path)[-1]
        self.find_files_by_marker = find_files_by_marker
        self.reference_channels = reference_channels
        self.bp_filter_cutoffs = bp_filter_cutoffs
        self.notch_filter_freq = notch_filter_freq
        self.ica_components = ica_components
        self.eog_channel = eog_channel
        self.clean_markers = clean_markers
        self.target_markers = target_markers
        self.epoching_times = epoching_times
        self.baseline_times = baseline_times

        self.parameters = {
            "pipeline": pipeline,
            "file_path": file_path,
            "find_files_by_marker": find_files_by_marker,
            "reference_channels": reference_channels,
            "bp_filter_cutoffs": bp_filter_cutoffs,
            "notch_filter_freq": notch_filter_freq,
            "ica_components": ica_components,
            "eog_channel": eog_channel,
            "clean_markers": clean_markers,
            "target_markers": target_markers,
            "epoching_times": epoching_times,
            "baseline_times": baseline_times,
        }

        # Pipeline
        self.pipeline_functions = {
            "load_data": [
                self.load_data,
                {"file_path": file_path, "find_files_by_marker": find_files_by_marker},
            ],
            "rereference": [self.apply_rereference, reference_channels],
            "filter": [self.apply_filter, bp_filter_cutoffs],
            "notch_filter": [self.apply_notch_filter, notch_filter_freq],
            "ica": [
                self.apply_ica,
                {"n_components": ica_components, "eog_channel": eog_channel},
            ],
            "marker_cleaning": [
                self.apply_marker_cleaning,
                {"clean_markers": clean_markers, "target_markers": target_markers},
            ],
            "epoching": [self.apply_epoching, epoching_times],
            "baseline_correction": [self.apply_baseline_correction, baseline_times],
            "averaging": [self.apply_averaging, None],
        }

    def load_data(self, file_path: str, find_files_by_marker: Optional[str] = None):
        """
        Load the EEG data from the provided file path

        Parameters
        ----------
        file_path: str
            The path to the EEG data files
        find_files_by_marker: str
            The marker to use to find the EEG data files, it will find the file that contains this marker
        """

        files = os.listdir(file_path)
        self.determine_data(files, find_files_by_marker)
        self.raw = mne.io.read_raw_brainvision(
            os.path.join(file_path, self.vhdr_file), preload=True
        )
        self.events, self.event_id = mne.events_from_annotations(self.raw)
        self.sfreq = self.raw.info["sfreq"]

    def determine_data(self, files: list, find_files_by_marker: Optional[str] = None):
        """
        Determine the EEG data files from the provided list of files

        Parameters
        ----------
        files: list
            The list of files in the provided file path
        """

        vmrk_files = [file for file in files if file.endswith(".vmrk")]

        correct_file = []
        for vmrk_file in vmrk_files:
            with open(os.path.join(self.file_path, vmrk_file), "r") as f:
                lines = f.readlines()
                if find_files_by_marker is not None:
                    if (
                        np.sum([True for line in lines if find_files_by_marker in line])
                        > 0
                    ):
                        correct_file.append(True)
                    else:
                        correct_file.append(False)
                else:
                    correct_file.append(True)

        if np.sum(correct_file) == 0:
            raise ValueError("No stimulation EEG data found in the provided files")

        if np.sum(correct_file) > 1:
            raise ValueError(
                "Multiple stimulation EEG data found in the provided files, you should specify a marker unique to the desired files"
            )

        self.vmrk_file = vmrk_files[np.argmax(correct_file)]
        self.vhdr_file = vmrk_file.replace(".vmrk", ".vhdr")
        self.eeg_file = vmrk_file.replace(".vmrk", ".eeg")

    def apply_filter(self, cutoffs: list[float]):
        """
        Apply a bandpass filter to the EEG data

        Parameters
        ----------
        cutoffs: list[float]
            The cutoff frequencies for the bandpass filter
        """

        self.raw.filter(cutoffs[0], cutoffs[1])

    def apply_notch_filter(self, freq: float):
        """
        Apply a notch filter to the EEG data

        Parameters
        ----------
        freq: float
            The frequency to notch filter
        """

        self.raw.notch_filter(freq)

    def apply_rereference(self, ref_channels: list | str):
        """
        Apply a new reference to the EEG data

        Parameters
        ----------
        ref_channels: list | str
            The channels to use as the new reference, or 'average' for average reference
        """

        self.raw.set_eeg_reference(ref_channels)

    def apply_ica(self, n_components: int, eog_channel: list):
        """
        Apply ICA to the EEG data

        Parameters
        ----------
        n_components: int
            The number of components to use in the ICA
        eog_channel: list
            The channel to use for EOG detection
        """

        ica = mne.preprocessing.ICA(n_components=n_components)
        ica.fit(self.raw)
        eog_indices, eog_scores = ica.find_bads_eog(self.raw, ch_name=eog_channel)
        ica.exclude = eog_indices
        ica.apply(self.raw)

    def apply_marker_cleaning(self, clean_markers: list, target_markers=list):
        """
        Apply marker cleaning to the EEG data

        Parameters
        ----------
        clean_markers: str
            The notations to remove the EEG data (usually a prefix)
        target_markers: list
            The markers to keep in the EEG data
        """

        for marker_to_clean in clean_markers:
            if marker_to_clean is not None:
                self.event_id = {
                    str(key).replace(marker_to_clean, ""): self.event_id[key]
                    for key in self.event_id
                }

        marker_range = [target_markers[key] for key in target_markers]
        marker_range = [item for sublist in marker_range for item in sublist]
        marker_exclusions = [
            self.event_id[key] for key in self.event_id if key not in marker_range
        ]
        self.event_id = {
            key: self.event_id[key] for key in self.event_id if key in marker_range
        }
        self.events = np.array(
            [event for event in self.events if event[2] not in marker_exclusions]
        )

        for marker_id in target_markers:
            event_codes = [
                self.event_id[key]
                for key in self.event_id
                if key in target_markers[marker_id]
            ]
            self.events = mne.merge_events(
                self.events, event_codes, int(marker_id), replace_events=True
            )
        self.event_id = {key: int(key) for key in target_markers}

    def apply_epoching(self, times: list[float]):
        """
        Apply epoching to the EEG data

        Parameters
        ----------
        times: list[float]
            The start and end times for the epochs
        """

        self.epochs = mne.Epochs(
            self.raw,
            self.events,
            self.event_id,
            times[0],
            times[1],
            baseline=None,
            event_repeated="merge",
            preload=True,
        )

    def apply_baseline_correction(self, times: list[float]):
        """
        Apply baseline correction to the EEG data

        Parameters
        ----------
        times: list[float]
            The start and end times for the baseline correction
        """

        self.epochs.apply_baseline((times[0], times[1]))

    def apply_averaging(self):
        """
        Apply averaging to the EEG data

        Parameters
        ----------
        None
        """

        if not self.target_markers:
            raise ValueError("No target markers provided for averaging")

        # TODO: Extracting epoch dataframes is really slow, find a way to speed this up
        self.erp = {}
        for key in self.target_markers:
            print(f"Loading condition {key} data for averaging...")
            self.erp[key] = np.mean(self.epochs[key].get_data(), axis=0)

    def plot_erp(self, electrode_index: list[int], save_plot: bool = False):
        """
        Plot the ERP data

        Parameters
        ----------
        electrode_index: int
            The index of the electrode to plot
        """

        self.erp_plot_filenames = []
        for e in electrode_index:
            for key in self.target_markers:
                plt.plot(self.erp[key][e], alpha=0.5, label=key)
            plt.xlabel("Time (ms)")
            start, end = [time * 1000 for time in self.epoching_times]
            xl = np.arange(start, end + 1, 200)
            xs = np.linspace(0, self.sfreq, len(xl))
            plt.xticks(xs, xl)
            plt.ylabel("Voltage")
            plt.title(f"ERP Data, Electrode {self.raw.ch_names[e]}")
            plt.legend()

            # Save the plot
            if save_plot:
                self.erp_plot_filenames.append(
                    os.path.join(
                        "quickeeg",
                        "plots",
                        f"{self.id}_e{self.raw.ch_names[e]}_erp.png",
                    )
                )
                plt.savefig(self.erp_plot_filenames[-1])
            else:
                plt.show()
            plt.close()

    def plot_eeg(self):
        """
        Plot the EEG data, mostly for debugging purposes

        Parameters
        ----------
        None
        """

        data = self.raw.get_data()

        for i in range(5, 10):
            plt.plot(data[i][2048 : 2048 * 3], alpha=0.5, label=self.raw.ch_names[i])
        plt.xlabel("Time")
        plt.ylabel("Voltage")
        plt.title("EEG Data")
        plt.legend()
        plt.show()

    def process(self):
        """
        Process the EEG data according to the pipeline

        Parameters
        ----------
        None
        """

        for step in self.pipeline:
            function = self.pipeline_functions[step][0]
            params = self.pipeline_functions[step][1]
            if params is None:
                function()
            elif type(params) == dict:
                function(**params)
            else:
                function(params)
