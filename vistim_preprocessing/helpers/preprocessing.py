
import os 
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

class Preprocessing:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        files = os.listdir(file_path)
        self.determine_data(files)
        self.raw = mne.io.read_raw_brainvision(os.path.join(self.file_path, self.vhdr_file), preload=True)
        self.events, self.event_id= mne.events_from_annotations(self.raw)

        self.sfreq = self.raw.info['sfreq']

    def determine_data(self, files):
        vmrk_files = [file for file in files if file.endswith('.vmrk')]

        correct_file = []
        for vmrk_file in vmrk_files:
            with open(os.path.join(self.file_path, vmrk_file), 'r') as f:
                lines = f.readlines()
                if np.sum([True for line in lines if 's11' in line]) > 0:
                    correct_file.append(True)
                else:   
                    correct_file.append(False)
                    
        if np.sum(correct_file) == 0:
            raise ValueError('No stimulation EEG data found in the provided files')
        
        if np.sum(correct_file) > 1:
            raise ValueError('Multiple stimulation EEG data found in the provided files')
        
        self.vmrk_file = vmrk_files[np.argmax(correct_file)]
        self.vhdr_file = vmrk_file.replace('.vmrk', '.vhdr')
        self.eeg_file = vmrk_file.replace('.vmrk', '.eeg')
    
    def report_data(self):
        self.raw.info
        
    def create_mne_raw_object(self):
        info = mne.create_info(self.raw.ch_names, self.raw.info['sfreq'], ch_types='eeg')
        self.raw = mne.io.RawArray(self.raw, info=info)

    def apply_filter(self, lowcut: float, highcut: float):
        self.raw.filter(lowcut, highcut)

    def apply_notch_filter(self, freq: float):
        self.raw.notch_filter(freq)

    def apply_rereference(self, ref_channels: list):
        self.raw.set_eeg_reference(ref_channels)

    def apply_ica(self, n_components: int, eog_channel: str):
        ica = mne.preprocessing.ICA(n_components=n_components)
        ica.fit(self.raw)
        eog_indices, eog_scores = ica.find_bads_eog(self.raw, ch_name=eog_channel)
        ica.exclude = eog_indices
        ica.apply(self.raw)      

    def rename_markers(self):
        self.events = mne.find_events(self.raw)
        self.event_id = {'Stimulus': 1, 'Response': 2}  

    def apply_epoching(self, tmin: float, tmax: float):

        #Remove markers that may be problematic in the future
        marker_range = [f'{i}' for i in range(11, 41)]
        self.event_id = {str(key).replace('Stimulus/s',''): self.event_id[key] for key in self.event_id}
        marker_exclusions = [self.event_id[key] for key in self.event_id if key not in marker_range]

        self.event_id = {key: self.event_id[key] for key in self.event_id if key in marker_range}
        self.events = np.array([event for event in self.events if event[2] not in marker_exclusions])

        #Note that some duplicate markers exist at the same time, so event_repeated='merge' will change their label, effectively removing them
        #Best to find out why this occured in the first place
        self.epochs = mne.Epochs(self.raw, self.events, self.event_id, tmin, tmax, event_repeated='merge')

    def apply_baseline_correction(self):
        self.epochs.apply_baseline((None, 0))

    def apply_averaging(self):
        self.erp = self.epochs.average()

    def apply_plot_erp(self):
        self.erp.plot()

    def apply_plot_topomap(self):
        self.erp.plot_topomap()

    def apply_plot_psd(self):
        self.raw.plot_psd()
    
    def plot_eeg(self):
        data = self.raw.get_data()

        for i in range(5, 10):
            plt.plot(data[i][2048:2048*3], alpha=0.5, label=self.raw.ch_names[i])
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.title('EEG Data')
        plt.legend()
        plt.show()

    def load_and_process_eeg(self):
        self.load_data()
        #self.plot_eeg()
        #self.create_mne_raw_object()
        self.apply_filter(0.1, 50)
        #self.plot_eeg()
        self.apply_notch_filter(60)
        #self.plot_eeg()
        #self.apply_rereference(['M1', 'M2'])
        #self.plot_eeg()
        #self.apply_ica(20, ['1L', '1R'])
        self.plot_eeg()
        self.apply_epoching(-.2, .8)
        self.apply_baseline_correction()
        self.apply_averaging()
        self.apply_plot_erp()
        self.apply_plot_topomap()
        self.apply_plot_psd()


if __name__ == '__main__':
    path = 'vistim_preprocessing/data'
    subject_id = 'A_011_FU1_x'
    file_path = os.path.join(path, subject_id)

    preprocessing = Preprocessing(file_path)
    preprocessing.load_and_process_eeg()