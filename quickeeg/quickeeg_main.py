import os
import sys
sys.dont_write_bytecode = True
import numpy as np

from helpers.preprocessing import Preprocessing
from helpers.report import Report

if __name__ == '__main__':

    ###########################################
    ############## Example usage ##############
    ###########################################

    #Subject information
    path = os.path.join('quickeeg','data')
    id = 'A_011_FU1_x'

    #Create pipeline
    pipeline = ['load_data',
                'rereference',
                'filter',
                'notch_filter',
                'ica',
                'marker_cleaning',
                'epoching',
                'baseline_correction',
                'averaging']

    #Processing parameters
    target_markers = {'11': [f'{i}' for i in range(11, 20)],
                      '21': [f'{i}' for i in range(21, 30)],
                      '31': [f'{i}' for i in range(31, 40)]}
        
    params = {'pipeline':               pipeline,
              'file_path':              os.path.join(path, id),
              'find_files_by_marker':   's11',
              'reference_channels':     'average',
              'bp_filter_cutoffs':      [0.1, 50],
              'notch_filter_freq':      60,
              'ica_components':         20,
              'eog_channel':            ['1L', '1R'],
              'target_markers':         target_markers,
              'epoching_times':         [-.2, .8],
              'baseline_times':         [-.2, 0]}

    #Run the pipeline
    preprocessing = Preprocessing(**params)
    preprocessing.process()

    electrodes=list(np.arange(0,len(preprocessing.raw.ch_names)))
    preprocessing.plot_erp(electrode_index=electrodes, save_plot=True)

    #Build the report
    reader_note = ' '.join(['This report was produced by the Quickeeg package for Vistim Labs.',
                    'This processing pipeline is an automated pipeline for EEG data processing.',
                    'Note that I would generally take a lot more care to look at the data before accepting these processing steps.',
                    'For example, this does not include the identification and interpolation of bad channels, nor the artifact rejection of bad epochs.',
                    'Both topographic interpolation and artifact rejection methods will need to be added to QuickEEG in the future.',
                    'However, this is a proof of concept rather than a true data analysis.'])
    
    custom_text = ['## Note for the reader', 
                    reader_note]
    
    report = Report(preprocessing)
    report.build_report(custom_text)