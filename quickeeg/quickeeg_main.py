import os
from helpers.preprocessing import Preprocessing

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
                #'ica',
                'marker_cleaning',
                'epoching',
                'baseline_correction',
                'averaging']

    #Processing parameters
    target_markers = {'11': [f'{i}' for i in range(11, 20)],
                      '21': [f'{i}' for i in range(21, 30)],
                      '31': [f'{i}' for i in range(31, 40)]}
    
    #temp:
    target_markers = {'11': '11', '21': '21', '31': '31'}
    
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
    preprocessing.plot_erp(electrode_index=2, save_plot=True)
    print('Debug stop')