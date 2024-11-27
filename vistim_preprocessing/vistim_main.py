import os
from helpers.plotter import plot_brainvision

if __name__ == "__main__":
    # Plot a patient's EEG files
    data_path = 'vistim_preprocessing/data'
    subject_dir = "A_009"
    file_path = os.path.join(data_path, subject_dir)
    for filename in os.listdir(file_path):
        if filename.endswith(".vhdr"):
            vhdr_file_path = os.path.join(file_path, filename)
            plot_brainvision(vhdr_file_path)
            print('debug hold')