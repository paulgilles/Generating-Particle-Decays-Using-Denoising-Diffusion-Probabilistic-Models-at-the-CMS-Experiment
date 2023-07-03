import os
import numpy as np
import shutil
import numpy as np
import matplotlib.pyplot as plt
import csv


def copy_toml_config(source, target):
    timestamp = target.split('/')[-1].split('_')[-2:]
    target += f"/{timestamp[0]}_{timestamp[1]}.toml"
    shutil.copyfile(source, target)


def load_npz(npz_file):
    with np.load(npz_file) as file:
        samples = file["arr_0"]
    return samples


def finding_progessCSV(npz_file):
    progessCSV_path=""
    for direction in npz_file.split("/")[:-2]: 
        progessCSV_path += direction + "/"
    progessCSV_path += "progress.csv"
    return progessCSV_path



def txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as txt_file:
        lines = txt_file.readlines()
    header = ['grad_norm', 'loss', 'loss_q0', 'loss_q1', 'loss_q2', 'loss_q3',
                         'mse', 'mse_q0', 'mse_q1', 'mse_q2', 'mse_q3', 'samples',
                         'step', 'vb', 'vb_q0', 'vb_q1', 'vb_q2', 'vb_q3']
    header_small = ['grad_norm', 'loss', 'loss_q0', 'loss_q1', 'loss_q2', 'loss_q3',
                         'mse', 'mse_q0', 'mse_q1', 'mse_q2', 'mse_q3', 'samples',
                         'step']
    all_lines = []
    all_lines += [header_small]
    current_line = []
    count = 0
    for line in lines:
        line = line.strip()
        if line.startswith('|'):
            count = 0
            key, value = line.strip('|').split('|')
            key = key.strip()
            value = value.strip()
            current_line += [value]
        else:
            if count == 0:
                all_lines += [current_line]
                current_line = []
                count += 1

    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in all_lines:
            writer.writerow(row)

    print(f"Die Daten wurden erfolgreich in die CSV-Datei '{output_file}' geschrieben.")


def plot_4x4_grid_hist(original_data, samples=None, results_folder=
                       ("/home/paulgilles/Bachelorarbeit/improved-diffusion"
                        "-main/plots/plots.png"), *args):
    anzahl_histogramme = original_data.shape[2]
    zeilen = 2
    spalten = 2
    slicing_index = 1

    colors = [args.color1, args.color2]
    labels = ["original data", "generated data"]
    xlabels = ["E", "px", "py", "pz"]

    fig = plt.figure(figsize=(10, 8))

    for i, xlabel in zip(range(anzahl_histogramme), xlabels):
        ax = fig.add_subplot(zeilen, spalten, i+1)
        for data, color, label in zip([original_data, samples], 
                                              colors, labels):
            if data != None:
                hist_daten = original_data[:, slicing_index, i].flatten()
                ax.hist(hist_daten, bins=50, color=color, alpha=0.7, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Häufigkeit')
        ax.set_title(f'Histogramm {i+1}')

    plt.tight_layout()
    plt.savefig(results_folder)


def extract_interation_from_path(path):
    number = path.split("_")[-1].split(".")[0]
    interation = ""
    first_digit_found=False
    if number == "000000" or number == "0000000":
        interation = 0
    else:
        for digit in number:
            if first_digit_found:
                interation += digit
            else:
                if int(digit) != 0:
                    interation += digit
                    first_digit_found=True

    return int(interation)



        
