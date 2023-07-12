import os
import numpy as np
import shutil
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import subprocess



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


def creating_samples_loop_folder(model_folder):
    """
    Creates a new folder named 'all_samples'.
    """
    target_dir = os.path.join(model_folder, "all_samples")
    if os.path.exists(target_dir):
        raise UserWarning("All samples does already exits.")
    os.mkdir(target_dir)
    return target_dir


def create_samples_from_models_in_dict(paths_dict, npz_file, model_type):
    model_folder = os.path.dirname(os.path.dirname(npz_file))
    all_samples_folder = creating_samples_loop_folder(model_folder)
    index = 0
    sample_folder = os.path.dirname(npz_file)
    for value in paths_dict.values():
        if index == len(paths_dict)-1:
            copy_sample_to_folder(sample_folder, all_samples_folder, model_type)
        else:
            for file in os.listdir(sample_folder):
                if ".toml" in file and not "evaluate" in file:
                    toml_config_sampling = os.path.join(sample_folder, file)
                    break
            command = ['python', f'scripts/sampling.py', "--toml_config", 
                       f"sampling_in_loop//{toml_config_sampling}", 
                       "--model_path", value]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.wait()
            output, error = process.communicate()
            print("\n##############################\n", error.decode(), "\n\n")
        index += 1
    return all_samples_folder

def copy_sample_to_folder(sample_dir, folder, model_type):
    for file in os.listdir(sample_dir):
        if "sample_num" in file:
            sample = os.path.join(sample_dir, file)
            steps = file.split("=")[-1].split(".")[0]
            break
    folder += "/"
    if model_type == "ema":
        folder += "ema_0.9999_"
    elif model_type == "model":
        folder += "model"
    elif model_type == "opt":
        folder += "opt"
    folder += steps + "_sample.npz"
    shutil.copyfile(sample, folder)


def check_if_samples_exists(npz_file):
    all_samples_folder = os.path.join(os.path.dirname(os.path.dirname(npz_file)),
                                      "all_samples")
    return os.path.exists(all_samples_folder)


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


def extract_interation_from_filename(filename):
    try:
        interation = int(filename[3:-3])
    except:
        try:
            interation = int(filename[5:-3])
        except:
            try:
                interation = int(filename[11:-3])
            except:
                raise ValueError(("Not able to extract iteration from"
                                 f" filename='{filename}'."))
    return interation


def get_equally_spaced_values(liste, x):
    indices = np.linspace(0,len(liste)-1, x)
    final_liste = []
    for index in indices:
        final_liste += [liste[math.floor(index)]]
    final_liste[-1]=liste[-1]
    return final_liste








def all_models_in_dict(npz_file, model_type="ema"):
    """Searches for models in all_samples and returns a dict with them.

    Args:
        npz_file (str): path to npz file which is evaluated
        model_type (str, optional): Can be "ema", "model" or "opt". Defaults to "ema".
    """    
    npz_dir = os.path.dirname(npz_file)
    if model_type!="ema" and model_type!="model" and model_type!="opt":
        raise ValueError("'model_type' must be 'ema', 'model' or 'opt'.")
    files = os.listdir(os.path.dirname(npz_dir))
    selected_files = {}
    for index, file in enumerate(files):
        if file.startswith(model_type):
            iteration = extract_interation_from_filename(file)
            selected_files[f"{iteration}"] = os.path.join(os.path.dirname(npz_dir), file)
    return selected_files



if __name__=="__main__":
    input = "/home/paulgilles/Bachelorarbeit/modified-improved-diffusion-main/Models/2023-06-29_21-19-31/2023-06-29_21-19-43.txt"
    output = "/home/paulgilles/Bachelorarbeit/modified-improved-diffusion-main/Models/2023-06-29_21-19-31/progress.csv"
    txt_to_csv(input, output)

