import numpy as np
import matplotlib.pyplot as plt
import os
from modified_improved_diffusion.importing import (
    preprocess, 
    electron_events, 
    muon_events, 
    calc_pseudo_rapidity, 
    calc_pT,
    calc_m,
    calc_Z_vector,
    postprocess,
    calc_KL
)
from modified_improved_diffusion.evaluation_util import ( 
    extract_interation_from_filename,
    load_npz,
    get_equally_spaced_values
)
import modified_improved_diffusion.gaussian_diffusion as gd

def plot_4x4_grid_hist(data):
    anzahl_histogramme = data.shape[2]
    zeilen = 2
    spalten = 2
    slicing_index = 1

    fig = plt.figure(figsize=(10, 8))

    for i in range(anzahl_histogramme):
        ax = fig.add_subplot(zeilen, spalten, i+1)
        hist_daten = data[:, slicing_index, i].flatten()
        ax.hist(hist_daten, bins=50, color='blue', alpha=0.7)
        ax.set_xlabel('Werte')
        ax.set_ylabel('Häufigkeit')
        ax.set_title(f'Histogramm {i+1}')

    # Zweiter Subplot ohne Achsen erstellen und Titel setzen
    ax2 = fig.add_subplot(111, frame_on=False)
    ax2.set_title('Titel des zweiten Subplots')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    plt.savefig('/home/paulgilles/Bachelorarbeit/improved-diffusion-main/plots/plots.png')
    plt.show()



def plot_1_hist(data, component="px", preprocessing=None):

    if preprocessing == None:
        pass
    else:
        if preprocessing=="min_max_norm":
            data = preprocess(data, min_max_norm=True)
        elif preprocessing=="mean_norm":
            data = preprocess(data, min_max_norm=False)
        else: 
            raise NotImplementedError(f"'{preprocessing}' is not implemented.")

    if component=="E":
        hist_data = data[:, :, 0].flatten()
    elif component=="px":
        hist_data = data[:, :, 1].flatten()
    elif component=="py":
        hist_data = data[:, :, 2].flatten()
    elif component=="pz":
        hist_data = data[:, :, 3].flatten()
    elif component=="pT":
        hist_data = calc_pT(data).flatten()
    elif component=="ps-rapidity":
        hist_data = calc_pseudo_rapidity(data)[:, 0].flatten()
    else: 
        raise NotImplementedError(f"‘{component}‘ is not implemented.")
    #@todo hier wird immer nur der erste Vektor genutzt. Das muss angepasst werden

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.hist(hist_data, bins=50, histtype="step")
    ax.set_xlabel(component)
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f"{component}, preprocessing={preprocessing}, particle_type='muons'")

    plt.tight_layout()
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/plots/1_hist_{component}_{preprocessing}.png"))


def plot_hist_history(path_dict, component, result_folder, create_pdf=False,
                      hist_history_num_cols=5, hist_history_iterations="all"):

    path_dict = return_files_corresponding_to_hist_history_iterations(
        path_dict, hist_history_iterations
    )
    paths, iterations = [], []
    for iteration, path in zip(path_dict.keys(), path_dict.values()):
        iterations += [int(iteration)]
        paths += [path]

    num_plots = len(paths)
    num_cols = hist_history_num_cols
    num_rows = (num_plots - 1) // num_cols + 1

    if component=="E":
        selected_column = 0
    elif component=="px":
        selected_column = 1
    elif component=="py":
        selected_column = 2
    elif component=="pz":
        selected_column = 3
    else: 
        raise NotImplementedError(f"‘{component}‘ is not implemented.")
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30,6*num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            print(paths[i])
            data = load_npz(paths[i])
            ax.hist(data[:, selected_column, 0], bins=50) #@todo hier nehme ich wieder nur den ersten Vektor
            ax.set_title(f"epoch: {iterations[i]}")
            ax.set_label(component)
            ax.set_ylabel("events")
    
    plt.tight_layout()
    file_name = f"plot_hist_history_component={component}"
    target = os.path.join(result_folder, file_name)
    plt.savefig(target + ".png")
    if create_pdf:
        plt.savefig(target + ".pdf")


def plot_comparison_distribution(result_folder, 
                                 component, create_pdf=False, 
                                 npz_file=None,
                                 particle_type="muons"):
    npz = load_npz(npz_file)
    if particle_type=="muons":
        data = muon_events("all", False)
    elif particle_type=="electrons":
        data = electron_events("all", False)
    else:
        raise ValueError(f"No valid input for 'particle_type'. {particle_type}")
    data = preprocess(data, min_max_norm=True)
    np.random.shuffle(data)
    data = data[:len(npz)]
    print(np.shape(data), np.shape(npz))
    npz = np.transpose(npz, (0,2,1))
    print(np.shape(npz))

    if component=="E":
        hist_data = data[:, :, 0].flatten()
        hist_npz = npz[:, :, 0].flatten()
    elif component=="px":
        hist_data = data[:, :, 1].flatten()
        hist_npz = npz[:, :, 1].flatten()
    elif component=="py":
        hist_data = data[:, :, 2].flatten()
        hist_npz = npz[:, :, 2].flatten()
    elif component=="pz":
        hist_data = data[:, :, 3].flatten()
        hist_npz = npz[:, :, 3].flatten()
    elif component=="pT":
        hist_data = calc_pT(data).flatten()
        hist_npz = calc_pT(npz).flatten()
    elif component=="ps-rapidity":
        hist_data = calc_pseudo_rapidity(data)[:, 0].flatten()
        hist_npz = calc_pseudo_rapidity(npz)[:, 0].flatten()
    else: 
        raise NotImplementedError(f"‘{component}‘ is not implemented.")
    #@todo hier wird immer nur der erste Vektor genutzt. Das muss angepasst werden

    print(np.shape(hist_data), np.shape(hist_npz))

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.hist(hist_data, bins=130, histtype="step", color = "blue", label = "data", range=(min(hist_npz), max(hist_npz)))
    ax.hist(hist_npz, bins=130, histtype="step", color = "red", label = "samples")
    ax.set_xlabel(component)
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f"comparison, {component}, particle_type='muons'")
    ax.legend()
    
    plt.tight_layout()
    file_name = f"plot_comparison_component={component}"
    target = os.path.join(result_folder, file_name)
    plt.savefig(target + ".png")
    if create_pdf:
        plt.savefig(target + ".pdf")


def plot_denoising(path, component, result_folder, denoising_steps=None,
                   create_pdf=False, denoising_num_cols=5, particle_type=None):
    npz = load_npz(path)
    if particle_type is not None:
        if particle_type=="muons":
            data = muon_events("all", False)
        elif particle_type=="electrons":
            data = electron_events("all", False)
        else:
            raise ValueError(f"No valid input for 'particle_type'. {particle_type}")
        data = preprocess(data, min_max_norm=True)
        np.random.shuffle(data)
        data = data[:np.shape(npz)[1]]

    num_plots = np.shape(npz)[0]
    num_cols = denoising_num_cols
    num_rows = (num_plots - 1) // num_cols + 1

    if component=="E":
        selected_column = 0
    elif component=="px":
        selected_column = 1
    elif component=="py":
        selected_column = 2
    elif component=="pz":
        selected_column = 3
    else: 
        raise NotImplementedError(f"‘{component}‘ is not implemented.")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30,6*num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            min_npz, max_npz = min(npz[i, :, 0, selected_column]), max(npz[i, :, 0, selected_column])
            ax.hist(npz[i, :, 0, selected_column], bins=50, histtype="step",
                    color = "red", label = "samples") #@todo hier nehme ich wieder nur den ersten Vektor
            if particle_type is not None:
                ax.hist(data[:, 0, selected_column].flatten(), 
                        bins=50, histtype="step", color="blue", 
                        label="data", range=(min_npz, max_npz))
            ax.set_title(f"denoising step: {denoising_steps[i]}")
            ax.set_label(component)
            ax.set_ylabel("events")
            ax.legend()
    
    plt.tight_layout()
    file_name = f"plot_denoising_component={component}"
    target = os.path.join(result_folder, file_name)
    plt.savefig(target + ".png")
    if create_pdf:
        plt.savefig(target + ".pdf")
    

def plot_csv_column(csv_file, column_name, result_folder, ignore_error, 
                    create_pdf=False):
    """
    Teilweise ChatGPT
    """
    print(csv_file, column_name, result_folder, ignore_error)
    data = np.genfromtxt(csv_file, delimiter=',', names=True)
    if column_name not in data.dtype.names:
        if not ignore_error:
            raise ValueError("Die angegebene Spalte existiert nicht.")
    else: 
        values = data[column_name]
        steps = data['step']

        fig, ax = plt.subplots(1,1)
        ax.plot(steps, values)
        if column_name in ["loss", "mse", "vb"]:
            ax.set_yscale("log")
        ax.set_xlabel('Step')
        ax.set_ylabel(column_name)
        ax.set_title('Plot von Spalte: ' + column_name)
        filename = result_folder + f"/{column_name}VSSteps"
        plt.savefig(filename + ".png")
        if create_pdf:
            plt.savefig(filename + ".pdf")

        print(f"Der Plot wurden erfolgreich in gespeichert unter '{filename}'.")


def plot_Z_analyse(result_folder, create_pdf=False, particle_type="muons", 
                   npz_file=None, min_max_norm=True, count_number=10,
                   intervals=None):
    
    npz = load_npz(npz_file)
    if particle_type=="muons":
        data = muon_events("all", False)
    elif particle_type=="electrons":
        data = electron_events("all", False)
    else:
        raise ValueError(f"No valid input for 'particle_type'. {particle_type}")
    _, minimum, maximum = preprocess(data, min_max_norm, count_number,intervals,
                                     full_output=True)
    np.random.shuffle(data)
    data = data[:len(npz)]
    print(np.shape(data), np.shape(npz))
    npz = np.transpose(npz, (0,2,1))
    print(np.shape(npz))
    npz = postprocess(npz, min_max_norm, minimum, maximum)

    num_plots = 7
    num_cols = 7
    num_rows = (num_plots - 1) // num_cols + 1

    z_data = calc_Z_vector(data)
    z_npz = calc_Z_vector(npz)
    measurands = ["m", "E", "eta", "px", "py", "pz","pT"]

    def return_correct_measurand(z, i, return_limit=False):
        if i==0:
            if return_limit: return [50,130]
            return calc_m(z)
        elif i==1:
            if return_limit: return None
            return z[:,0]
        elif i==2:
            if return_limit: return None
            return calc_pseudo_rapidity(z)
        elif 2 < i < 5:
            if return_limit: return [-100,100]
            return z[:,i-2]
        elif i == 5:
            if return_limit: return None
            return z[:,i-2]
        elif i == 6:
            if return_limit: return None
            return calc_pT(z)
        else: 
            raise ValueError(f"index is out of range: i={i}")

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols,5*num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            min_npz = min(return_correct_measurand(z_npz, i))
            max_npz = max(return_correct_measurand(z_npz, i))
            xlimit = return_correct_measurand(z_data, i, return_limit=True)
            if xlimit is not None:
                ax.set_xlim(*xlimit)
                range=(xlimit[0], xlimit[1])
            else:
                range=(min_npz, max_npz)
            KL_div = calc_KL(return_correct_measurand(z_npz, i), return_correct_measurand(z_data,i))
            ax.hist(return_correct_measurand(z_npz, i), bins=50, histtype="step",
                    label = f"samples \nKL={round(KL_div,4)}", range=range)
            if particle_type is not None:
                ax.hist(return_correct_measurand(z_data,i), 
                        bins=50, histtype="step", 
                        label="data", range=range)
            ax.set_xlabel(measurands[i])
            ax.set_ylabel("events")
            ax.legend()
    
    plt.tight_layout()
    file_name = f"plot_Z_analyse"
    target = os.path.join(result_folder, file_name)
    plt.savefig(target + ".png")
    if create_pdf:
        plt.savefig(target + ".pdf")
    print(f"Plot saved as '{target}.pdf/.png'")
    

def plot_schedule(noise_schedule, steps, result_folder, create_pdf=False):
    if type(noise_schedule) == str:
        betas = [gd.get_named_beta_schedule(noise_schedule, steps)]
    else:
        betas = []
        for schedule in noise_schedule:
            betas += [gd.get_named_beta_schedule(schedule, steps)]
    alphas_cumprod = []
    for beta in betas:
        beta = np.array(beta, dtype=np.float64)
        alphas = 1.0 - beta
        alphas_cumprod += [np.cumprod(alphas, axis=0)]
    x = np.arange(steps) / steps

    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    for y in alphas_cumprod:
        ax.plot(x, y)
    ax.set_xlabel("timestep / max_timestep")
    ax.set_ylabel("alphas_cumprod")
    ax.legend()
    plt.tight_layout()
    file_name = f"plot_schedule_{noise_schedule}"
    target = os.path.join(result_folder, file_name)
    plt.savefig(target + ".png")
    if create_pdf:
        plt.savefig(target + ".pdf")
    print(f"Plot saved as '{target}.pdf/.png'")   




def return_files_corresponding_to_hist_history_iterations(path_dict,
                                                          hist_history_iterations):
    """Filters the in path_dict given files using the through
       hist_history_iterations given criteria.

    Args:
        path_dict (dict): dict which has iterations as keys and paths as values
        hist_history_iterations (list, int or str): if set to "all", all
            files will be analysed. If it is a list, all files corresponding
            to the given indexes will be used. If it is an int, so many 
            files will be analysed.

    Returns:
        dict: path_dir but with only the important keys and values
    """
    if hist_history_iterations=="all":
        return path_dict
    else: 
        if type(hist_history_iterations)==list:
            for key in path_dict.keys():
                if int(key) in hist_history_iterations:
                    pass
                else:
                    del path_dict[key]
            return path_dict      
        elif type(hist_history_iterations)==int:
            if len(path_dict) < hist_history_iterations:
                raise ValueError(("There are not enough models to display:"
                                 f"maximal: {len(path_dict)}. given: "
                                 f"{hist_history_iterations}"))
            else:
                keys_as_int = [int(key) for key in path_dict.keys()]
                iterations = get_equally_spaced_values(keys_as_int, 
                                                       hist_history_iterations)
                for iteration in iterations:
                    del path_dict[f"{iteration}"]
                return path_dict




if __name__=="__main__":
    plots_folder = "/home/paulgilles/Bachelorarbeit/modified-improved-diffusion-main/plots"
    plot_schedule(["linear", "cosine", "fermi"], 4000, plots_folder)