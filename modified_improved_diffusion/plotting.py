import numpy as np
import matplotlib.pyplot as plt
from modified_improved_diffusion.importing import (
    preprocess, 
    electron_events, 
    muon_events, 
    calc_pseudo_rapidity, 
    calc_pT
)
from modified_improved_diffusion.evaluation_util import (
    load_npz, 
    extract_interation_from_path
)

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


def plot_hist_history(paths, component):

    num_plots = len(paths)
    num_cols = 6
    num_rows = (num_plots - 1) // num_cols + 1

    epochs = []
    for path in paths:
        epochs += [extract_interation_from_path(path)]

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
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30,6*num_rows), sharex="col", sharey="row")
    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            data = load_npz(paths[i])
            ax.hist(data[:, selected_column, 0], bins=50, range=(-10,10)) #@todo hier nehme ich wieder nur den ersten Vektor
            ax.set_title(f"epoch: {epochs[i]}")
            ax.set_xlim(-3,3)
            ax.set_label(component)
            ax.set_ylabel("events")
    
    plt.tight_layout()
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/Models/2023-06-25_13-05-34/history_{component}_"
                 f"{epochs}.pdf"))
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/Models/2023-06-25_13-05-34/history_{component}_"
                 f"{epochs}.png"))


def plot_comparison_distribution(particle_type, npz_file, component):

    npz = load_npz(npz_file)
    if particle_type=="muons":
        data = muon_events("all", False)
    elif particle_type=="electrons":
        data = electron_events("all", False)
    else:
        raise ValueError(f"No valid input for 'particle_type'. {particle_type}")
    data = preprocess(data, min_max_norm=False)
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
    ax.hist(hist_data, bins=80, histtype="step", color = "blue", label = "data", range=(min(hist_npz), max(hist_npz)))
    ax.hist(hist_npz, bins=80, histtype="step", color = "red", label = "samples")
    ax.set_xlabel(component)
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f"comparison, {component}, particle_type='muons'")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                "diffusion-main/Models/2023-06-28_21-54-32"
                 f"/comparison_{component}_clip=new_clipping.png"))

    

def plot_denoising(path, component):

    
    npz = load_npz(path)

    num_plots = np.shape(npz)[0]
    num_cols = 5
    num_rows = (num_plots - 1) // num_cols + 1

    denoising_step = []
    for step in range(np.shape(npz)[0]):
        denoising_step += [step * 500]

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
    print(npz)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30,6*num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            ax.hist(npz[i, :, selected_column, 0]) #@todo hier nehme ich wieder nur den ersten Vektor
            ax.set_title(f"denoising step: {denoising_step[i]}")
            ax.set_label(component)
            ax.set_ylabel("events")
    
    plt.tight_layout()
    timestep = "2023-06-28_21-54-32"
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/Models/{timestep}/denoising_process_{component}_"
                 f"{denoising_step}.pdf"))
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/Models/{timestep}/denoising_process_{component}_"
                 f"{denoising_step}.png"))
    



if __name__=="__main__":

    do_plot_hist_history=False
    if do_plot_hist_history:
        data = muon_events("all", shuffle=True)
        paths = []
        for epoch in ["00", "03", "06", "09", "12", "15", "18", "21", "24", "27", "30"]:
            paths += [("Models/2023-06-25_13-05-34/samples_2048_2023-06"
                    f"-25_13-05-34_{epoch}0000.npz")]
        plot_hist_history(paths, "px")
    

    do_plot_denoising=True
    if do_plot_denoising:
        plot_denoising(("/home/paulgilles/Bachelorarbeit/"
                        "modified-improved-diffusion-main/"
                        "denoising_images_Clip=.npz"), 
                        "px")


    do_plot_comparison=False
    if do_plot_comparison:
        plot_comparison_distribution("muons",
                                     ("/home/paulgilles/Bachelorarbeit/modified-"
                                      "improved-diffusion-main/Models/2023-06-28"
                                      "_21-54-32/samples_2048_2023-06-28_21-54-32"
                                      "_330000_clip=new_clipping.npz"), 
                                      "px")
