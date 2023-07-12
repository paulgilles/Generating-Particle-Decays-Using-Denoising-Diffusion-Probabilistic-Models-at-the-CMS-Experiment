from modified_improved_diffusion.modified_script_util import create_gaussian_diffusion
from modified_improved_diffusion.importing import electron_events, muon_events, preprocess, postprocess
import torch as th
import numpy as np
import matplotlib.pyplot as plt


def q_sample(x_start, noise, diffusion):
    return diffusion.sqrt_alphas_cumprod * x_start + diffusion.sqrt_one_minus_alphas_cumprod * noise

def create_diffusion_plot(particle_type, component, preprocessing=True):

    if particle_type=="muons":
        data = muon_events("all", False)
    elif particle_type=="electrons":
        data = electron_events("all", False)
    else:
        raise ValueError(f"No valid input for 'particle_type'. {particle_type}")

    if preprocessing:
        data = preprocess(data, min_max_norm=False)

    diffusion = create_gaussian_diffusion(
        steps=10,
        learn_sigma=True,
        sigma_small=False,
        noise_schedule="cosine",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        timestep_respacing="",
    )

    result = []
    data = data.reshape(len(data), 8)
    for x_start in data:
        noise =np.random.normal(0,1,np.shape(x_start))
        x_t_values = []
        for index in range(np.shape(x_start)[-1]):
            x_t = q_sample(x_start[index], noise=noise[index], diffusion=diffusion)
            x_t_values.append(x_t)
        result.append(x_t_values)
    result = np.array(result)

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
    #@todo Das ist noch nicht ganz sauber. Eigentlich hat das array 8 Komponenten
    #also zwei verschiedene Vierervektoren. Hier greife ich aber immer nur auf 
    #den ersten zu. Das muss noch gepüft und geändert werden

    num_plots = np.shape(result)[-1]
    num_cols = 10
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(60, 6 * num_rows), sharex="col", sharey="row")

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            ax.hist(result[:, selected_column, i], bins=50, range=(-5,5))
            ax.set_title(f"Timestep {i}")
            ax.set_xlim(-5,5)

    plt.tight_layout()
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/Models/2023-06-25_13-05-34/diffusion_{component}_"
                 f"{np.shape(result)[-1]}_{particle_type}.pdf"))
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/Models/2023-06-25_13-05-34/diffusion_{component}_"
                 f"{np.shape(result)[-1]}_{particle_type}.png"))

    print(("\nmean and std from last timestep: \n"
           f"mean = {np.mean(result[:, selected_column, -1])}, {np.std(result[:, selected_column, -1])}"))
    


def check_normalization(particle_type, min_max_norm=False):
    if particle_type=="muons":
        data = muon_events("all", False)
    elif particle_type=="electrons":
        data = electron_events("all", False)
    
    normalized, min, max = preprocess(data, min_max_norm=min_max_norm, full_output=True)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), sharey="row")
    
    component = [0,1,2,3] * 3
    labels = ["E", "px", "py", "pz"] * 3
    for i, ax in enumerate(axes.flat):
        if i < 4:
            range=(-100,100)
            if i==0: range=(0,100)
            if i==3: range=(-200,200)
            ax.hist(data[:, 0, component[i]], bins=50, range=range)
        elif i < 8: 
            range=(-1,1)
            ax.hist(normalized[:, 0, component[i]], bins=100, range=range)
        else: 
            range=(-100, 100)
            post_data = postprocess(normalized, min_max_norm, min, max)
            ax.hist(post_data[:, 0, component[i]], bins=50, range=range)
        ax.set_xlabel(labels[i])
        ax.set_ylabel("events")

    plt.tight_layout()
    plt.savefig(("/home/paulgilles/Bachelorarbeit/modified-improved-"
                 f"diffusion-main/plots/norm_"
                 f"{particle_type}_min_max_norm={min_max_norm}_post.png"))
    
    print("Plot saved.")






def main():

    do_create_diffusion_plot=False
    if do_create_diffusion_plot:
        create_diffusion_plot("muons", "pz")

    do_check_normalization=True
    if do_check_normalization:
        check_normalization("muons", min_max_norm=True)



if __name__=="__main__":
    main()