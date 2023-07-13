import argparse
import tomli as tomllib
import os
import numpy as np
import matplotlib.pyplot as plt
from modified_improved_diffusion.modified_script_util import (
    add_dict_to_argparser,
    copy_toml_config,
    filter_args,
    write_to_txt
)
from modified_improved_diffusion.evaluation_util import (
    load_npz,
    finding_progessCSV,
    all_models_in_dict,
    create_samples_from_models_in_dict,
    check_if_samples_exists
)
from modified_improved_diffusion.plotting import (
    plot_csv_column,
    plot_denoising,
    plot_comparison_distribution,
    plot_hist_history,
    plot_Z_analyse
)
from modified_improved_diffusion.importing import (
    calc_wasserstein_sum
)


def main():
    args = create_argparser().parse_args()
    args_dict = vars(args)
    sample_folder, npz_filename = os.path.split(args.npz_file)
    model_timestep = os.path.basename(os.path.dirname(sample_folder))
    copy_toml_config(source=args.toml_config, target=sample_folder, evaluate=True)


    if args.plot_losses:
        progressCSV_path = finding_progessCSV(args.npz_file)
        for loss_type in ["loss", "mse", "vb"]:
            plot_csv_column(progressCSV_path, loss_type, sample_folder, 
                            ignore_error=True, 
                            **filter_args(plot_csv_column, args_dict))


    if args.plot_denoising:
        files = os.listdir(sample_folder)
        selected_file = None
        for file in files:
            if file.startswith("denoising_"):
                selected_file = os.path.join(sample_folder, file)
                break
        for component in ["E", "px", "py", "pz"]:
            plot_denoising(selected_file, component, 
                               sample_folder,
                               **filter_args(plot_denoising, args_dict))  


    if args.plot_comparison:
        for component in ["E", "px", "py", "pz"]:
            print(filter_args(plot_comparison_distribution, args_dict))
            plot_comparison_distribution(sample_folder,
                                         component, 
                                         **filter_args(
                                         plot_comparison_distribution,
                                         args_dict)
                                         )


    if args.plot_hist_history:
        path_dict = all_models_in_dict(**filter_args(all_models_in_dict, args_dict))
        if not check_if_samples_exists(args.npz_file):
            create_samples_from_models_in_dict(path_dict, args.npz_file, args.model_type)
        else: 
            print("UserWarning: Using samples in 'all_samples'-folder")
        for component in ["E", "px", "py", "pz"]:
            plot_hist_history(path_dict, component, sample_folder,
                              **filter_args(plot_hist_history, args_dict))

    
    if args.plot_Z_analyse:
        with plt.style.context(args.style_label):
            plot_Z_analyse(sample_folder, **filter_args(plot_Z_analyse, args_dict))
        

    wasserstein_sum=""
    if args.calc_wasserstein_sum:
        wasserstein_sum = calc_wasserstein_sum(sample_folder, **filter_args(calc_wasserstein_sum, args_dict))
    

    write_to_txt({"wasserstein_sum": wasserstein_sum, "model_name": model_timestep}, 
                 saving_dir=os.path.dirname(sample_folder), new_line = False)



def create_argparser():

    given_arguments = dict(
        npz_file="",
        toml_config=""
    )
    first_parser = argparse.ArgumentParser()
    add_dict_to_argparser(first_parser, given_arguments)
    args = first_parser.parse_args()

    try:
        toml_config_path = args.toml_config
    except: 
        raise AttributeError(("You have to give an toml config file:"
                              "--toml_config path/to/file"))

    with open(toml_config_path, "rb") as f:
        toml_config = tomllib.load(f)

    sample_folder, npz_filename = os.path.split(args.npz_file)

    files = os.listdir(sample_folder)
    for file in files:
        if "sampling" in file and ".toml" in file:
            toml_sample_file = file
    toml_sample_path = os.path.join(sample_folder, toml_sample_file)
    with open(toml_sample_path, "rb") as f:
        toml_sample = tomllib.load(f)

    all_argmuents = {**given_arguments, **toml_config, **toml_sample}
    second_parser = argparse.ArgumentParser()
    add_dict_to_argparser(second_parser, all_argmuents)
    return second_parser


if __name__ == "__main__":
    main()

