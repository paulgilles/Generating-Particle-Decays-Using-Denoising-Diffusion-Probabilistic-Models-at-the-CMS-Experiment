import argparse
import tomli as tomllib
import os
from modified_improved_diffusion.modified_script_util import (
    add_dict_to_argparser
)
from modified_improved_diffusion.evaluation_util import (
    copy_toml_config,
    load_npz,
    finding_progessCSV,
)
import modified_improved_diffusion.plotting as plt
import numpy as np


def main():
    args = create_argparser().parse_args()
    args_dict = vars(args)
    sample_folder, npz_filename = os.path.split(args.npz_file)
    copy_toml_config(source=args.toml_config, target=sample_folder)


    if args.plot_losses:
        progressCSV_path = finding_progessCSV(args.npz_file)
        print("hey; ", progressCSV_path)
        for loss_type in ["loss", "mse", "vb"]:
            plt.plot_csv_column(progressCSV_path, loss_type, sample_folder, ignore_error=True)


    if args.plot_denoising:
        files = os.listdir(sample_folder)
        selected_file = None
        for file in files:
            if file.startswith("denoising_"):
                selected_file = os.path.join(sample_folder, file)
                break
        print(selected_file)
        for component in ["E", "px", "py", "pz"]:
            plt.plot_denoising(selected_file, component, 
                               sample_folder,
                               num_cols=args.denoising_num_cols,
                               particle_type=args.particle_type)


    if args.plot_comparison:
        for component in ["E", "px", "py", "pz"]:
            plt.plot_comparison_distribution(args.particle_type,
                                             args.npz_file, 
                                             sample_folder,
                                             component, 
                                             create_pdf=args.create_pdf)


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

    all_argmuents = {**given_arguments, **toml_config}
    second_parser = argparse.ArgumentParser()
    add_dict_to_argparser(second_parser, all_argmuents)
    return second_parser

if __name__ == "__main__":
    main()

