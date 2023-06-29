import argparse
import tomli as tomllib
from modified_improved_diffusion.modified_script_util import (
    add_dict_to_argparser
)
from modified_improved_diffusion.evaluation_util import (
    creating_results_folder,
    copy_toml_config,
    load_npz,
    finding_progessCSV,
    plot_csv_column
)
import modified_improved_diffusion.plotting as plt
import numpy as np


def main():
    args = create_argparser().parse_args()
    args_dict = vars(args)
    result_folder = creating_results_folder(args.npz_file)
    copy_toml_config(source=args.toml_config, target=result_folder)

    data = load_npz(npz_file=args.npz_file)

    progressCSV_path = finding_progessCSV(args.npz_file)
    plot_csv_column(progressCSV_path, "loss", result_folder, *args_dict)
    plot_csv_column(progressCSV_path, "mse", result_folder, *args_dict)
    plot_csv_column(progressCSV_path, "vb", result_folder, *args_dict)
    


    


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

