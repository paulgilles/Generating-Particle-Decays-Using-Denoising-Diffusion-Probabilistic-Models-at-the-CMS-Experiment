"""
Train a diffusion model on images.
"""

import argparse
import tomli as tomllib
import numpy as np
import os
import modified_improved_diffusion.dist_util as dist_util
import modified_improved_diffusion.logger as logger

from modified_improved_diffusion.debugging import debug_mode, debug_print
from modified_improved_diffusion.importing import load_data
from modified_improved_diffusion.modified_resample import create_named_schedule_sampler
from modified_improved_diffusion.modified_script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    creating_models_folder,
    copy_toml_config
)
from modified_improved_diffusion.modified_train_util import TrainLoop

def main():
    saving_folder = creating_models_folder()
    args = create_argparser().parse_args()
    copy_toml_config(args.toml_config, saving_folder)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        particle_type=args.particle_type,
        batch_size=args.batch_size,
        class_cond=args.class_cond,
        preprocessing=args.preprocessing,
        min_max_norm=args.min_max_norm
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():

    if debug_mode:
        given_arguments = dict(
            toml_config="modified-improved-diffusion-main/config_training/test.toml",
            particle_type="muons"
            )
    else: 
        given_arguments = dict(
        particle_type="",
        toml_config=""
        )

    first_parser = argparse.ArgumentParser()
    add_dict_to_argparser(first_parser, given_arguments)
    args = first_parser.parse_args()

    try:
        toml_config_path = args.toml_config
    except: 
        raise AttributeError("You have to give an toml config file: --toml_config path/to/file")

    with open(toml_config_path, "rb") as f:
        toml_config = tomllib.load(f)
    all_arguments = {**given_arguments, **toml_config}
    second_parser = argparse.ArgumentParser()
    add_dict_to_argparser(second_parser, all_arguments)
    return second_parser


if __name__ == "__main__":
    main()

