"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import tomli as tomllib
import numpy as np
import torch as th
import torch.distributed as dist

from modified_improved_diffusion import dist_util, logger
from modified_improved_diffusion.modified_script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    copy_toml_config
)

def main():
    
    args = create_argparser().parse_args()
    saving_folder = ("/home/paulgilles/Bachelorarbeit/"
                     "modified-improved-diffusion-main"
                    f"/Models/{args.model_path.split('/')[-2]}")
    os.environ["OPENAI_LOGDIR"] = saving_folder
    copy_toml_config(args.toml_config, saving_folder, sampling=True)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_vectors = []
    all_labels = []
    batch_index = 0
    denoising_process_images = np.zeros((10, 2048, 2, 4)) #@audit hardcoded sample size
    while len(all_vectors) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, denoising_process_images_result = sample_fn(
            model,
            (args.batch_size, 2, 4),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            batch_index=batch_index,
            denoising_process_images=denoising_process_images
        )

        #@audit Hier wird eigentlich auch noch der Wertebereich angepasst
        sample = sample.permute(0, 2, 1) #@todo ich glaube das ist unnötig hier. 
        sample = sample.contiguous() 

        np.savez(f"denoising_images_Clip=.npz", denoising_process_images_result)


        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_vectors.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_vectors) * args.batch_size} samples")

        batch_index += 1

    arr = np.concatenate(all_vectors, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = (f"{arr.shape[0]}_{args.model_path.split('/')[-2]}"
                     f"_{args.model_path.split('_')[-1].split('.')[0]}")
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_clip={args.clip_denoised}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    debugger = False 
    if debugger: 
        given_arguments = dict(
            model_path="/home/paulgilles/Bachelorarbeit/modified-improved-diffusion-main/Models/2023-06-26_14-35-11/ema_0.9999_330000.pt",
            toml_config="/home/paulgilles/Bachelorarbeit/modified-improved-diffusion-main/config_sample/test.toml"  #@audit zum Debugging umgestellt
        )
    else: 
        given_arguments = dict(
            model_path="",
            toml_config=""
        )        
    first_parser = argparse.ArgumentParser()
    add_dict_to_argparser(first_parser, given_arguments)
    args = first_parser.parse_args()
    
    try:
        toml_config_path = args.toml_config
    except: 
        raise ValueError(("You have to give an toml config file: "
                          "‘--toml_config path/to/file‘"))

    with open(toml_config_path, "rb") as f:
        toml_config = tomllib.load(f)
    
    try: 
        path = args.model_path.split("/")[:-1]
        date = args.model_path.split("/")[-2]
        config_training = "/".join(path) + "/" + date + ".toml"
    except:
        raise ValueError(("You have to give an trained model: " 
                          "‘--model_path path/to/model‘."))

    with open(config_training, "rb") as f:
        config_training = tomllib.load(f)

    all_arguments = {**given_arguments, **toml_config, **config_training}
    all_arguments["batch_size"] = all_arguments["microbatch"]
    second_parser = argparse.ArgumentParser()
    add_dict_to_argparser(second_parser, all_arguments)
    return second_parser


if __name__ == "__main__":
    main()
