import argparse
import os
from datetime import datetime
import shutil

from . import gaussian_diffusion as gd
from .modified_respace import SpacedDiffusion, space_timesteps
from .forwardModel import forwardNet

NUM_CLASSES = 1000 #@audit class conditioning hard coded NUM_CLASSES

def creating_models_folder():
    """
    Creates a new folder named after the date and time and sets the OPENAI_LOGDIR there.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    #base_dir = os.getcwd()
    base_dir = "/home/paulgilles/Bachelorarbeit/modified-improved-diffusion-main"
    target_dir = os.path.join(base_dir, "Models", timestamp)
    os.mkdir(target_dir)
    os.environ["OPENAI_LOGDIR"] = target_dir
    return target_dir

def copy_toml_config(source, target, sampling=False):
    target += f"/{target.split('/')[-1]}"  
    if sampling: target += "_sampling"
    target += ".toml"
    shutil.copyfile(source, target)



def model_and_diffusion_defaults(): #@note Die Nutzung dieser Funktion scheint umständlich
    """
    Defaults for training.
    """
    return dict(
        emb_layer=[2],
        structure=[16,10,8,10,16],
        dropout_layer=[0,4],
        norm_layer=[2],
        emb_nodes=16,
        dropout=0.3,
        learn_sigma=True,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=4000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=False
        )


def create_model_and_diffusion(
    *,
    emb_layer,
    structure,
    emb_nodes,
    class_cond,
    learn_sigma,
    sigma_small,
    dropout,
    dropout_layer,
    norm_layer,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm
):
    model = create_model(
        emb_layer=emb_layer,
        structure=structure,
        emb_nodes=emb_nodes,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        dropout_layer=dropout_layer,
        norm_layer=norm_layer
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    *,
    emb_layer,
    structure,
    emb_nodes,
    learn_sigma,
    class_cond,
    use_checkpoint,
    use_scale_shift_norm,
    dropout,
    dropout_layer,
    norm_layer
):

    return forwardNet(
        emb_nodes=emb_nodes,
        emb_layer=emb_layer,
        structure=structure,
        out_channels=(2 if not learn_sigma else 4),
        norm_layer=norm_layer,
        dropout=dropout,
        dropout_layer=dropout_layer,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_scale_shift_norm=use_scale_shift_norm
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
