"""
check the command line to see if the torchmetrics has the correct version
there is no error right now inside the code to generate the images
"""
"""
The DDP is not implemented inside this code right now, we will use DP right now but this needs to be optimized before we release our paper.
"""

#@title loading utils
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

from ldm.models.diffusion.ddim import DDIMSampler

model = get_model()
sampler = DDIMSampler(model)
model = torch.compile(model)

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


classes = [i for i in range(1000)]   # define classes to be sampled here
n_samples_per_class = 8

ddim_steps = 250
ddim_eta = 0.0
scale = 1.5   # for unconditional guidance


all_samples = list()
image_counter = 0

with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )
        
        for class_label in tqdm(classes):
            for _ in range(80):
                # print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                # save all the generated samples indendently
                for i, sample in enumerate(x_samples_ddim):
                    sample = rearrange(sample, 'c h w -> h w c')
                    sample = (sample.cpu().numpy() * 255).astype(np.uint8)
                    sample = Image.fromarray(sample)
                    sample.save(f"images/{str(image_counter).zfill(6)}.png")
                    image_counter += 1
