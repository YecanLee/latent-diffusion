"""
Latent Diffusion Model generation file with customized cfg conditions and classes labels
"""

#@title loading utils
import torch
import os
from omegaconf import OmegaConf
from tqdm import trange

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

import numpy as np 
from PIL import Image
from einops import rearrange

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
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


model = get_model()
sampler = DDIMSampler(model)
model = torch.compile(model)

classes = [i for i in range(1000)]   # define classes to be sampled here
n_samples_per_class = 50

ddim_steps = 250
ddim_eta = 0.0
scale = 1.5   # for unconditional guidance
save_path = "generated_images/"   # path to save generated images
num_classes = 1000
batch_size = 1

all_samples = list()
image_counter = 0

os.makedirs(save_path, exist_ok=True)
with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
            )
        
        for class_label in trange(num_classes):
            for _ in range(n_samples_per_class):
                xc = torch.tensor(batch_size*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=batch_size,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                for i, sample in enumerate(x_samples_ddim):
                    sample = rearrange(sample, 'c h w -> h w c')
                    sample = (sample.cpu().numpy() * 255).astype(np.uint8)
                    sample = Image.fromarray(sample)
                    sample.save(os.path.join(save_path, f"{str(image_counter).zfill(6)}.png"))
                    image_counter += 1
