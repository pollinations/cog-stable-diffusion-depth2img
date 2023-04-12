MODEL_ID = "stabilityai/stable-diffusion-2-depth"
MODEL_CACHE = "diffusers-cache"



def convert_to_depth(init_image):
    depth = np.array(init_image.convert("L"))
    depth = depth.astype(np.float32) / 255.0
    depth = depth[None, None]
    depth = torch.from_numpy(depth)
    # remove first dimension
    depth = depth.squeeze(0)
    return depth

import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline, DDIMScheduler
import numpy as np
import diffusers


# load images from folder ./depth

import os
import glob

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
    revision='fp16',
).to("cuda")

print("model loaded", pipe.scheduler.compatibles)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


# # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# init_image = Image.open("depthmap.gif")

# # only first frame of animated gif
# init_image = init_image.convert("RGB")

prompt = "black and white movie scene with woman and man, railing and stairs going down. kodak vision 3 500t 5219"
n_propmt = "bad, deformed, ugly, bad anotomy"


# depth = convert_to_depth(init_image)

# print(depth.shape)
# image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=1.0, depth_map=depth, num_inference_steps=20).images[0]

# image.save("out.png")




last_image = None
for filename in sorted(glob.glob('depth/*.png')):

    # disable grad and set up for inference

    generator = torch.Generator(device="cuda").manual_seed(1)
    # generator = None
    print("image", filename)
    depth_image = Image.open(filename)
    depth_image = depth_image.convert("RGB")
    depth = convert_to_depth(depth_image)

    strength = 0.5
    if last_image is None:
        last_image = depth_image
        strength = 1.0

    print(depth.shape)
    image = pipe(prompt=prompt, generator=generator, image=last_image, negative_prompt=n_propmt, strength=strength, num_inference_steps=50, depth_map=depth).images[0]
    image.save("out/" + os.path.basename(filename))

    
    last_image = image
    last_image = last_image.convert("RGB")
    


# run ffmpeg command to create video from images
import os

os.system("ffmpeg -y -framerate 2 -i out/%*.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p out.mp4")
