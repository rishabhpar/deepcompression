import numpy as np
from tqdm import tqdm
from PIL import Image
import os
from pathlib import Path

# Mean and standard deviation used for PyTorch models
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

OUT_DIR = 'test_deployment_preproc'
Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
        # For PyTorch models ONLY: normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])

        filename = filename.replace('.png', '')
        np.save(f'{OUT_DIR}/{filename}', input_image)

