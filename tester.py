import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import time
import argparse
import sys

# TODO: add argument parser
parser = argparse.ArgumentParser()
# TODO: add one argument for selecting PyTorch or TensorFlow option of the code
parser.add_argument("--fraction", help="[0.05, 0.25, 0.5, 0.75, 0.9]", type=float)
# TODO: add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument("--epoch", help="[0, 3, 5]", type=int)
# TODO: Modify the rest of the code to use those arguments correspondingly
args = parser.parse_args()

onnx_model_name = f'onnx_models/model_epoch_{args.epoch}_frac_{args.fraction}.onnx'

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation used for PyTorch models
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_correct = 0
runtime = 0

print(f'-----------RUNNING {onnx_model_name}-----------')

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("test_deployment", filename)).resize((32, 32)) as img:
        #print("Image shape:", np.float32(img).shape)

        # For PyTorch models ONLY: normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        # For PyTorch models ONLY: Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # For PyTorch models ONLY: change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])

        # Run inference and get the prediction for the input image
        time_start = time.time()
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        runtime += time.time() - time_start

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        if pred_class in filename:
            num_correct += 1

test_acc = num_correct / len(tqdm(os.listdir("test_deployment")))
print(f"Test Accuracy: {test_acc}, Runtime: {runtime}")

log_name = onnx_model_name.replace('.onnx', '')

with open(f'logs/{log_name}_log.txt', 'a') as f:
    f.write(f"==========================\n")
    f.write(f"Model name: {onnx_model_name}\n")
    f.write(f"Test Accuracy: {test_acc}, Runtime: {runtime}\n")
    f.write(f"--------------------------\n")
