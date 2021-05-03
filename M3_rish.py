import torch
from torchinfo import summary
from mobilenet_rm_filt_pt import MobileNetv1, remove_channel
import torch.nn.utils.prune as prune
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
from pathlib import Path
import time
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_qat
import glob
import sys
import numpy as np
import re
import abc
import subprocess
import json
import argparse
from PIL import Image

def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.ANTIALIAS)
    image_data = np.asarray(image).astype(np.float32)
    image_data = image_data.transpose([2,0,1])
    mean = np.array([0.079, 0.05, 0]) + 0.406
    std = np.array([0.005, 0, 0.001]) + 0.224
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        unconcatenated_batch_data.append(preprocess_image(image_filepath, 32,32))
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

class MobilenetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
    
    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, 32, 32, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input.1': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    PT_MODEL_DIR = "m3/structural_pruned_noisy2"
    ONNX_MODEL_DIR = "m3/structural_pruned_noisy2"
    ONNX_QUANTIZED_DIR = "m3/quantized_structural_noisy0"
    calibration_dataset_path = 'test_deployment'

    for pt_model_name in glob.glob(f"{PT_MODEL_DIR}/*.pt"): # Iterate over all .pt files
        cleaned_file_name = pt_model_name.replace('.pt', '').replace(f"{PT_MODEL_DIR}", '').replace("\\","")

        model = torch.load(pt_model_name, map_location=device)
        model.to(device)
        model.cpu() # Send cleaned model to CPU for ONNX export

        torch.onnx.export(model, torch.randn(1, 3, 32, 32), f'{ONNX_MODEL_DIR}/{cleaned_file_name}.onnx', export_params=True, opset_version=12)

    for onnx_model_name in glob.glob(f"{ONNX_MODEL_DIR}/*.onnx"): # Iterate over all .onnx files
        cleaned_file_name = onnx_model_name.replace('.onnx', '').replace(f"{ONNX_MODEL_DIR}", '').replace("\\","")
        # quantize_qat(onnx_model_name,
        #             f'{ONNX_QUANTIZED_DIR}/{cleaned_file_name}_dynamic_quantized_uint8.onnx')
        # # quantize_dynamic(onnx_model_name,
        # #             f'{ONNX_QUANTIZED_DIR}/{cleaned_file_name}_dynamic_quantized_uint8.onnx',
        # #             weight_type=QuantType.QUInt8)
        quantize_static(onnx_model_name,
                    f'{ONNX_QUANTIZED_DIR}/{cleaned_file_name}_static_quantized_uint8.onnx',
                    MobilenetDataReader(calibration_dataset_path) ,
                    weight_type=QuantType.QUInt8)
        quantize_static(onnx_model_name,
                    f'{ONNX_QUANTIZED_DIR}/{cleaned_file_name}_static_quantized_int8.onnx',
                    MobilenetDataReader(calibration_dataset_path) ,
                    weight_type=QuantType.QInt8)