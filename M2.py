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
from onnxruntime.quantization import quantize_dynamic, QuantType
import glob

if __name__ == '__main__':

    ONNX_SAVE_DIR = "onnx_models"
    PT_SAVE_DIR = "pt_models_m1"
    Path(ONNX_SAVE_DIR).mkdir(exist_ok=True, parents=True)
    Path(PT_SAVE_DIR).mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch_size = 1

    ONNX_MODEL_DIR_M1 = "onnx_models_m1"
    ONNX_MODEL_DIR_M2 = "onnx_models"

    for onnx_model_name in glob.glob(f"{ONNX_MODEL_DIR_M1}/*.onnx"): # Iterate over all .onnx files
        cleaned_file_name = onnx_model_name.replace('.onnx', '').replace(f"{ONNX_MODEL_DIR_M1}", '').replace("\\","")
        quantized_model  = quantize_dynamic(onnx_model_name, f'{ONNX_MODEL_DIR_M2}/{cleaned_file_name}_quantized.onnx', weight_type=QuantType.QUInt8)