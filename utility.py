"""
Name: Jungryul Ahn
Date: February 12, 2026
Email: jrahn331@g.skku.edu
"""

import os
import tensorflow as tf
import argparse

def set_gpu(gpu, frac):
    """
    Function to specify which GPU to use

    INPUT
        gpu: string label for gpu (i.e. '0')
        gpu_frac: gpu memory fraction (i.e. 0.7 for 70% of the total memory)
    OUTPUT
        tf sess config
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    return gpu_options




