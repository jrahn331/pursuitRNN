#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Name: Robert Kim
# Date: October 11, 2019
# Email: rkim@salk.edu
# Description: Contains several general-purpose utility functions

import os
import tensorflow as tf
import argparse

def set_gpu(gpu, frac):
    """
    Function to specify which GPU to use

    INPUT
        gpu: string label for gpu (i.e. '0')
        frac: gpu memory fraction (i.e. 0.3 for 30% of the total memory)
    OUTPUT
        tf sess config
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
    return gpu_options




