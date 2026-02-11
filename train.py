"""
Training script for continuous-time firing-rate RNN for smooth pursuit eye movement task

Name: Jungryul Ahn
Date: February 12, 2026
Email: jrahn331@g.skku.edu
"""

import os
import time
import argparse
import datetime
import numpy as np
import scipy.io
import tensorflow as tf

from model import FR_RNN
from model import generate_input_target_pursuit_stage1
from model import generate_input_target_pursuit_stage2
from model import generate_input_target_pursuit
from model import construct_tf
from model import loss_op

# Import utility functions
from utility import set_gpu

# ==========================================================
# Load behavioral data (.mat)
# ==========================================================
MAT_FILE_NAME = "behavior_data/eyeVelData_EV70_MA70_Norm_recon_dir_rotate.mat"
mat_file = scipy.io.loadmat(MAT_FILE_NAME)
mat_file_value = mat_file['Data']


# ==========================================================
# Argument parser
# ==========================================================
parser = argparse.ArgumentParser(description='Train firing-rate RNN (pursuit task)')
parser.add_argument("--n_trials", required=True, type=int,
                    help="Number of training trials")
parser.add_argument("--output_dir", required=True, type=str,
                    help="Output directory")
parser.add_argument("--N", required=True, type=int,
                    help="Number of recurrent neurons")
parser.add_argument("--gain", default=1.5, type=float,
                    help="Recurrent weight gain")
parser.add_argument("--decay_taus", required=True,
                    nargs=1, type=float,
                    help="Synaptic decay time constant (single value)")
parser.add_argument("--gpu", type=str, default="0",
                    help="GPU id to use")
parser.add_argument("--gpu_frac", type=float, default=0.9,
                    help="Fraction of GPU memory to use")
parser.add_argument("--mode", type=str, choices=["activation", "prior_input"], required=True,
                    help="activation: 2-stage training / prior_input: prior input training")
parser.add_argument("--stage", type=int, required=True,
                    help="Training stage (1 or 2)")
parser.add_argument("--pretrained_path", type=str, default=None,
                    help="Path to Stage1 trained model (.mat)")
parser.add_argument("--activation", type=str, default="input_gain",
                    help="Activation parameter for Stage2 (input_gain/threshold/response_gain=)")
args = parser.parse_args()
set_gpu(args.gpu, args.gpu_frac)

# ==========================================================
# Output directory
# ==========================================================
out_dir = os.path.join(args.output_dir, 'models', 'pursuit')
os.makedirs(out_dir, exist_ok=True)

# ==========================================================
# Task settings
# ==========================================================
settings = {
    'T': 150,
    'stim_on': 150,
    'stim_dur': 150,
    'DeltaT': 1,
    'taus': args.decay_taus,
    'task': 'pursuit',
    'mat_file_value': mat_file_value
}


# ==========================================================
# Initialize weights
# ==========================================================
N = args.N

if args.mode == "activation":
    input_dim = 2
else:
    input_dim = 3

if args.mode == "activation":
    w_in = np.float32(np.random.randn(N, 2) / np.sqrt(2))
elif args.mode == "prior_input":
    w_in = np.float32(np.random.randn(N, 3) / np.sqrt(3))

w_out = np.float32(np.zeros((2, N)))
net = FR_RNN(
    N=N,
    w_in=w_in,
    gain=args.gain,
    w_out=w_out
)
print("Network initialized")

# ==========================================================
# Training parameters
# ==========================================================
training_params = {
    'learning_rate': 1e-4,
    'activation': args.activation
}

# ==========================================================
# Build TensorFlow graph
# ==========================================================
tf.reset_default_graph()

if args.stage == 2:

    if args.pretrained_path is None:
        raise ValueError("Stage2 requires --pretrained_path")

    pretrained = scipy.io.loadmat(args.pretrained_path)

    net.W = pretrained['w']
    net.w_in = pretrained['w_in']
    net.w_out = pretrained['w_out']
    net.b_out = float(pretrained['b_out'].squeeze())

input_node, z, x, r, o, w, w_in, w_out, b_out, taus = \
    construct_tf(net, settings, training_params,
                 stage=args.stage, input_dim=input_dim)

loss, training_op = loss_op(o, z, stage=args.stage)

print("TensorFlow graph constructed.")


# ==========================================================
# Training loop
# ==========================================================
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Training started...")

losses = np.zeros((args.n_trials, 1))

for tr in range(args.n_trials):

    start_time = time.time()

    if args.mode == "activation":
        if args.stage == 1:
            u, target = generate_input_target_pursuit_stage1(settings)
        elif args.stage == 2:
            u, target = generate_input_target_pursuit_stage2(settings)
    elif args.mode == "prior_input":
        if args.stage != 1:
            raise ValueError("prior_input mode only supports --stage 1")
        u, target = generate_input_target_pursuit(settings)

    _, t_loss = sess.run(
        [training_op, loss],
        feed_dict={input_node: u, z: target}
    )

    losses[tr] = t_loss

    print(f"Trial {tr} | Loss: {t_loss:.4f} | "
          f"Time: {time.time()-start_time:.2f}s")

print("Training finished.")


# ==========================================================
# Save model
# ==========================================================
var = {}
var['w'] = sess.run(w)
var['w_in'] = sess.run(w_in)
var['w_out'] = sess.run(w_out)
var['b_out'] = sess.run(b_out)
var['N'] = N
var['losses'] = losses
var['tau'] = settings['taus'][0]
try:
    input_gain_val = sess.run(tf.get_default_graph().get_tensor_by_name('input_gain:0'))
    threshold_val = sess.run(tf.get_default_graph().get_tensor_by_name('threshold:0'))
    response_gain_val = sess.run(tf.get_default_graph().get_tensor_by_name('response_gain:0'))
except KeyError:
    input_gain_val = np.ones((N, 1), dtype=np.float32)
    threshold_val = np.zeros((N, 1), dtype=np.float32)
    response_gain_val = np.ones((N, 1), dtype=np.float32)

var['input_gain'] = input_gain_val
var['threshold'] = threshold_val
var['response_gain'] = response_gain_val

var['activation'] = training_params['activation']

fname_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
fname = f"Task_pursuit_N_{N}_Tau_{settings['taus'][0]}_{fname_time}.mat"

scipy.io.savemat(os.path.join(out_dir, fname), var)

print("Model saved.")



