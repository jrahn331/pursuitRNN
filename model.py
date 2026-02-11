"""
Training script for continuous-time firing-rate RNN for smooth pursuit eye movement task

Name: Jungryul Ahn
Date: February 11, 2026
Email: jrahn331@g.skku.edu
"""

import numpy as np
import tensorflow as tf
import scipy.io
import math
import random

# ==========================================================
# RNN Model
# ==========================================================

class FR_RNN:
    def __init__(self, N, w_in, gain, w_out, b_out=0.0):

        self.N = N
        self.w_in = w_in
        self.w_out = w_out
        self.gain = gain
        self.b_out = b_out

        self.W = np.random.randn(N, N).astype(np.float32)
        self.W = self.W / np.sqrt(N) * gain



# ===========================================================
# Stage 1 - Pursuit Task Generator
# ===========================================================
def generate_input_target_pursuit_stage1(settings):
    T = settings["T"]
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    mat_file_value = settings["mat_file_value"]

    day_idx = random.choice(range(len(mat_file_value[0])))
    day_data = mat_file_value[0, day_idx]

    contrast_type = random.choice(["highCont", "lowCont"])

    hv = day_data[contrast_type]["wide"][0, 0]["hvel"][0, 0].flatten()
    vv = day_data[contrast_type]["wide"][0, 0]["vvel"][0, 0].flatten()

    tar_dir = day_data["tgdir"].item() + 7 * np.random.randn()

    amp = 1.0 if contrast_type == 'highCont' else 0.1

    tar_x = np.zeros((T,))
    tar_y = np.zeros((T,))

    tar_x[:] = amp * math.cos(np.deg2rad(tar_dir))
    tar_y[:] = amp * math.sin(np.deg2rad(tar_dir))

    tar_x += np.random.randn(T) / 100
    tar_y += np.random.randn(T) / 100

    eye_vel = np.zeros((T, 2))
    eye_vel[:, 0] = hv[stim_on:stim_on + stim_dur]
    eye_vel[:, 1] = vv[stim_on:stim_on + stim_dur]

    u = np.vstack((tar_x, tar_y))

    return u, eye_vel


def generate_input_target_pursuit_stage2(settings):
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    mat_file_value = settings['mat_file_value']

    day_idx = 0
    day_data = mat_file_value[0, day_idx]

    base_dir = day_data['tgdir'].item()

    contrast_type = random.choice(['highCont', 'lowCont'])
    prior_type = random.choice(
        ['narrow', 'narrow_minus15', 'narrow_plus15']
    )

    contrast_data = day_data[contrast_type]

    prior_data = contrast_data[prior_type]

    prior_data = prior_data[0,0]

    hv = prior_data['hvel'][0,0].flatten()
    vv = prior_data['vvel'][0,0].flatten()

    noise_std = 7

    if prior_type == 'narrow_minus15':
        tar_dir = base_dir - 15 + noise_std * np.random.randn()
    elif prior_type == 'narrow_plus15':
        tar_dir = base_dir + 15 + noise_std * np.random.randn()
    else:
        tar_dir = base_dir + noise_std * np.random.randn()

    amp = 1.0 if contrast_type == 'highCont' else 0.1

    tar_x = np.zeros((T,))
    tar_y = np.zeros((T,))

    tar_x[:] = amp * math.cos(np.deg2rad(tar_dir))
    tar_y[:] = amp * math.sin(np.deg2rad(tar_dir))

    tar_x += np.random.randn(T) / 100
    tar_y += np.random.randn(T) / 100

    eye_vel = np.zeros((T,2))
    eye_vel[:, 0] = hv[stim_on:stim_on + stim_dur]
    eye_vel[:, 1] = vv[stim_on:stim_on + stim_dur]

    u = np.vstack((tar_x, tar_y))

    return u, eye_vel


def generate_input_target_pursuit(settings):
    T = settings['T']
    stim_on = settings['stim_on']
    stim_dur = settings['stim_dur']

    mat_file_value = settings['mat_file_value']

    day_idx = random.choice(range(len(mat_file_value[0])))
    day_data = mat_file_value[0, day_idx]

    contrast_type = random.choice(["highCont", "lowCont"])
    base_dir = day_data['tgdir'].item()

    if base_dir == 0:
        prior_type = random.choice(
            ['wide', 'narrow', 'narrow_minus15', 'narrow_plus15']
        )
    else:
        prior_type = 'wide'

    prior_data = day_data[contrast_type][prior_type][0, 0]

    hv = prior_data['hvel'][0, 0].flatten()
    vv = prior_data['vvel'][0, 0].flatten()

    noise_std = 7

    if prior_type == 'wide':
        tar_dir = base_dir + noise_std * np.random.randn()
        prior_val = 0.0

    elif prior_type == 'narrow':
        tar_dir = base_dir + noise_std * np.random.randn()
        prior_val = 0.1

    elif prior_type == 'narrow_minus15':
        tar_dir = base_dir - 15 + noise_std * np.random.randn()
        prior_val = 0.1

    elif prior_type == 'narrow_plus15':
        tar_dir = base_dir + 15 + noise_std * np.random.randn()
        prior_val = 0.1

    amp = 1.0 if contrast_type == 'highCont' else 0.1

    tar_x = np.zeros((T,))
    tar_y = np.zeros((T,))
    prior = np.ones((T,)) * prior_val

    tar_x[:] = amp * math.cos(np.deg2rad(tar_dir))
    tar_y[:] = amp * math.sin(np.deg2rad(tar_dir))

    tar_x += np.random.randn(T) / 100
    tar_y += np.random.randn(T) / 100

    eye_vel = np.zeros((T, 2))
    eye_vel[:, 0] = hv[stim_on:stim_on + stim_dur]
    eye_vel[:, 1] = vv[stim_on:stim_on + stim_dur]

    u = np.vstack((tar_x, tar_y, prior))

    return u, eye_vel


# ==========================================================
# Construct TF Graph
# ==========================================================
def construct_tf(fr_rnn, settings, training_params, stage=1, input_dim=2):
    T = settings['T']
    tau = settings['taus'][0]
    DeltaT = settings['DeltaT']
    stim = tf.placeholder(tf.float32, [input_dim, T], name='u')
    z = tf.placeholder(tf.float32, [T, 2])
    x = []
    r = []
    x.append(tf.random_normal([fr_rnn.N, 1]) / 100)

    # ==========================================================
    # Stage 1: normal training
    # ==========================================================
    if stage == 1:
        def activation_fn(x):
            return tf.clip_by_value(tf.tanh(x), 0, 1)
        w = tf.get_variable('w', initializer=fr_rnn.W)
        w_in = tf.get_variable('w_in', initializer=fr_rnn.w_in)
        w_out = tf.get_variable('w_out', initializer=fr_rnn.w_out)
        b_out = tf.Variable(0.0, name='b_out')
        r.append(activation_fn(x[0]))

    # ==========================================================
    # Stage 2: freeze weights, train activation parameters
    # ==========================================================
    elif stage == 2:
        w = tf.get_variable('w', initializer=fr_rnn.W,
                            dtype=tf.float32, trainable=False)
        w_in = tf.get_variable('w_in', initializer=fr_rnn.w_in,
                               dtype=tf.float32, trainable=False)
        w_out = tf.get_variable('w_out', initializer=fr_rnn.w_out,
                                dtype=tf.float32, trainable=False)
        b_out = tf.get_variable('b_out', initializer=fr_rnn.b_out,
                                dtype=tf.float32, trainable=False)

        # activation parameters
        input_gain = tf.Variable(np.ones((fr_rnn.N, 1)),
                            dtype=tf.float32, name='input_gain', trainable=True)
        threshold = tf.Variable(np.zeros((fr_rnn.N, 1)),
                           dtype=tf.float32, name='threshold', trainable=True)
        response_gain = tf.Variable(np.ones((fr_rnn.N, 1)),
                            dtype=tf.float32, name='response_gain', trainable=True)
        act_type = training_params['activation']

        def activation_fn(x):
            if act_type == 'input_gain':
                return tf.clip_by_value(tf.tanh(input_gain * x), 0, 1)
            elif act_type == 'threshold':
                return tf.clip_by_value(tf.tanh(x - threshold), 0, 1)
            elif act_type == 'response_gain':
                return response_gain * tf.clip_by_value(tf.tanh(x), 0, 1)
            else:
                return tf.clip_by_value(tf.tanh(x), 0, 1)
        r.append(activation_fn(x[0]))


    o = []
    for t in range(1, T):
        next_x = (1 - DeltaT / tau) * x[t - 1] + \
                 (DeltaT / tau) * (
                     tf.matmul(w, r[t - 1]) +
                     tf.matmul(w_in, tf.expand_dims(stim[:, t - 1], 1))
                 ) + \
                 tf.random_normal([fr_rnn.N, 1]) / 100
        x.append(next_x)
        r.append(activation_fn(next_x))
        o.append(tf.matmul(w_out, r[t]) + b_out)

    return stim, z, x, r, o, w, w_in, w_out, b_out, tau


# ==========================================================
# Loss & Optimizer
# ==========================================================
def loss_op(o, z, stage=1):
    loss = tf.zeros(1)

    for i in range(len(o)):
        loss += tf.square(tf.transpose(o[i]) - z[i])

    loss = tf.reduce_sum(loss)
    loss = tf.sqrt(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    if stage == 1:
        training_op = optimizer.minimize(loss)

    elif stage == 2:
        train_vars = [v for v in tf.trainable_variables()
                      if 'input_gain' in v.name or
                         'threshold' in v.name or
                         'response_gain' in v.name]

        training_op = optimizer.minimize(loss, var_list=train_vars)

    return loss, training_op

