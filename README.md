# Smooth Pursuit Eye Movement RNN

This repository contains the implementation of a **continuous-time firing-rate recurrent neural network (RNN)** trained to perform a smooth pursuit eye movement task based on behavioral data.

The network can be trained using two different strategies:

### 1. Two-stage activation training ('activation')
  - Stage 1: Train recurrent and output weights
  - Stage 2: Freeze weights and train activation parameters only
### 2. Prior-input training ('prior_input')
  - Network receives an explicit prior signal as an additional input channel
  - Only single-stage training is supported

## Requirements
The code was developed and tested with:
- Python 3.6
- tesnorflow 1.10.0
- tensorflow-gpu 1.10.0 (optional, recommended if GPU is available)
- numpy 1.14.5
- scipy 1.5.0

## Usage

### Training Script ('train.py')

The main script `train.py` trains a firing-rate RNN for the smooth pursuit task using behavioral eye velocity data.
It supports two different training modes:

### Input Arguments
The script takes the following command-line arguments:

#### Required arguments
- `n_trials`: Number of training trials (iterations)
- `N`: Number of recurrent neurons in the RNN
- `decay_taus`: Synaptic decay time constant(ms)
- `mode`: Training mode
    `activation` : two-stage training (Stage1 + Stage2)
    `prior_input` : training with explicit prior input channel
- `stage`: Training stage
    `1` : train weights (or full network in prior_input mode)
    `2` : train activation parameters (only valid in activation mode)
- `output_dir`: Directory to save trained model. Models are saved under `<output_dir>/models/pursuit`.

#### Optional arguments  
- `gpu` (default `0`): GPU ID to use
- `gpu_frac` (default `0.9`): Fraction of GPU memory to allocate
- `gain` (default `1.5`): Gain for initial recurrent weight matrix
- `pretrained_path`: Path to Stage-1 trained model (.mat) (required for Stage-2).
- `activation` (default `input_gain`): Activation parameter type trained in Stage-2. Options: input_gain, threshold, response_gain

#### Note  
> - `--stage` 2 is only valid when `--mode` activation  
> - `--mode` prior_input only supports `--stage` 1
  
## Example Usage
### Stage 1: Train network weights
```
python train.py --gpu 0 --gpu_frac 0.7 \
--n_trials 4000 --N 250 \
--decay_taus 30 --gain 1.5 \
--mode activation --stage 1 \
--output_dir ../
```

### Stage 2: Train activation function parameters
```
python train.py --gpu 0 --gpu_frac 0.7 \
--n_trials 2000 --N 250 \
--decay_taus 30 --gain 1.5 \
--mode activation --stage 2 --activation threshold \
--pretrained_path PATH_TO_STAGE1_MODEL.mat --output_dir ../
```

### Prior-input training
```
python train.py --gpu 0 --gpu_frac 0.7 \
--n_trials 6000 --N 250 \
--decay_taus 30 --gain 1.5 \
--mode prior_input --stage 1 \
--output_dir ../
```

### Outputs
After training, the script automatically:
  Saves the trained model as a MATLAB .mat file containing:
    - Recurrent weights `w`
    - Input weights `w_in`
    - Output (or redaout) weights `w_out`
    - Output bias `b_out`
    - Activation parameters
    - Training loss history
    - Example filename: Task_pursuit_N_250_Tau_30_2026_02_11_18122.mat

### Notes
- Behavioral eye velocity data is included.
- If you use this repository for your research, please cite our work:
  'Tuned inhibitory control of neuronal firing thresholds explains predictive sensorimotor behavior'

## Contact
Jungryul Ahn
Email: jrahn331@g.skku.edu
