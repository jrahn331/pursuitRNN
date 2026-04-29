import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.io import savemat

# ============================================================
# Global parameters
# ============================================================

N = 500
dt = 1
T = 150
steps = int(T / dt)

phi = np.linspace(-np.pi, np.pi, N, endpoint=False)

# ============================================================
# Connectivity
# ============================================================

def circ_dist(a, b):
    return np.angle(np.exp(1j * (a - b)))

D = circ_dist(phi[:, None], phi[None, :])

sigma_exc = 0.5
sigma_inh = 2.0

W = np.exp(-0.5 * (D / sigma_exc)**2) - 2.0 * np.exp(-0.5 * (D / sigma_inh)**2)
W = W / np.sqrt(N)

# ============================================================
# LIF parameters
# ============================================================

V_rest = -70.0
V_reset = -70.0
V_th = -50.0
tau_m = 30.0

alpha = np.exp(-dt / tau_m)

# ============================================================
# Tuned inhibition
# ============================================================

inh_strength = 0.5

# distance from 0° (circular)
abs_phi = np.abs(np.angle(np.exp(1j * phi)))  # 0 ~ π

inh_profile = inh_strength * (abs_phi / np.pi)

# ============================================================
# External input
# ============================================================

I_base = 25.0
I_gain = 20.0
input_noise_std = 1.0

def input_current(theta, contrast, noise):

    tuning = np.cos(phi - theta)
    tuning = (tuning + 1) / 2

    return I_base + I_gain * contrast * tuning + noise

# ============================================================
# Simulation
# ============================================================

def run_bump(theta=0.0, contrast=1.0, use_inh=False, noise=None):

    V_rest_vec = np.ones(N) * V_rest

    if use_inh:
        V_rest_vec = V_rest_vec - inh_profile

    V = V_rest_vec.copy()
    spikes = np.zeros((steps, N))
    r = np.zeros(N)

    I_ext = input_current(theta, contrast, noise)

    for t in range(steps):

        r = alpha * r + spikes[t-1] if t > 0 else r

        I_rec = W @ r

        dV = (-(V - V_rest_vec) + I_ext + I_rec) / tau_m
        V += dt * dV

        fired = V >= V_th
        spikes[t, fired] = 1.0
        V[fired] = V_rest_vec[fired]

    return spikes



# ============================================================
# Population tuning analysis
# ============================================================

def circular_gaussian(x, A, mu, kappa, C):
    return A * np.exp(kappa * np.cos(x - mu)) + C

def fit_circular_gaussian(phi, rate):

    A0 = rate.max() - rate.min()
    C0 = rate.min()
    mu0 = phi[np.argmax(rate)]
    kappa0 = 1.0

    p0 = [A0, mu0, kappa0, C0]

    bounds = (
        [0, -np.pi, 0, 0],
        [np.inf, np.pi, 50, np.inf]
    )

    popt, _ = curve_fit(
        circular_gaussian,
        phi,
        rate,
        p0=p0,
        bounds=bounds,
        maxfev=10000
    )

    return popt

def kappa_to_fwhm_deg(kappa):

    # numerical safety
    val = np.log(0.5) / np.maximum(kappa, 1e-8) + 1
    val = np.clip(val, -1, 1)

    fwhm_rad = 2 * np.arccos(val)

    return np.degrees(fwhm_rad)

def run_one_episode(theta, n_trials=100):

    rate_accum = {
        "high_no_inh": np.zeros(N),
        "high_inh":    np.zeros(N),
        "low_no_inh":  np.zeros(N),
        "low_inh":     np.zeros(N),
    }

    conditions = [
    ("high_no_inh", 1.0, False),
    ("high_inh",    1.0, True),
    ("low_no_inh",  0.1, False),
    ("low_inh",     0.1, True),
    ]

    for _ in range(n_trials):

        for name, contrast, inh in conditions:

            noise = np.random.normal(0, input_noise_std, size=N)

            spikes = run_bump(
                theta,
                contrast,
                use_inh=inh,
                noise=noise  
            )

            rate = spikes.sum(axis=0) / (T / 1000.0)
            rate_accum[name] += rate

    for key in rate_accum:
        rate_accum[key] /= n_trials

    results = {}

    for key, rate in rate_accum.items():

        popt = fit_circular_gaussian(phi, rate)
        _, _, kappa, _ = popt

        results[key] = kappa_to_fwhm_deg(kappa)

    return results

n_episodes = 100

width_samples = {
    "high_no_inh": [],
    "high_inh": [],
    "low_no_inh": [],
    "low_inh": []
}

for _ in range(n_episodes):

    res = run_one_episode(theta=0, n_trials=100)

    for key in width_samples:
        width_samples[key].append(res[key])

from scipy.io import savemat

savemat("tuning_width_samples.mat", {
    "high_no_inh": np.array(width_samples["high_no_inh"]),
    "high_inh":    np.array(width_samples["high_inh"]),
    "low_no_inh":  np.array(width_samples["low_no_inh"]),
    "low_inh":     np.array(width_samples["low_inh"]),
})



# ============================================================
# Pursuit bias analysis
# ============================================================

def population_vector_decode(rate, phi):
    x = np.sum(rate * np.cos(phi))
    y = np.sum(rate * np.sin(phi))
    return np.arctan2(y, x)

def circular_diff(a, b):
    return np.degrees(np.angle(np.exp(1j * (a - b))))

theta_stim = np.deg2rad(15)
n_trials_decode = 100

decoded_samples = {
    "high_no_inh": [],
    "high_inh": [],
    "low_no_inh": [],
    "low_inh": []
}

conditions = [
    ("high_no_inh", 1.0, False),
    ("high_inh",    1.0, True),
    ("low_no_inh",  0.1, False),
    ("low_inh",     0.1, True),
]

for _ in range(n_trials_decode):

    noise = np.random.normal(0, input_noise_std, size=N)

    for name, contrast, inh in conditions:

        spikes = run_bump(
            theta_stim,
            contrast,
            use_inh=inh,
            noise=noise
        )

        rate = spikes.sum(axis=0) / (T / 1000.0)

        decoded = population_vector_decode(rate, phi)

        decoded_samples[name].append(decoded)

decoded_high_noinh = np.array(decoded_samples["high_no_inh"])
decoded_high_inh   = np.array(decoded_samples["high_inh"])
decoded_low_noinh  = np.array(decoded_samples["low_no_inh"])
decoded_low_inh    = np.array(decoded_samples["low_inh"])

bias_high = circular_diff(decoded_high_inh, decoded_high_noinh)
bias_low  = circular_diff(decoded_low_inh, decoded_low_noinh)


# ============================================================
# Print summary
# ============================================================

print("\n=== Decoding bias (15 deg stimulus) ===")
print(f"High contrast mean |Δ|: {np.mean(np.abs(bias_high)):.3f} deg")
print(f"Low  contrast mean |Δ|: {np.mean(np.abs(bias_low)):.3f} deg")


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(5,4))

plt.bar(
    ["High contrast", "Low contrast"],
    [np.mean(np.abs(bias_high)), np.mean(np.abs(bias_low))],
    color=["black", "gray"]
)

plt.ylabel("|Δ decoded angle| (deg)")
plt.title("Decoding bias (15° stimulus, 100 trials)")
plt.grid(axis='y')
plt.show()


# ============================================================
# Save to MATLAB
# ============================================================

savemat("decoding_results_15deg.mat", {
    "decoded_high_noinh": decoded_high_noinh,
    "decoded_high_inh":   decoded_high_inh,
    "decoded_low_noinh":  decoded_low_noinh,
    "decoded_low_inh":    decoded_low_inh,
    "bias_high": bias_high,
    "bias_low":  bias_low,
})






