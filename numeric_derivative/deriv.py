import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def sample_function(t):
    return np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 23 * t)

def deriv_sample_function(t):
    return 2 * np.pi * 10 * np.cos(2 * np.pi * 10 * t) + 2 * np.pi * 23 * np.cos(2 * np.pi * 23 * t)

def sample_function_lorentzian(x, kappa):
    return (kappa/2) ** 2 / ((kappa/2) ** 2 + x ** 2)

def deriv_lorentzian(x, kappa):
    return (kappa/2) ** 2 / ((kappa/2) ** 2 + x ** 2) ** 2 * (-2 * x)

def sample_function_gaussian(x, sigma):
    return np.exp(-x ** 2 / sigma ** 2)

def deriv_gaussian(x, sigma):
    return np.exp(-x ** 2 / sigma ** 2) * (-2 * x / sigma ** 2)

def fourier_series(amplitudes, frequencies, t, cut_off=None):
    fourier_sum = np.zeros(len(t), dtype=complex)
    if cut_off is None:
        cut_off = frequencies[-1]
    for amplitude, frequency in zip(amplitudes, frequencies):
        if frequency > cut_off:
            break
        fourier_sum += amplitude * np.exp(2j * np.pi * frequency * t)
    return np.real(fourier_sum / len(amplitudes))

def fourier_series(amplitudes, frequencies, t, cut_off=None):
    if cut_off is None:
        cut_off = frequencies[-1]

    summands = amplitudes * np.exp(2j * np.pi * t.reshape(len(t), 1) @ frequencies.reshape(1, len(frequencies)))
    msk = frequencies > cut_off
    summands[:, msk] = 0
    return np.real(np.mean(summands, axis=1))

def _deriv_fourier(amplitudes, frequencies, t, cut_off=None):
    if cut_off is None:
        cut_off = frequencies[-1]

    summands = amplitudes * np.exp(2j * np.pi * t.reshape(len(t), 1) @ frequencies.reshape(1, len(frequencies)))
    summands *= 2j * np.pi * frequencies
    msk = frequencies>cut_off
    summands[:, msk] = 0
    return np.real(np.mean(summands, axis=1))

def deriv_simple(x, y):
    return (y[1:] - y[:-1]) / (x[1:] - x[:-1]), (x[1:] + x[:-1]) / 2

def deriv_fourier(x, y, cut_off=None, ret_func=False):
    # if x[0] < 0:
    #     x_new = x + x[0]
    # else:
    #     x_new = x
    x_new = x
    rfft = np.fft.rfft(y)
    rfft_freq = np.fft.rfftfreq(len(y), d=x_new[1] - x_new[0])
    if ret_func:
        return fourier_series(rfft, rfft_freq, x_new, cut_off), x
    else:
        return _deriv_fourier(rfft, rfft_freq, x_new, cut_off), x


def deriv_polyfit(x, y, window_size, degree, ret_func=False):
    if window_size % 2 == 0:
        window_size += 1
    pre_post_elements = window_size // 2
    fit_array = np.array([y[n - pre_post_elements : n + pre_post_elements + 1]
                          for n in range(pre_post_elements, len(y) - pre_post_elements)]).T

    fit_x = np.linspace(-1 * (window_size - 1) / 2, (window_size - 1) / 2, window_size) * (x[1] - x[0])

    if ret_func:
        params = np.polyfit(fit_x, fit_array, deg=degree)
        return np.polyval(params, 0), x[pre_post_elements:-pre_post_elements]
    else:
        params = np.arange(1, degree + 1)[::-1] * np.polyfit(fit_x, fit_array, deg=degree)[:-1].T
        return np.polyval(params.T, 0), x[pre_post_elements:-pre_post_elements]



runtime = np.linspace(0, 1, 1000)
noise = np.random.normal(0, 0.05, len(runtime))
sample_data = sample_function(runtime)
analytical_deriv = deriv_sample_function(runtime)
sample_data_noisy = sample_data + noise

runtime = np.linspace(-50, 50, 1000)
sample_data = sample_function_lorentzian(runtime, 10)
analytical_deriv = deriv_lorentzian(runtime, 10)
sample_data_noisy = sample_data + noise
sample_data_noisy = sample_data

# runtime = np.linspace(-50, 50, 1000)
# sample_data = sample_function_gaussian(runtime, 10)
# analytical_deriv = deriv_gaussian(runtime, 10)
# sample_data_noisy = sample_data + noise
#sample_data_noisy = sample_data

fig, ax = plt.subplots()
#ax.plot(runtime, sample_data_noisy)
ax.plot(runtime, sample_data)
#fit_data = fourier_series(rfft, rfft_freq, runtime, 30)
#ax.plot(runtime, fit_data)

#ax.plot(runtime, analytical_deriv)

polyfit_deriv, polyfit_runtime = deriv_polyfit(runtime, sample_data_noisy, window_size=29, degree=2, ret_func=True)
ax.plot(polyfit_runtime, polyfit_deriv)

fourier_deriv, fourier_runtime = deriv_fourier(runtime, sample_data_noisy, cut_off=None, ret_func=True)
ax.plot(fourier_runtime, fourier_deriv)

simple_deriv, simple_runtime = deriv_simple(runtime, sample_data_noisy)
#ax.plot(simple_runtime, simple_deriv)
plt.show()
