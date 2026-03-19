import numpy as np
from scipy.stats import qmc
from pLANDO_LowDim import *
from Lotka_Volterra.Lotka_Volterra_model import *
from Lotka_Volterra.Lotka_Volterra_Deriv import *


def LatinHypercube(dim_sample, low_bounds, upp_bounds, num_samples):
    """
    Function that is used to sample the parameters from a latin hypercube.
    :param dim_sample: The dimension that we sample
    :param low_bounds: lower bound of the sampling interval
    :param upp_bounds: upper bound of the sampling interval
    :param num_samples: number of desired samples
    :return:
    """
    sampler = qmc.LatinHypercube(d=dim_sample)
    sample = sampler.random(n=num_samples)

    l_bounds = low_bounds
    u_bounds = upp_bounds
    sample_params = qmc.scale(sample, l_bounds, u_bounds)
    return sample_params


NumSamples_Train = 400
NumSamples_Test = 500

varied = 1
low_bounds = [0.015]
upp_bounds = [0.1]
fixed_params = np.array([0.002, 0.2, 0.0025])

params_sampled_lh = LatinHypercube(dim_sample=varied, low_bounds=low_bounds, upp_bounds=upp_bounds,
                                   num_samples=NumSamples_Train + NumSamples_Test)

params_not_varied = np.tile(fixed_params, (NumSamples_Train + NumSamples_Test, 1))
param_samples = np.concatenate((params_sampled_lh, params_not_varied), axis=1)

params_test = params_sampled_lh[NumSamples_Train:]

np.save('LV_1D_param_samples.npy', param_samples)

dofs = 2
T_train = 400

sensors = 600
sparsity_lando = 1e-6

Init_Condition = [70, 20]

train_frac = 0.444
depth = 3
width = 32
epochs = 36000

t_vals_test = [300, 450, 500]

pLANDO = ParametricLANDO(param_samples=param_samples, T_end_train=T_train, num_sensors=sensors, kernel=quadratic_kernel,
                         sparsity_tol=sparsity_lando, X_snapshot=Lotka_Volterra_Snapshot, Y_deriv=Lotka_Volterra_Deriv,
                         num_params_varied=varied, dofs=dofs, batch_size_frac=0.2)

W_tildes, SparseDicts_all, X_all, Y_all = pLANDO.OfflinePhase()
X_test = X_all[NumSamples_Train:]

mean_relative_errors = []
for t in t_vals_test:
    interp_model, rel_errors = pLANDO.OnlinePhase(
        T_end_test=t,
        train_frac=train_frac,
        fnn_depth=depth, fnn_width=width,
        epochs=epochs, IC_predict=Init_Condition, verb=True)

    mean_relative_errors.append(np.mean(rel_errors))

print(f"t^* = 300: Mean relative error {mean_relative_errors[0]}\n"
      f"t^* = 450: Mean relative error {mean_relative_errors[1]}\n"
      f"t^* = 500: Mean relative error {mean_relative_errors[2]}")

np.save('mean_relative_errors_1d_lv.npy', np.array(mean_relative_errors))
