import numpy as np
import pickle
import matplotlib.pyplot as plt

params = {
    'axes.labelsize': 15.4,
    'font.size': 15.4,
    'legend.fontsize': 15.4,
    'xtick.labelsize': 15.4,
    'ytick.labelsize': 15.4,
    'text.usetex': False,
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
}
plt.rcParams.update(params)


t_test = [300, 450, 500]

pdmd_rel_errs_mean = np.load('../Parametric_DMD/LV_rel_errs_mean_2d.npy')
pdmd_rel_errs_std = np.load('../Parametric_DMD/LV_rel_errs_std_2d.npy')

with open(f'../Lotka_Volterra/errors_dict_2D_NN.pkl', 'rb') as f:
    loaded_dict_plando = pickle.load(f)


test_error_mean = []
test_error_std = []

mean_err_300 = loaded_dict_plando['IC=[80, 20]'][6][0]
std_err_300 = loaded_dict_plando['IC=[80, 20]'][6][1]

test_error_mean.append(mean_err_300)
test_error_std.append(std_err_300)


mean_err_450 = loaded_dict_plando['IC=[80, 20]'][9][0]
std_err_450 = loaded_dict_plando['IC=[80, 20]'][9][1]

test_error_mean.append(mean_err_450)
test_error_std.append(std_err_450)


mean_err_500 = loaded_dict_plando['IC=[80, 20]'][10][0]
std_err_500 = loaded_dict_plando['IC=[80, 20]'][10][1]

test_error_mean.append(mean_err_500)
test_error_std.append(std_err_500)

plt.figure()
plt.errorbar(t_test, test_error_mean, yerr=test_error_std, marker='o', capsize=5, color='red')
plt.errorbar(t_test, pdmd_rel_errs_mean, yerr=pdmd_rel_errs_std, marker='o', capsize=5, color='black')
plt.grid(True)
plt.yscale('log')
plt.axvline(x=400, linestyle='--', color='black')
plt.savefig('lv_comparison_pdmd_plando_2d.svg')

