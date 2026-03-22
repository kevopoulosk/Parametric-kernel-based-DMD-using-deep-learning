import numpy as np
import matplotlib.pyplot as plt
import pickle

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


t_test = [0.15, 4]
ranks_pod = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]


with open(f'errors_dict_Heat_equation.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

pdmd_test_error_mean_015 = np.load('../Parametric_DMD/HEAT_rel_errs_t015_mean.npy')
pdmd_test_error_std_015 = np.load('../Parametric_DMD/HEAT_rel_errs_t015_std.npy')

pdmd_test_error_mean_05 = np.load('../Parametric_DMD/HEAT_rel_errs_t050_mean.npy')
pdmd_test_error_std_05 = np.load('../Parametric_DMD/HEAT_rel_errs_t050_std.npy')

pdmd_test_error_mean_400 = np.load('../Parametric_DMD/HEAT_rel_errs_t400_mean.npy')
pdmd_test_error_std_400 = np.load('../Parametric_DMD/HEAT_rel_errs_t400_std.npy')

pdmd_means = [pdmd_test_error_mean_015, pdmd_test_error_mean_05, pdmd_test_error_mean_400]
pdmd_stds = [pdmd_test_error_std_015, pdmd_test_error_std_05, pdmd_test_error_std_400]


test_error_mean = []
test_error_std = []
for t, means, stds in zip(t_test, pdmd_means, pdmd_stds):
    test_error_mean = loaded_dict[f"t = {t}"][0][1]
    test_error_std = loaded_dict[f"t = {t}"][0][2]

    plt.figure(figsize=(8, 4))
    plt.plot(ranks_pod, means, '-o', color='black')
    plt.grid(True)
    plt.axhline(test_error_mean, color='r', label='mean pLANDO')
    plt.axhspan(test_error_mean - test_error_std, test_error_mean + test_error_std,
                color='r', alpha=0.2,
                label='±1 std')
    plt.yscale('log')
    plt.savefig(f'heat_pdmd_plando_ranks_comparison_{t}.svg')