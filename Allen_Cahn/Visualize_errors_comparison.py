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


t_test = [0.17, 0.75]
ranks_pod = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]


with open(f'errors_dict_Allen_Cahn.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

pdmd_test_error_mean_017 = np.load('../Parametric_DMD/AC_rel_errs_t017_mean.npy')
pdmd_test_error_std_017 = np.load('../Parametric_DMD/AC_rel_errs_t017_std.npy')

pdmd_test_error_mean_075 = np.load('../Parametric_DMD/AC_rel_errs_t075_mean.npy')
pdmd_test_error_std_075 = np.load('../Parametric_DMD/AC_rel_errs_t075_std.npy')

pdmd_means = [pdmd_test_error_mean_017, pdmd_test_error_mean_075]
pdmd_stds = [pdmd_test_error_std_017, pdmd_test_error_std_075]


test_error_mean = []
test_error_std = []
for t, means, stds in zip(t_test, pdmd_means, pdmd_stds):
    test_error_mean = loaded_dict[f"t = {t}"][0][1]
    test_error_std = loaded_dict[f"t = {t}"][0][2]

    plt.figure()
    plt.plot(ranks_pod, means, '-o', color='black')
    plt.grid(True)
    plt.axhline(test_error_mean, color='r', label='mean pLANDO')
    plt.axhspan(test_error_mean - test_error_std, test_error_mean + test_error_std,
                color='r', alpha=0.2,
                label='±1 std')
    plt.yscale('log')
    plt.savefig(f'ac_pdmd_plando_ranks_comparison_{t}.svg')