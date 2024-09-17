import matplotlib.pyplot as plt
import pickle

save_directory = "/Users/konstantinoskevopoulos/Desktop/Heat_eq_Results"

t_test = [0.15, 0.45, 1, 1.5, 2, 2.5, 3, 3.5, 4]

log_plot = [True, False]

with open(f'errors_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

test_error_mean = []
test_error_std = []
for t in t_test:
    test_error_mean.append(loaded_dict[f"t = {t}"][0][1])
    test_error_std.append(loaded_dict[f"t = {t}"][0][2])

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

for log in log_plot:
    plt.figure()
    plt.errorbar(t_test, test_error_mean, yerr=test_error_std, marker='o', capsize=5)
    plt.grid(True)
    if log:
        plt.yscale("log")
    plt.axvline(x=2, linestyle='--', color='black')
    if log:
        plt.savefig(save_directory+"/pLANDO_Errors_log.png")
    else:
        plt.savefig(save_directory + "/pLANDO_Errors.png")
    plt.clf()







