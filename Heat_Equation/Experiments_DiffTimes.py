from Parametric_LANDO.pLANDO_HighDim import *
import pickle
from matplotlib.ticker import ScalarFormatter

test_t_instances = [0.15, 0.45, 1, 1.5, 2, 2.5, 3, 3.5, 4]
t_pod_experiment = 3
t_train = 2
dt = 0.01
kernel = linear_kernel
nu_sparsity = 1e-5
fraction_train = 0.5
fraction_validation = 0.17

depth = 4
width = 110
epochs = 20000
batch = 0.33
svd_truncation = 10
verb = False

save_directory = "/Users/konstantinoskevopoulos/Desktop/Heat_eq_Results"

parametric_lando = ParametricLANDO(kernel=kernel, horizon_train=t_train, dt=dt, sparsity_tol=nu_sparsity,
                                   directory_processed_data="/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis"
                                                            "/SnapshotData_Processed",
                                   directory_samples="/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis"
                                                     "/Parameter_Samples",
                                   training_frac=fraction_train, validation_frac=fraction_validation)

_, sparse_dicts = parametric_lando.OfflinePhase()

dict_samples = np.mean([sparse_dicts[i].shape[1] for i in range(len(sparse_dicts))])
print(f"Mean number of dictionary samples, nu={nu_sparsity}: {int(dict_samples)}")

dict_errors = {}

pbar = tqdm(total=len(test_t_instances), desc=rf"pLANDO for several $t^*$...")
for t in test_t_instances:

    dict_errors[f"t = {t}"] = []

    error_results = parametric_lando.OnlinePhase(T_end_test=t,
                                                 fnn_depth=depth,
                                                 fnn_width=width,
                                                 epochs=epochs,
                                                 batch_size=batch,
                                                 trunc_rank=svd_truncation,
                                                 verbose=verb)

    dict_errors[f"t = {t}"].append(error_results)
    pbar.update()
pbar.close()
#
#
# ### Save data from this run into dictionary form
# with open('errors_dict.pkl', 'wb') as f:
#     pickle.dump(dict_errors, f)


num_bases = np.arange(2, 22, 2)
system_energies = []
pod_projection_errors = []
for base in num_bases:
    system_energy, pod_projection_error = parametric_lando.OnlinePhase(T_end_test=t_pod_experiment,
                                                                       fnn_depth=depth,
                                                                       fnn_width=width,
                                                                       epochs=epochs,
                                                                       batch_size=batch,
                                                                       trunc_rank=base,
                                                                       verbose=verb,
                                                                       pod_experiment=True)
    system_energies.append(system_energy)
    pod_projection_errors.append(pod_projection_error)

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

fig, ax1 = plt.subplots(figsize=(7.4, 4.75))

ax1.plot(num_bases, [100*system_energies[i] for i in range(len(system_energies))], '-o', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.yaxis.get_offset_text().set_visible(False)  # Hide the offset text
ax1.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))  # Disable offset and scientific notation
ax1.ticklabel_format(style='plain', axis='y')
ax1.set_xlabel('number of reduced bases')
ax1.set_ylabel('percentage of energy (%)', color='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel(r'relative $L_2$ error (%)', color='tab:red')
ax2.plot(num_bases, [100*pod_projection_errors[i] for i in range(len(pod_projection_errors))], '-o', color='tab:red')
ax2.set_yscale("log")
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.grid(True)
fig.tight_layout()
plt.savefig(save_directory + '/POD_Experiment.png')
plt.show()
