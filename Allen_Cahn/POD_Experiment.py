from Parametric_LANDO.pLANDO_HighDim import *

problem = "allen_cahn"

if problem == "allen_cahn":
    t_instances = [0.17, 0.65, 0.95]
    t_train = 0.6
    dt = 0.001
    fraction_train = 0.4
    fraction_validation = 0.15
    nu_sparsity = 1e-6
    reduced_bases = np.arange(2, 24, 2)

    directory_processed_data = "/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/SnapshotData"
    directory_samples = "/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/Parameter_Samples"
    samples_filename = "/samples_allen_cahn.txt"
    save_directory = "/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
else:
    t_instances = [0.15, 1, 3.5]
    t_train = 2
    dt = 0.01
    kernel = linear_kernel
    nu_sparsity = 1e-5
    fraction_train = 0.5
    fraction_validation = 0.17
    reduced_bases = np.arange(2, 24, 2)

    directory_processed_data = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/SnapshotData_Processed"
    directory_samples = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/Parameter_Samples"
    samples_filename = "/samples_heat_eq.txt"
    save_directory = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis"

kernel = linear_kernel
depth = 4
width = 110
epochs = 20000
batch = 0.33
verb = False

parametric_lando = ParametricLANDO(kernel=kernel, horizon_train=t_train, dt=dt, sparsity_tol=nu_sparsity,
                                   directory_processed_data=directory_processed_data,
                                   directory_samples=directory_samples,
                                   training_frac=fraction_train, validation_frac=fraction_validation,
                                   problem=problem, samples_filename=samples_filename)

_, sparse_dicts = parametric_lando.OfflinePhase()

dict_errors = {}
pbar = tqdm(total=len(t_instances), desc="Progress of experiments")
for t in t_instances:

    dict_errors[f"t = {t}"] = []

    for base in reduced_bases:
        err_results = parametric_lando.OnlinePhase(T_end_test=t,
                                                   fnn_depth=depth,
                                                   fnn_width=width,
                                                   epochs=epochs,
                                                   batch_size=batch,
                                                   trunc_rank=base,
                                                   verbose=verb,
                                                   pod_plando_err_exp=True)

        dict_errors[f"t = {t}"].append(err_results)
    pbar.update()
pbar.close()


### Visualise the different errors
t_tests = list(dict_errors.keys())
errors = list(dict_errors.values())

colors = ['tab:blue', 'tab:red', 'tab:green']

plt.figure()
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
for i in range(len(t_tests)):
    plt.errorbar(reduced_bases, [errors[i][k][1] for k in range(len(reduced_bases))],
                 yerr=[errors[i][k][2] for k in range(len(reduced_bases))],
                 fmt='-o', capsize=5,  label=fr'$t^*=${t_instances[i]}', color=colors[i])

plt.grid(True)
plt.legend(ncol=3, bbox_to_anchor=(0.99, 1.18), frameon=False)
filename = "/POD_pLANDO_Errors.png"
plt.savefig(save_directory + filename)
plt.clf()

