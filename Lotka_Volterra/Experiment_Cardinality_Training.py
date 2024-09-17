from Parametric_LANDO.pLANDO_LowDim import *
from Lotka_Volterra_model import *
from Lotka_Volterra_Deriv import *


T_train = 400
sensors = 600
sparsity_lando = 1e-6

NumSamples_Test = 500
train_frac = 0.75
depth = 3
width = 32
epochs = 36000

varied = 1
low_bounds = [0.015]
upp_bounds = [0.1]
fixed_params = np.array([0.002, 0.2, 0.0025])
init_condition = [80, 20]

colors = ['tab:blue', 'tab:red', 'tab:green']


training_set_cardinalities = np.arange(50, 500, 50)
T_tests = [50, 300, 600]

save_directory = "/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results/Param1D"


dict_errors = {}
pbar = tqdm(total=len(T_tests), desc="Progress of experiments")
for t in T_tests:
    dict_errors[f"T_test={t}"] = []

    for card in training_set_cardinalities:
        parametric_lando = ParametricLANDO(kernel=quadratic_kernel, horizon_train=T_train,
                                           num_samples_train=card,
                                           num_sensors=sensors, sparsity_tol=sparsity_lando, batch_frac=0.2,
                                           params_varied=varied, low_bound=low_bounds, upp_bound=upp_bounds,
                                           fixed_params_val=fixed_params, generate_snapshot=Lotka_Volterra_Snapshot,
                                           generate_deriv=Lotka_Volterra_Deriv)

        ### First, the offline phase of the parametrization algorithm is performed
        w_tildes, sparse_dicts, mu_samples_train = parametric_lando.OfflinePhase()

        ### We proceed to the online phase of the algorithm
        interp_model, X_train, y_train, X_valid, y_valid, _, _, reconstruct_rel_errs = parametric_lando.OnlinePhase(
            T_end_test=t,
            fraction_train=train_frac,
            fnn_depth=depth, fnn_width=width,
            epochs=epochs, IC_predict=init_condition, verb=False)

        mean_error_train, mean_error_test, std_error_test = parametric_lando.TestPhase(num_samples_test=NumSamples_Test,
                                                                                       interp_model=interp_model,
                                                                                       reconstruction_relative_errors=reconstruct_rel_errs,
                                                                                       x_train=X_train,
                                                                                       y_train=y_train,
                                                                                       directory_1d=None,
                                                                                       visuals=False)

        dict_errors[f"T_test={t}"].append((mean_error_test, std_error_test))
    pbar.update()
pbar.close()


### Visualise the different errors
t_tests = list(dict_errors.keys())
errors = list(dict_errors.values())

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
    plt.errorbar(training_set_cardinalities, [errors[i][k][0] for k in range(len(training_set_cardinalities))],
                 yerr=[errors[i][k][1] for k in range(len(training_set_cardinalities))],
                 fmt='-o', capsize=5,  label=fr'$t^*=${T_tests[i]}', color=colors[i])

plt.grid(True)
plt.legend(ncol=3, bbox_to_anchor=(0.99, 1.18), frameon=False)
filename = "/Training_Cardinality.png"
plt.savefig(save_directory + filename)
plt.clf()










