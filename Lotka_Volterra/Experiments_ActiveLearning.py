from Active_Learning.Active_Learning_LowDim import *

n_init = 20
n_all = 40
dim_init = 1
num_nets = 7
low_init = [0.015]
upp_init = [0.12]
fixed_params = np.array([0.002, 0.2, 0.0025])
plando = ParametricLANDO(kernel=quadratic_kernel, horizon_train=400, sparsity_tol=1e-6, num_samples_train=None,
                         num_sensors=600, batch_frac=0.2, params_varied=1, low_bound=low_init, upp_bound=upp_init,
                         fixed_params_val=fixed_params, rbf=False, sampling_al=True, after_al=False)
num_valid = 700
depths = [3 for _ in range(num_nets)]
widths = [32 for _ in range(num_nets)]
activations = ["snake" for _ in range(num_nets)]
onlinephase_args = [500, [70, 20]]
ensemble_args = [num_nets, dim_init, 2, depths, widths, activations, 35000, 0.2]

validation_samples = LatinHypercube(dim_init, low_init, upp_init, num_valid)

### Obtain the "most informative" training dataset
train_dataset = ActiveLearning_Algorithm(n_init, plando, n_all, dim_init, low_init, upp_init,
                                         num_valid, onlinephase_args, ensemble_args)

###Now we run pLANDO with the optimal training
parametric_rom_active = ParametricLANDO(kernel=quadratic_kernel, horizon_train=400, sparsity_tol=1e-6,
                                        num_samples_train=None,
                                        num_sensors=600, batch_frac=0.2, params_varied=1, low_bound=low_init,
                                        upp_bound=upp_init,
                                        fixed_params_val=fixed_params, rbf=False, sampling_al=True, after_al=True)

parametric_rom_active.OfflinePhase(samples_train_al=train_dataset, samples_valid_al=validation_samples)

interp_model, X_train, y_train, X_valid, y_valid, rel_err_train, rel_err_valid, reconstruction_relative_errors = parametric_rom_active.OnlinePhase(
    T_end_test=onlinephase_args[0],
    IC_predict=onlinephase_args[1],
    epochs=35000,
    verb=False,
    fnn_depth=3,
    fnn_width=32)

mean_error_train_al, mean_error_test_al = parametric_rom_active.TestPhase(num_samples_test=800,
                                                                          interp_model=interp_model,
                                                                          x_train=X_train, y_train=y_train,
                                                                          reconstruction_relative_errors=reconstruction_relative_errors,
                                                                          visuals=True,
                                                                          directory_1d=f"/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results_Alternative/Param1D/Active_Learning/AL/")

########################################################################################################################
### Perform pLANDO with LHS sampling
parametric_rom_lhs = ParametricLANDO(kernel=quadratic_kernel, horizon_train=400, sparsity_tol=1e-6,
                                     num_samples_train=800,
                                     num_sensors=600, batch_frac=0.2, params_varied=1, low_bound=low_init,
                                     upp_bound=upp_init,
                                     fixed_params_val=fixed_params, rbf=False, sampling_al=False, after_al=True)

parametric_rom_lhs.OfflinePhase(samples_train_al=None, samples_valid_al=None)

fnn_lhs, X_train_lhs, y_train_lhs, X_valid_lhs, y_valid_lhs, rel_err_train_lhs, rel_err_valid_lhs, reconstruction_relative_errors_lhs = parametric_rom_lhs.OnlinePhase(
    fraction_train=0.05,
    T_end_test=onlinephase_args[0],
    IC_predict=onlinephase_args[1],
    epochs=35000,
    verb=False,
    fnn_depth=3,
    fnn_width=32)

mean_error_train_lhs, mean_error_test_lhs = parametric_rom_active.TestPhase(num_samples_test=800,
                                                                            interp_model=fnn_lhs,
                                                                            x_train=X_train_lhs, y_train=y_train_lhs,
                                                                            reconstruction_relative_errors=reconstruction_relative_errors_lhs,
                                                                            visuals=True,
                                                                            directory_1d=f"/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results_Alternative/Param1D/Active_Learning/LHS/")

labels = ['Active Learning', 'LHS']
train_means = [mean_error_train_al, mean_error_train_lhs]  # Example values
test_means = [mean_error_test_al,  mean_error_test_lhs]   # Example values


x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots()

# Plotting the bars
rects1 = ax.bar(x - width/2, train_means, width, label='Train', color='blue', alpha=0.7)
rects2 = ax.bar(x + width/2, test_means, width, label='Test', color='red', alpha=0.7)

# Adding some text for labels, title and custom x-axis tick labels, etc.
ax.set_yscale('log')
ax.set_ylabel(r'Mean $L_2$ relative error')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Function to add a y-axis label above the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig("/Users/konstantinoskevopoulos/Desktop/ActiveLearning_Results/Lotka_Volterra/After_al_plots/AL_VS_LHS.png")


