from Parametric_LANDO import *
import os

### This file performs the experiments - results for the thesis research project

T_train = 400
NumSamples_Train = 600
sensors = 600
sparsity_lando = 1e-6

NumSamples_Test = 500
train_frac = 0.8
depth = 3
width = 32
epochs = 50000

varied = 1
low_bounds = [0.01]
upp_bounds = [0.1]
fixed_params = np.array([0.002, 0.2, 0.0025])

T_tests = [150, 300, 450, 600]
Init_Conditions = [[80, 20], [70, 13], [53, 13]]

save_directory = "/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results/Param1D"

parametric_lando = ParametricLANDO(kernel=quadratic_kernel, horizon_train=T_train, num_samples_train=NumSamples_Train,
                                   num_sensors=sensors, sparsity_tol=sparsity_lando, batch_frac=0.2,
                                   params_varied=varied, low_bound=low_bounds, upp_bound=upp_bounds,
                                   fixed_params_val=fixed_params)

### First, the offline phase of the parametrization algorithm is performed
w_tildes, sparse_dicts, mu_samples_train = parametric_lando.OfflinePhase()

dict_errors = {}
pbar = tqdm(total=len(Init_Conditions), desc="Progress of experiments")
for init in Init_Conditions:
    dict_errors[f"IC={init}"] = []
    for t in T_tests:
        ### Make the directory in computer to save the plots
        try:
            os.mkdir(save_directory + f'/IC={init}, t_end={t}/')
        except:
            print('Sth went wrong, please check')

        mean_error_test = parametric_lando.OnlinePhase(num_samples_test=NumSamples_Test,
                                                       T_end_test=t,
                                                       fraction_train=train_frac,
                                                       fnn_depth=depth, fnn_width=width,
                                                       epochs=epochs, IC_predict=init, verb=False)

        dict_errors[f"IC={init}"].append(mean_error_test)

    pbar.update()
pbar.close()

print(f"The error dictionary is {dict_errors}")

### Visualise the different errors
initial_conditions = list(dict_errors.keys())
errors = list(dict_errors.values())

t = [150, 300, 450, 600]
for i in range(len(initial_conditions)):
    plt.semilogy(t, errors[i], '-o', label=initial_conditions[i])

plt.xlabel(r'$t^{*}$')
plt.ylabel(r'Mean $L_2$ relative error')
plt.grid(True)
plt.legend()
plt.show()
