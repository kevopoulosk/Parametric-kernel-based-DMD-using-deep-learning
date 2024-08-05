from Sparse_Dictionary_Learning.pLANDO_LowDim import *
import os
import pickle

### This file performs the experiments - results for the thesis research project

T_train = 400
NumSamples_Train = 700
sensors = 600
sparsity_lando = 1e-6

NumSamples_Test = 1200
train_frac = 0.8
depth = 2
width = 30
epochs = 35000
rbf = False

varied = 2
low_bounds = [0.015, 0.0012]
upp_bounds = [0.12, 0.0022]
fixed_params = np.array([0.2, 0.0025])

T_tests = np.arange(50, 650, 50)
Init_Conditions = [[70, 20], [80, 20]]

save_directory = "/Users/konstantinoskevopoulos/Desktop/Lotka_Volterra_Results/Param2D"

parametric_lando = ParametricLANDO(kernel=quadratic_kernel, horizon_train=T_train, num_samples_train=NumSamples_Train,
                                   num_sensors=sensors, sparsity_tol=sparsity_lando, batch_frac=0.2,
                                   params_varied=varied, low_bound=low_bounds, upp_bound=upp_bounds,
                                   fixed_params_val=fixed_params, rbf=rbf)

### First, the offline phase of the parametrization algorithm is performed
w_tildes, sparse_dicts, mu_samples_train = parametric_lando.OfflinePhase()

dict_errors = {}
dict_errors_general = {}
pbar = tqdm(total=len(Init_Conditions), desc="Progress of experiments")
for init in Init_Conditions:
    dict_errors[f"IC={init}"] = []
    dict_errors_general[f"IC={init}"] = []

    for t in T_tests:
        ### Make the directory in computer to save the plots
        try:
            os.mkdir(save_directory + f'/IC={init}, t_end={t}/')
        except:
            print('Sth went wrong, please check')

        mean_error_train, mean_error_test = parametric_lando.OnlinePhase(T_end_test=t,
                                                                         fraction_train=train_frac,
                                                                         fnn_depth=depth, fnn_width=width,
                                                                         epochs=epochs, IC_predict=init, verb=False)

        dict_errors[f"IC={init}"].append(mean_error_test)
        dict_errors_general[f"IC={init}"].append([mean_error_train, mean_error_test])

    pbar.update()
pbar.close()

print(f"The error dictionary is {dict_errors}")

if rbf:
    with open('errors_dict_2D_RBF.pkl', 'wb') as f:
        pickle.dump(dict_errors_general, f)
else:
    with open('errors_dict_2D_NN.pkl', 'wb') as f:
        pickle.dump(dict_errors_general, f)

### Visualise the different errors
initial_conditions = list(dict_errors.keys())
errors = list(dict_errors.values())


for i in range(len(initial_conditions)):
    plt.semilogy(T_tests, errors[i], '-o', label=initial_conditions[i])

plt.xlabel(r'$t^{*}$')
plt.axvline(x=400, linestyle='--', color='black')
plt.ylabel(r'Mean $L_2$ relative error')
plt.grid(True)
plt.legend()

if rbf:
    filename = "/errors_semilogy_RBF.png"
else:
    filename = "/errors_semilogy_NN.png"

plt.savefig(save_directory + filename)
plt.show()

plt.figure()
for i in range(len(initial_conditions)):
    plt.plot(T_tests, errors[i], '-o', label=initial_conditions[i])

plt.xlabel(r'$t^{*}$')
plt.axvline(x=400, linestyle='--', color='black')
plt.ylabel(r'Mean $L_2$ relative error')
plt.grid(True)
plt.legend()

if rbf:
    filename = "/errors_RBF.png"
else:
    filename = "/errors_NN.png"

plt.savefig(save_directory + filename)
plt.show()
