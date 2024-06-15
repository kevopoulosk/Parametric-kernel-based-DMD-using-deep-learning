from Active_Learning.Active_Learning_HighDim import *


n_init = 40
n_all = 55
dim_init = 2
num_nets = 7
low_init = [0.0001, 0.5]
upp_init = [0.001, 4]
num_valid = 500

validation_samples = LatinHypercube(dim_init, low_init, upp_init, num_valid)
file_path_data = os.path.join("/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/Active_Learning_Samples/",
                              "samples_allen_cahn_al.txt")
np.savetxt(file_path_data, validation_samples)

plando = ParametricLANDO(kernel=linear_kernel, horizon_train=0.6, dt=0.001, sparsity_tol=1e-6,
                         directory_processed_data="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                  "/Active_Learning_Data",
                         directory_samples="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                           "/Active_Learning_Samples", training_frac=None,
                         validation_frac=None, problem="ac",
                         samples_filename="/samples_allen_cahn_al.txt", active=True)

depths = [4 for _ in range(num_nets)]
widths = [110 for _ in range(num_nets)]
activations = ["relu" for _ in range(num_nets)]
onlinephase_args = [0.75, 50, 4, 110, 0.33, 15000, None, False]
ensemble_args = [num_nets, dim_init, 50, depths, widths, activations, 15000, 0.33]

### Obtain the "most informative" training dataset
train_dataset = ActiveLearning_Algorithm(n_init, plando, n_all, dim_init, low_init, upp_init,
                                         num_valid, onlinephase_args, ensemble_args)

file_path_data = os.path.join("/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/AL_Samples_After/",
                              "samples_allen_cahn_al.txt")
np.savetxt(file_path_data, validation_samples)


###Now we run pLANDO with the optimal training dataset
parametric_rom_active = ParametricLANDO(kernel=linear_kernel, horizon_train=0.6, dt=0.001, sparsity_tol=1e-6,
                                        directory_processed_data="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                                 "/AL_Data_After",
                                        directory_samples="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                          "/AL_Samples_After", training_frac=None,
                                        validation_frac=None, problem="ac",
                                        samples_filename="/samples_allen_cahn_al.txt", active=True)

parametric_rom_active.OfflinePhase(samples_train_al=train_dataset, samples_valid_al=validation_samples)

X_train, y_train, X_valid, y_valid, mean_error_train_al, mean_error_test_al = parametric_rom_active.OnlinePhase(
                                                                                T_end_test=0.75,
                                                                                trunk_rank=50,
                                                                                fnn_depth=4,
                                                                                fnn_width=110,
                                                                                batch_size=0.33,
                                                                                epochs=20000,
                                                                                directory_save=None)

########################################################################################################################
### Perform pLANDO with LHS sampling
parametric_rom_lhs = ParametricLANDO(kernel=linear_kernel, horizon_train=0.6, dt=0.001, sparsity_tol=1e-6,
                                     directory_processed_data="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                              "/SnapshotData",
                                     directory_samples="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                       "/Parameter_Samples", training_frac=0.055,
                                     validation_frac=0.47, problem="ac",
                                     samples_filename="/samples_allen_cahn.txt", active=False)

parametric_rom_lhs.OfflinePhase(samples_train_al=None, samples_valid_al=None)

results_online_lhs = parametric_rom_lhs.OnlinePhase(T_end_test=0.75,
                                                    fnn_depth=4,
                                                    fnn_width=110,
                                                    epochs=20000,
                                                    batch_size=0.33,
                                                    trunk_rank=50,
                                                    verbose=True,
                                                    directory_save="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                                   "/Active_Learning_Results")

mean_error_train_lhs = results_online_lhs[0]
mean_error_test_lhs = results_online_lhs[1]

### Visualise training and test performance Active Learning Vs LHS

labels = ['Active Learning', 'LHS']
train_means = [mean_error_train_al, mean_error_train_lhs]
test_means = [mean_error_test_al, mean_error_test_lhs]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()

rects1 = ax.bar(x - width / 2, train_means, width, label='Train', color='blue', alpha=0.7)
rects2 = ax.bar(x + width / 2, test_means, width, label='Test', color='red', alpha=0.7)

ax.set_yscale('log')
ax.set_ylabel(r'Mean $L_2$ relative error')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


# Function to add a y-axis label above the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.6f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.savefig("/Users/konstantinoskevopoulos/Desktop/ActiveLearning_Results/Allen_Cahn/After_al_plots/AL_VS_LHS.png")
