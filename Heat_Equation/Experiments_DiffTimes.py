from HighDim_pLANDO import *
import pickle

test_t_instances = [0.15, 0.45, 1, 1.5, 2, 2.5, 3, 3.5, 4]
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
svd_truncation = 50
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

    ### Make the directory in computer to save the plots
    try:
        os.mkdir(save_directory + f'/t_test={t}/')
    except:
        print('Sth went wrong, please check')

    error_results = parametric_lando.OnlinePhase(T_end_test=t,
                                                 fnn_depth=depth,
                                                 fnn_width=width,
                                                 epochs=epochs,
                                                 batch_size=batch,
                                                 trunk_rank=svd_truncation,
                                                 verbose=verb)

    dict_errors[f"t = {t}"].append(error_results)
    pbar.update()
pbar.close()


### Save data from this run into dictionary form
with open('errors_dict.pkl', 'wb') as f:
    pickle.dump(dict_errors, f)
