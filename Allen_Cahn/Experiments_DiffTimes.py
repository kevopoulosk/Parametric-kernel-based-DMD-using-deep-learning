from Sparse_Dictionary_Learning.pLANDO_HighDim import *
import pickle

test_t_instances = [0.05, 0.17, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.97]
t_train = 0.6
dt = 0.001
kernel = linear_kernel
nu_sparsity = 1e-6
fraction_train = 0.4
fraction_validation = 0.15

depth = 4
width = 110
epochs = 20000
batch = 0.33
svd_truncation = 50
verb = True

save_directory = "/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"

parametric_lando = ParametricLANDO(kernel=kernel, horizon_train=t_train, dt=dt, sparsity_tol=nu_sparsity,
                                   directory_processed_data="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                            "/SnapshotData",
                                   directory_samples="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                     "/Parameter_Samples",
                                   training_frac=fraction_train, validation_frac=fraction_validation,
                                   problem="ac", samples_filename="/samples_allen_cahn.txt")

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
                                                 verbose=verb,
                                                 directory_save=save_directory+f'/t_test={t}/')

    dict_errors[f"t = {t}"].append(error_results)
    pbar.update()
pbar.close()


### Save data from this run into dictionary form
with open('errors_dict.pkl', 'wb') as f:
    pickle.dump(dict_errors, f)
