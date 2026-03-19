from pLANDO_HighDim import *

test_t_instances = [0.17, 0.75, 0.85]
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
svd_truncation = 15
verb = True


parametric_lando = ParametricLANDO(kernel=kernel, horizon_train=t_train, dt=dt, sparsity_tol=nu_sparsity,
                                   directory_processed_data="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                            "/SnapshotData",
                                   directory_samples="/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis"
                                                     "/Parameter_Samples",
                                   training_frac=fraction_train, validation_frac=fraction_validation,
                                   problem="ac", samples_filename="/samples_allen_cahn.txt")

_, sparse_dicts = parametric_lando.OfflinePhase()

errors = []
for t in test_t_instances:
    error_results = parametric_lando.OnlinePhase(T_end_test=t,
                                                 fnn_depth=depth,
                                                 fnn_width=width,
                                                 epochs=epochs,
                                                 batch_size=batch,
                                                 trunc_rank=svd_truncation,
                                                 verbose=verb)

    errors.append(error_results[1])

np.save('mean_relative_errors_allen_cahn.npy', np.array(errors))

print(f"t^* = 0.17: Mean relative error {errors[0]}\n"
      f"t^* = 0.75: Mean relative error {errors[1]}\n"
      f"t^* = 0.85: Mean relative error {errors[2]}")

