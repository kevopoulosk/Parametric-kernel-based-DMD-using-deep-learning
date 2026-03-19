from pLANDO_HighDim import *

test_t_instances = [0.15, 3, 4]
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
verb = True


parametric_lando = ParametricLANDO(kernel=kernel, horizon_train=t_train, dt=dt, sparsity_tol=nu_sparsity,
                                   directory_processed_data="/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis"
                                                            "/SnapshotData_Processed",
                                   directory_samples="/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis"
                                                     "/Parameter_Samples",
                                   training_frac=fraction_train, validation_frac=fraction_validation)

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

    mean_relative_test_error = error_results[1]
    errors.append(mean_relative_test_error)

np.save('mean_relative_errors_heat_eq.npy', np.array(errors))

print(f"t^* = 0.15: Mean relative error {errors[0]}\n"
      f"t^* = 3: Mean relative error {errors[1]}\n"
      f"t^* = 4: Mean relative error {errors[2]}")


