from Lotka_Volterra.Lotka_Volterra_model import *
from tqdm import tqdm
import time
from DMD import *


def relative_error(y_test, prediction):
    err_list = []
    for row in range(y_test.shape[0]):

        err = np.linalg.norm(y_test[row] - prediction[row]) / np.linalg.norm(y_test[row])

        err_list.append(err)

    return np.mean(err_list), np.std(err_list)


seed = 42
np.random.seed(42)

T_all = 500
NumSamples_Train = 400
NumSamples_Test = 500

num_sensors = 1000

dt = num_sensors / T_all
T_train = 400

Init_Condition = [80, 20]

t_test_times = [300, 450, 500]

param_samples = np.load('../Table_Experiments/LV_1D_param_samples.npy')

### Simulate 'high-fidelity' data for the LV system
X_all = np.empty((NumSamples_Train+NumSamples_Test, 2, num_sensors))
pbar = tqdm(total=NumSamples_Train + NumSamples_Test, desc='Generating HF data for Lotka-Volterra model', leave=True)
for i, param_sample in enumerate(param_samples):
    X, _ = Lotka_Volterra_Snapshot(params=param_sample, T=T_train, num_sensors=num_sensors)
    X_all[i] = X

    pbar.update()
pbar.close()

training_snapshots = X_all[:NumSamples_Train, :, :int(T_train*dt)]
training_param_samples = param_samples[:NumSamples_Train, 0].reshape(-1, 1)

t300_snapshot_test = X_all[NumSamples_Train:, :, int(300 * dt)]
t450_snapshot_test = X_all[NumSamples_Train:, :, int(450 * dt)]
t500_snapshot_test = X_all[NumSamples_Train:, :, -1]

errors_mean = []
errors_std = []

### pDMD training
start_train = time.time()

dmd = DMD(svd_rank=-1)
rom = POD(rank=2)
interpolator = RBF()

dmds = [DMD(svd_rank=-1) for _ in range(training_param_samples.shape[0])]
pdmd_partitioned = ParametricDMD(dmds, rom, interpolator)

pdmd_partitioned.fit(training_snapshots, training_param_samples)

end_train = time.time()

### Test pDMD
start_test = time.time()

pdmd_partitioned.parameters = param_samples[NumSamples_Train:, 0].reshape(-1, 1)

pdmd_partitioned.dmd_time['t0'] = pdmd_partitioned.original_time['t0']
pdmd_partitioned.dmd_time['tend'] = int(500 * dt)

pdmd_result = pdmd_partitioned.reconstructed_data

end_test = time.time()

training_time = end_train - start_train
testing_time = end_test - start_test

print('DONE', pdmd_result.shape)
print(f'Training time per sample: {training_time / NumSamples_Train}')
print(f'Testing time per sample: {testing_time / NumSamples_Test}')

t300_snapshot_pred = pdmd_result[:, :, int(dt * 300) + 1]
t450_snapshot_pred = pdmd_result[:, :, int(dt * 450) + 1]
t500_snapshot_pred = pdmd_result[:, :, -1]


mean_rel_err_300, std_rel_err_300 = relative_error(y_test=t300_snapshot_test, prediction=t300_snapshot_pred)
print(f't^* = 300', mean_rel_err_300)

errors_mean.append(mean_rel_err_300)
errors_std.append(std_rel_err_300)


mean_rel_err_450, std_rel_err_450 = relative_error(y_test=t450_snapshot_test, prediction=t450_snapshot_pred)
print(f't^* = 450:', mean_rel_err_450)

errors_mean.append(mean_rel_err_450)
errors_std.append(std_rel_err_450)


mean_rel_err_500, std_rel_err_500 = relative_error(y_test=t500_snapshot_test, prediction=t500_snapshot_pred)
print(f't^* = 500:', mean_rel_err_500)

errors_mean.append(mean_rel_err_500)
errors_std.append(std_rel_err_500)

np.save('LV_rel_errs_mean.npy', errors_mean)
np.save('LV_rel_errs_std.npy', errors_std)


