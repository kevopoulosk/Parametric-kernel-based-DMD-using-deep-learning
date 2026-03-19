from DMD import *
from tqdm import tqdm
import time

seed = 42
np.random.seed(42)


def relative_error(y_test, prediction):
    err_list = []
    for row in range(y_test.shape[0]):

        err = np.linalg.norm(y_test[row] - prediction[row]) / np.linalg.norm(y_test[row])

        err_list.append(err)

    return np.mean(err_list), np.std(err_list)


directory_data = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/SnapshotData_Processed"
directory_samples = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/Parameter_Samples"
samples_filename = "/samples_heat_eq.txt"

num_samples = 300
t_all = 4

dt = 0.01
t_train = 2

train_horizon = int(t_train / dt)

fraction_train = 0.5
num_train_samples = int(num_samples*fraction_train)

X_all = np.empty((num_samples, 4437, int(t_all / dt)))

pbar = tqdm(total=num_samples, desc='Loading samples...', leave=True)
for i in range(num_samples):
    X = np.load(directory_data + f"/sample{i}.npy")
    X_all[i] = X
    pbar.update()
pbar.close()

param_samples = np.loadtxt(directory_samples + samples_filename).reshape(-1, 1)

training_snapshots = X_all[:num_train_samples, :, :train_horizon]
training_param_samples = param_samples[:num_train_samples]

t015_snapshot_test = X_all[num_train_samples:, :, 15]
t050_snapshot_test = X_all[num_train_samples:, :, 50]
t400_snapshot_test = X_all[num_train_samples:, :, -1]

ranks_pod = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

rel_errs_t015_mean = []
rel_errs_t015_std = []
rel_errs_t050_mean = []
rel_errs_t050_std = []
rel_errs_t400_mean = []
rel_errs_t400_std = []
for rank_pod in ranks_pod:
    start_train = time.time()

    dmd = DMD(svd_rank=-1)
    rom = POD(rank=rank_pod)
    interpolator = RBF()

    dmds = [DMD(svd_rank=-1) for _ in range(training_param_samples.shape[0])]
    pdmd_partitioned = ParametricDMD(dmds, rom, interpolator)

    pdmd_partitioned.fit(training_snapshots, training_param_samples)

    end_train = time.time()

    start_test = time.time()

    pdmd_partitioned.parameters = param_samples[num_train_samples:]

    pdmd_partitioned.dmd_time['t0'] = pdmd_partitioned.original_time['t0']
    pdmd_partitioned.dmd_time['tend'] = 400

    pdmd_result = pdmd_partitioned.reconstructed_data

    end_test = time.time()

    training_time = end_train - start_train
    testing_time = end_test - start_test

    print('DONE', pdmd_result.shape)
    print(f'Training time per sample: {training_time / num_samples}')
    print(f'Testing time per sample: {testing_time / num_samples}')

    t015_snapshot_pred = pdmd_result[:, :, 16]
    t050_snapshot_pred = pdmd_result[:, :, 51]
    t400_snapshot_pred = pdmd_result[:, :, -1]

    mean_rel_err_015, std_rel_err_015 = relative_error(y_test=t015_snapshot_test, prediction=t015_snapshot_pred)
    print(f'Rank: {rank_pod} --> t^* = 0.15', mean_rel_err_015)

    rel_errs_t015_mean.append(mean_rel_err_015)
    rel_errs_t015_std.append(std_rel_err_015)

    mean_rel_err_050, std_rel_err050 = relative_error(y_test=t050_snapshot_test, prediction=t050_snapshot_pred)
    print(f'Rank: {rank_pod} --> t^* = 0.50:', mean_rel_err_050)
    rel_errs_t050_mean.append(mean_rel_err_050)
    rel_errs_t050_std.append(std_rel_err050)

    mean_rel_err_400, std_rel_err_400 = relative_error(y_test=t400_snapshot_test, prediction=t400_snapshot_pred)
    print(f'Rank: {rank_pod} --> t^* = 400:', mean_rel_err_400)
    rel_errs_t400_mean.append(mean_rel_err_400)
    rel_errs_t400_std.append(std_rel_err_400)


np.save('HEAT_rel_errs_t015_mean.npy', np.array(rel_errs_t015_mean))
np.save('HEAT_rel_errs_t015_std.npy', np.array(rel_errs_t015_std))

np.save('HEAT_rel_errs_t050_mean.npy', np.array(rel_errs_t050_mean))
np.save('HEAT_rel_errs_t050_std.npy', np.array(rel_errs_t050_std))

np.save('HEAT_rel_errs_t400_mean.npy', np.array(rel_errs_t400_mean))
np.save('HEAT_rel_errs_t400_std.npy', np.array(rel_errs_t400_std))

