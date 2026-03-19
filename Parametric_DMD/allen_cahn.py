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


directory_data = "/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/SnapshotData"
directory_samples = "/Users/konstantinoskevopoulos/Documents/Allen_Cahn_Thesis/Parameter_Samples"
samples_filename = "/samples_allen_cahn.txt"

num_samples = 1000
t_all = 1001

dt = 0.001
t_train = 0.6

train_horizon = int(t_train / dt)

fraction_train = 0.4
num_train_samples = int(num_samples * fraction_train)

X_all = np.empty((num_samples, 250, 1001))

pbar = tqdm(total=num_samples, desc='Loading samples...', leave=True)
for i in range(num_samples):
    X = np.load(directory_data + f"/sample{i}.npy")
    X_all[i] = X
    pbar.update()
pbar.close()

param_samples = np.loadtxt(directory_samples + samples_filename)

training_snapshots = X_all[:num_train_samples, :, :train_horizon]
training_param_samples = param_samples[:num_train_samples]

t017_snapshot_test = X_all[num_train_samples:, :, 171]
t075_snapshot_test = X_all[num_train_samples:, :, 751]
t085_snapshot_test = X_all[num_train_samples:, :, 851]

ranks_pod = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

rel_errs_t017_mean = []
rel_errs_t017_std = []
rel_errs_t075_mean = []
rel_errs_t075_std = []
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
    pdmd_partitioned.dmd_time['tend'] = 850

    pdmd_result = pdmd_partitioned.reconstructed_data

    end_test = time.time()

    training_time = end_train - start_train
    testing_time = end_test - start_test

    print('DONE', pdmd_result.shape)
    print(f'Training time per sample: {training_time/num_samples}')
    print(f'Testing time per sample: {testing_time/ num_samples}')

    t017_snapshot_pred = pdmd_result[:, :, 171]
    t075_snapshot_pred = pdmd_result[:, :, 751]
    t085_snapshot_pred = pdmd_result[:, :, -1]

    mean_rel_err_017, std_rel_err_017 = relative_error(y_test=t017_snapshot_test, prediction=t017_snapshot_pred)
    print(f'Rank: {rank_pod} --> t^* = 0.17:', mean_rel_err_017)

    rel_errs_t017_mean.append(mean_rel_err_017)
    rel_errs_t017_std.append(std_rel_err_017)

    mean_rel_err_075, std_rel_err075 = relative_error(y_test=t075_snapshot_test, prediction=t075_snapshot_pred)
    print(f'Rank: {rank_pod} --> t^* = 0.75:', mean_rel_err_075)
    rel_errs_t075_mean.append(mean_rel_err_075)
    rel_errs_t075_std.append(std_rel_err075)

    mean_rel_err_085, std_rel_err_085 = relative_error(y_test=t085_snapshot_test, prediction=t085_snapshot_pred)
    print(f'Rank: {rank_pod} --> t^* = 0.85:', mean_rel_err_085)

np.save('AC_rel_errs_t017_mean.npy', np.array(rel_errs_t017_mean))
np.save('AC_rel_errs_t017_std.npy', np.array(rel_errs_t017_std))

np.save('AC_rel_errs_t075_mean.npy', np.array(rel_errs_t075_mean))
np.save('AC_rel_errs_t075_std.npy', np.array(rel_errs_t075_std))

### Don't save for t = 0.85 because the error blows up


