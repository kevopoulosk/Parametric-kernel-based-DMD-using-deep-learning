from Parametric_LANDO.pLANDO_LowDim import *
from Lotka_Volterra_model import *
from Lotka_Volterra_Deriv import *


def relative_error(y_test, prediction):
    err_list = []
    for row in range(y_test.shape[0]):

        err = np.linalg.norm(y_test[row] - prediction[row]) / np.linalg.norm(y_test[row])

        err_list.append(err)

    return np.mean(err_list)


param_sample = np.array([0.056, 0.002, 0.2, 0.0025])

init_condition = [80, 20]
Tend = 500
num_sensors = 600
kernel = quadratic_kernel

### Sparsity values, for which we perform the experiment
nus_sparsity = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


### Compute snapshot matrix
X, _ = Lotka_Volterra_Snapshot(params=param_sample, T=Tend, num_sensors=num_sensors)
Y = Lotka_Volterra_Deriv(X, *param_sample)

scaledX = Scale(X)

Xperm, perm = Permute(X)
Yperm = Y[:, perm]

mean_rel_errs = []
num_snapshots_sparse = []
for nu in nus_sparsity:

    SparseDict = SparseDictionary(Xperm, scaledX, kernel, tolerance=nu)
    W_tilde = Yperm @ np.linalg.pinv(kernel(SparseDict, scaledX * Xperm))

    kernel_compute = kernel(SparseDict, scaledX * X)

    model = W_tilde @ kernel_compute

    mean_rel_error = relative_error(model, Y)

    mean_rel_errs.append(mean_rel_error)
    num_snapshots_sparse.append(kernel_compute.shape[0])


colors = ['tab:blue', 'tab:red', 'tab:green']
params = {
            'axes.labelsize': 15.4,
            'font.size': 15.4,
            'legend.fontsize': 15.4,
            'xtick.labelsize': 15.4,
            'ytick.labelsize': 15.4,
            'text.usetex': False,
            'axes.linewidth': 2,
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            'xtick.major.size': 2,
            'ytick.major.size': 2,
        }
plt.rcParams.update(params)

fig, ax1 = plt.subplots()

# Plot the first list with a blue line and left y-axis
ax1.loglog(nus_sparsity, mean_rel_errs, '-o', color='blue')
ax1.set_ylabel(r'$\frac{\|\mathbf{Y}_{ref} - \mathbf{Y}_{pred}\|_{L_2}}{\|\mathbf{Y}_{ref}\|_{L_2}}$', color='blue')
ax1.set_xlabel(r"$\nu$ Sparsity coefficient")
ax1.grid()
ax1.tick_params(axis='y', labelcolor='blue')


# Create a second y-axis for the red line
ax2 = ax1.twinx()
ax2.semilogx(nus_sparsity, num_snapshots_sparse, '-o', color='red')
ax2.set_ylabel('Dictionary size', color='red')
ax2.tick_params(axis='y', labelcolor='red')
plt.show()