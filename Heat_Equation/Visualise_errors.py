import numpy as np
import matplotlib.pyplot as plt
import pickle

t_test = [0.15, 0.45, 1, 1.5, 2, 2.5, 3, 3.5, 4]

with open(f'errors_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

test_error_mean = []
test_error_std = []
for t in t_test:
    test_error_mean.append(loaded_dict[f"t = {t}"][0][1])
    test_error_std.append(loaded_dict[f"t = {t}"][0][3])

plt.figure()
plt.errorbar(t_test, test_error_mean, yerr=test_error_std, marker='o', capsize=5)
plt.grid(True)
plt.ylabel(r"Mean $L_2$ Relative Error")
plt.xlabel(r"t")
plt.yscale("log")
plt.axvline(x=2, linestyle='--', color='black')
plt.show()

plt.figure()
plt.semilogy(loaded_dict['t = 3.5'][0][-2], 'o')
plt.xlabel("Parameter Index")
plt.ylabel(r"Mean $L_2$ Relative Error")
plt.grid(True)
plt.show()

plt.figure()
plt.semilogy(loaded_dict['t = 3.5'][0][-1], 'o')
plt.xlabel("Parameter Index")
plt.ylabel(r"Mean $L_2$ Relative Error")
plt.grid(True)
plt.show()


