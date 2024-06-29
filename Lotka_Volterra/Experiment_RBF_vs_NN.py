import numpy as np
import pickle
import matplotlib.pyplot as plt


def RBF_vs_NN_Visual(train_instances, test_instances):
    """
    Function that visualizes the results of the generalization and test performance of RBF and NN
    :param train_instances:
    :param test_instances:
    :return:
    """
    labels = ("RBF\nIC=[70, 20]", "NN\nIC=[70, 20]", "RBF\nIC=[80, 20]", "NN\nIC=[80, 20]")
    means = {
        'Train': (train_instances),
        'Test': (test_instances),
    }

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()

    for attribute, measurement in means.items():
        offset = width * multiplier
        if attribute == 'Test':
            rects = ax.bar(x + offset, measurement, width, label=attribute, color='red')
        else:
            rects = ax.bar(x + offset, measurement, width, label=attribute)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'Mean $L_2$ relative error')
    ax.set_yscale('log')
    ax.set_xticks(x + width * (len(means) - 1) / 2, labels)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncols=2)

    plt.show()


cases = ["2D"]
interp = ["RBF", "NN"]
IC = [[70, 20], [80, 20]]


for case in cases:
    errors_train = []
    errors_test = []
    for init in IC:
        for interp_method in interp:
            with open(f'errors_dict_{case}_{interp_method}.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
                errors_train.append(loaded_dict[f"IC={init}"][-5][0])
                errors_test.append(loaded_dict[f"IC={init}"][-5][1])

    RBF_vs_NN_Visual(train_instances=errors_train, test_instances=errors_test)
