import numpy as np
from pydmd import ParametricDMD, DMD
from ezyrb import POD, RBF


class pDMD:
    def __init__(self, training_snapshots:np.ndarray, training_params: np.ndarray, truncation_pod: int):

        ### 1D problem training snapshots:
        ### n_train * n_space * n_time_instants
        ### e.g (10, 500, 160)
        ### number of parameters, space_discretization, time_discretization

        ### MULTIDIM snapshots:
        ### flatten snapshots before fit --> become 1D
        ### n_train *(n_space1 * n_space2 * ...) * n_time_instants
        ### e.g (10, 4000, time_instants) ---> snapshots must be flattened


        ### training params has shape (n_train, n_params)

        self.training_snapshots = training_snapshots
        self.training_params = training_params

        self.dmds = [DMD(svd_rank=-1) for _ in range(len(training_params))]
        self.rom = POD(rank=truncation_pod)
        self.interpolator = RBF()

        self.pdmds_partitioned = ParametricDMD(self.dmds, self.rom, self.interpolator)
        print('dmd init ok')

    def train(self):
        ### Train the model
        self.pdmds_partitioned.fit(self.training_snapshots, self.training_params)
        print('train ok')

    def test(self, testing_params):
        self.pdmds_partitioned.parameters = testing_params

        result_out = self.pdmds_partitioned.reconstructed_data
        print('prediction ok')

        return result_out