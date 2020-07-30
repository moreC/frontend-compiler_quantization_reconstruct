import logging

import numpy as np
from numba import njit

from fast_pruning.pruning.solver_base import SolverBase

logger = logging.getLogger()


class GDPruneSolver(SolverBase):
    def __init__(self, backend='cpp', sparsity=0.1, num_workers=8):
        super().__init__(num_workers=num_workers)
        self.backend = backend
        self.sparsity = sparsity
        self.warm_up_iterations = 100
        self.total_iterations = 1000
        self.pruning_freq = 100
        self.lr = 0.001

    def solve(self, xtx: np.array, xty: np.array, yty: np.array):
        logger.info('solve w with XTX shape: {}, XTy shape: {}, yty shape: {}'.format(xtx.shape, xty.shape, yty.shape))

        if self.backend == 'cpp':
            raise NotImplementedError('cpp for this solver is not implemented yet')
        elif self.backend == 'python':
            w = np.vstack(
                self.parallel_run(_py_solve_single, [{
                    'G': xtx,
                    'b': xty[:, i],
                    'sparsity': self.sparsity,
                    'warm_up_iterations': self.warm_up_iterations,
                    'total_iterations': self.total_iterations,
                    'pruning_freq': self.pruning_freq,
                    'lr': self.lr
                } for i in range(yty.shape[0])]))
        else:
            raise ValueError('unknown backend: {}'.format(self.backend))
        return w.T


@njit(parallel=False, fastmath=True, nogil=True)
def _py_solve_single(G, b, sparsity, warm_up_iterations, total_iterations, pruning_freq, lr):
    """
    solve 1/2 * w G^T w + b^T w + reg |w| problem with greedy coordinate descent

    Parameters
    ----------
    G: 2-d matrix with X^T X

    b: 1-d matrix with X^T y
    """

    w = np.zeros(b.shape, np.float32)

    for i in range(1, warm_up_iterations + 1):
        w -= (np.dot(G, w) - b) * lr

    mask = np.ones(b.shape, np.float32)
    sparsity = np.array(sparsity).astype(np.float32)
    lr = np.array(lr).astype(np.float32)
    for i in range(1, total_iterations + pruning_freq + 1):

        if i % pruning_freq == 0 and i <= total_iterations:
            current_sparsity = sparsity + (0.0 - sparsity) * (1.0 - (i - 0) / (total_iterations - 0))**3
            abs_w = np.abs(w)
            thr = np.quantile(abs_w, current_sparsity)
            mask = np.abs(abs_w >= thr).astype(np.float32)
            w = w * mask
        w -= (np.dot(G, w) - b) * lr * mask

    return w
