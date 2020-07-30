import logging

import numpy as np
from numba import njit

from fast_pruning.pruning.solver_base import SolverBase
from pruning import eigen_parallel_inc_solve, eigen_cross_channel_inc_solve

logger = logging.getLogger()


class GCDIncSolver(SolverBase):
    def __init__(self, backend='cpp', sparsity=0.1, tol=1e-6, finetune_freq=1, num_workers=8, cross_channel=False):
        super().__init__(num_workers=num_workers)
        self.backend = backend
        self.sparsity = sparsity
        self.tol = tol
        self.finetune_freq = finetune_freq
        self.cross_channel = cross_channel

    def solve(self, xtx: np.array, xty: np.array, yty: np.array, has_bias: bool):
        logger.info('solve w with XTX shape: {}, XTy shape: {}, yty shape: {}'.format(xtx.shape, xty.shape, yty.shape))

        if self.backend == 'cpp':
            if self.cross_channel:
                w = eigen_cross_channel_inc_solve(number_of_threads=self.num_workers,
                                                  G=xtx,
                                                  B=xty,
                                                  sparsity=self.sparsity,
                                                  has_bias=has_bias,
                                                  finetune_freq=self.finetune_freq).T
            else:
                w = eigen_parallel_inc_solve(number_of_threads=self.num_workers,
                                             G=xtx,
                                             B=xty,
                                             sparsity=self.sparsity,
                                             has_bias=has_bias,
                                             finetune_freq=self.finetune_freq).T
        elif self.backend == 'python':
            if self.cross_channel:
                raise ValueError('cross_channel not supported for python backend')
            else:
                w = np.vstack(
                    self.parallel_run(_py_solve_single, [{
                        'G': xtx,
                        'b': xty[:, i],
                        'sparsity': self.sparsity,
                        'tol': self.tol,
                        'has_bias': has_bias,
                        'finetune_freq': self.finetune_freq
                    } for i in range(yty.shape[0])]))
        else:
            raise ValueError('unknown backend: {}'.format(self.backend))
        return w.T


@njit(parallel=False, fastmath=True, nogil=True)
def _get_init_w_g(G, b):
    w = np.zeros(b.shape)
    g = -b
    # g = np.dot(G, w) - b
    return w, g


@njit(parallel=False, fastmath=True, nogil=True)
def _gcd_step(G, w, g, target_index):
    ori_wi = float(w[target_index])
    w[target_index] -= g[target_index] / G[target_index, target_index]
    delta = (w[target_index] - ori_wi)
    g += delta * G[:, target_index]
    return delta


@njit(parallel=False, fastmath=True, nogil=True)
def sparse_finetuning(G, w, g):
    for i in range(w.size * 10):
        delta = _gcd_step(G=G, w=w, g=g, target_index=np.argmax(np.abs(g) * (np.abs(w) > 0).astype(np.float64)))
        if np.abs(delta) <= 1e-5:
            break


@njit(parallel=False, fastmath=True, nogil=True)
def _py_solve_single(G, b, sparsity, tol, finetune_freq, has_bias=True):
    """
    solve 1/2 * w G^T w + b^T w + reg |w| problem with greedy coordinate descent

    Parameters
    ----------
    G: 2-d matrix with X^T X

    b: 1-d matrix with X^T y

    has_bias: G matrix includes the bias term or not
    """
    w, g = _get_init_w_g(G, b)

    # update bias
    if has_bias:
        _gcd_step(G=G, w=w, g=g, target_index=g.shape[0] - 1)

    num_act = int((w.shape[0] - 1) * (1.0 - sparsity))
    for act in range(num_act * 2):
        delta = _gcd_step(G=G, w=w, g=g, target_index=np.argmax(np.abs(g)))
        if np.abs(delta) <= tol or np.sum(np.abs(w) > 1e-5) >= num_act:
            break

        if act % finetune_freq == 0:
            for i in range(act):
                delta = _gcd_step(G=G, w=w, g=g, target_index=np.argmax(np.abs(g) * (np.abs(w) > 0).astype(np.float64)))
                if np.abs(delta) <= 1e-5:
                    break
    sparse_finetuning(G=G, w=w, g=g)
    return w
