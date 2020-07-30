import logging

import numpy as np

from fast_pruning.pruning.solver_base import SolverBase

logger = logging.getLogger()


class GCDL1Solver(SolverBase):
    def __init__(self, backend='cpp', reg=0.01, tol=1e-5):
        super().__init__()
        self.backend = backend
        self.reg = reg
        self.tol = tol

    def solve(self, xtx: np.array, xty: np.array, yty: np.array, has_bias: bool):
        logger.info('solve w with XTX shape: {}, XTy shape: {}, yty shape: {}'.format(xtx.shape, xty.shape, yty.shape))

        if self.backend == 'cpp':
            raise NotImplementedError('cpp backend of this GCDL1Solver is not implemented yet')
        elif self.backend == 'python':

            w = np.vstack([
                self._py_solve_single(G=xtx, b=xty[:, i], reg=self.reg, has_bias=has_bias) for i in range(yty.shape[0])
            ])
        else:
            raise ValueError('unknown backend: {}'.format(self.backend))
        return w

    @staticmethod
    def _get_init_w_g(G, b):
        w = np.zeros(b.shape)
        g = -b
        # g = np.dot(G, w) - b
        return w, g

    @staticmethod
    def _gcd_step(G, w, g, target_index):
        ori_wi = float(w[target_index])
        w[target_index] -= g[target_index] / G[target_index, target_index]
        delta = (w[target_index] - ori_wi)
        g += delta * G[:, target_index]
        return delta

    @staticmethod
    def _gcd_l1_step(G, w, g, target_index, reg, has_bias):
        ori_wi = float(w[target_index])
        w[target_index] -= g[target_index] / G[target_index, target_index]
        if not has_bias or (target_index != w.shape[0] - 1):
            T = reg / G[target_index, target_index]
            w[target_index] = np.where(
                np.abs(w[target_index]) > T, np.where(w[target_index] > 0, w[target_index] - T, w[target_index] + T),
                0.0)
        delta = (w[target_index] - ori_wi)
        g += delta * G[:, target_index]
        return delta

    def sparse_finetuning(self, G, w, g):
        for i in range(w.size * 10):
            delta = self._gcd_step(G=G,
                                   w=w,
                                   g=g,
                                   target_index=np.argmax(np.abs(g) * (np.abs(w) > 0).astype(np.float64)))
            if np.abs(delta) <= 1e-5:
                break

    def _py_solve_single(self, G, b, reg, has_bias=True):
        """
        solve 1/2 * w G^T w + b^T w + reg |w| problem with greedy coordinate descent

        Parameters
        ----------
        G: 2-d matrix with X^T X

        b: 1-d matrix with X^T y

        reg: weight of regularization parameters

        has_bias: G matrix includes the bias term or not
        """
        w, g = self._get_init_w_g(G, b)

        # update bias
        if has_bias:
            self._gcd_step(G=G, w=w, g=g, target_index=g.shape[0] - 1)

        while True:
            delta = self._gcd_l1_step(G=G, w=w, g=g, target_index=np.argmax(np.abs(g)), reg=reg, has_bias=has_bias)
            if np.abs(delta) <= self.tol:
                break

            # for i in range(10):
            #     delta = self._gcd_l1_step(G=G,
            #                               w=w,
            #                               g=g,
            #                               target_index=np.argmax(np.abs(g) * (np.abs(w) > 0).astype(np.float64)),
            #                               reg=reg,
            #                               has_bias=has_bias)
            #     if np.abs(delta) <= 1e-5:
            #         break

        # self.sparse_finetuning(G=G, w=w, g=g)
        return w
