import numpy as np
from concurrent.futures import ThreadPoolExecutor
from abc import ABCMeta, abstractmethod


class SolverBase(metaclass=ABCMeta):
    def __init__(self, num_workers=8):
        """
        Base Class of Solver
        """
        self.num_workers = num_workers

    def parallel_run(self, func, iterable_kwargs):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list()
            for kwargs in iterable_kwargs:
                results.append(executor.submit(func, **kwargs))
            results = [r.result() for r in results]
        return results

    def evaluate(self, xtx, xty, yty, w):
        sparsity = np.mean(np.abs(w) <= 1e-10)
        loss = np.mean(np.dot(w.T, np.dot(xtx, w)) - 2 * np.dot(w.T, xty) + yty)
        return sparsity, loss

    @abstractmethod
    def solve(self, xtx: np.array, xty: np.array, yty: np.array, has_bias: bool):
        """
        Parameters
        ----------
        xtx: 2d np.array
            matrix of X.T dot X
        xty: 2d np.array
            matrix of X.T dot y
        yty: 2d np.array
            matrix of y.T dot y
        has_bias: bool, default True
            matrix X has bias or not, if True, we assume last column of matrix X is constant 1.0
        """
        raise NotImplementedError
