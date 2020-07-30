import logging
from typing import Dict
from fast_pruning.pruning.gcd_inc_solver import GCDIncSolver
from fast_pruning.pruning.gd_prune_solver import GDPruneSolver
from fast_pruning.data.pruning_dataset import PruningDataset
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


class ModelPruner:
    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def pruning(self, dataset: PruningDataset, target_sparsity_dict: Dict[str, float], default_sparsity: float,
                cross_channel=False):
        pruned_weight_dict = dict()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for key, record in dataset.get_record_dict().items():
                target_sparsity = target_sparsity_dict.get(key, default_sparsity)
                if target_sparsity == 0.0:
                    logger.info('target sparsity is 0.0, skip')
                    continue
                logger.info(f'submit {key}')
                pruned_weight_dict[key] = executor.submit(self.prune_single, key, record, target_sparsity, cross_channel)

        for key, future in pruned_weight_dict.items():
            pruned_weight_dict[key] = future.result()

        return pruned_weight_dict

    def prune_single(self, key, record, target_sparsity, cross_channel):
        solver = GCDIncSolver(backend='cpp', sparsity=target_sparsity, cross_channel=cross_channel)
        w = solver.solve(xtx=record.xtx, xty=record.xty, yty=record.yty, has_bias=False)
        sparsity, loss = solver.evaluate(xtx=record.xtx, xty=record.xty, yty=record.yty, w=w)
        logging.info(f'result of {key}, sparsity: {sparsity}, loss: {loss}')
        return w
