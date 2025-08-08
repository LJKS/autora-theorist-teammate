"""
Example Theorist
"""
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import search_node
import caffeinate_teammate
import data
import torch

HPARAM_PATH = 'hparams/bilevel_optimization_hparams.json'

class ExampleRegressor(BaseEstimator):
    def __init__(self):
        self.teammate = search_node.SearchNode()

    def fit(self,
            conditions: Union[pd.DataFrame, np.ndarray],
            observations: Union[pd.DataFrame, np.ndarray]):
        dataloader = data.np2data(conditions, observations, hparam_file=HPARAM_PATH)
        self.teammate = caffeinate_teammate.bilevel_optimization(model=self.teammate, dataloader=dataloader, hparam_file=HPARAM_PATH)

    def predict(self,
                conditions: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        #get model device
        device = next(self.teammate.parameters()).device
        conditions = torch.tensor(conditions, dtype=torch.float32, device=device)
        output, _ = self.teammate.test_run(conditions, node_sampling='expected_nodes')
        return output.cpu().detach().numpy()
