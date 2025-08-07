import torch
from torch import nn
import l0_layer

COMPONENT_DICT = {}
REDUCE_OPS = {'sum': lambda x: torch.sum(x, dim=1)}
REDUCE_OPS_NOTATION = {'sum': '+'}
class SearchNode(nn.Module):
    def __init__(self, components: list, reduce_op='sum'):
        '''
        args:
            components: list of strings (keys in COMPONENT_DICT) components to be used in the search node
        '''
        super().__init__()
        self.component_strings = components
        self.components = [COMPONENT_DICT[component]() for component in components]
        self.size = len(self.components)
        self.gates = l0_layer.LZeroGate((self.size,))
        assert reduce_op in REDUCE_OPS, f"Invalid reduce operation: {reduce_op}. Available operations: {list(REDUCE_OPS.keys())}"
        self.reduce_op = REDUCE_OPS[reduce_op]
        assert reduce_op in REDUCE_OPS_NOTATION, f"Invalid reduce operation notation: {reduce_op}. Available notations: {list(REDUCE_OPS_NOTATION.keys())}"
        self.reduce_op_notation = REDUCE_OPS_NOTATION[reduce_op]


    def forward(self, independent_variables, gate_activated=True, test=False):
        """
        Forward pass through the search node, applying each component in parallel
        args:
            independent_variables: input data (batched) to be processed by the equation components
        """
        component_results = [component(independent_variables) for component in self.components] #shape: (batch_size, num_components)
        weighted_component_results = self.gates(component_results)
        output = self.reduce_op(weighted_component_results)
        return output

    def to_equation(self):
