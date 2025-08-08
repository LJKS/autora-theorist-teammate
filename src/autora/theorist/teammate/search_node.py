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


    def forward(self, independent_variables):
        """
        Forward pass through the search node, applying each component in parallel
        args:
            independent_variables: input data (batched) to be processed by the equation components
        """
        component_results = [component(independent_variables) for component in self.components] #shape: (batch_size, num_components)
        weighted_component_results = self.gates(component_results)
        output = self.reduce_op(weighted_component_results)
        return output

    def node_sampling_strategies(self, key):
        sampling_strategies = {'all': self.sampling_strategy_all_nodes,
                               'expected': self.sampling_strategy_expected_nodes,
                               'probable_gates_only': lambda: self.sampling_strategy_expected_nodes(cutoff=0.9)}

        #returns an iterator over own subnodes
        if key in sampling_strategies:
            return sampling_strategies[key]
        else:
            raise ValueError(
                f"Invalid node sampling strategy: {key}. Available strategies: {list(sampling_strategies.keys())}")

    def sampling_strategy_all_nodes(self):
        """
        Returns a list of all nodes in the search node
        """
        return self.components

    def sampling_strategy_expected_nodes(self, cutoff=0.5):
        """
        Returns a list of nodes that are expected to be activated based on the gate probabilities
        """
        open_gate_probabilities = self.gates.open_gate_probability()
        open_gate_probabilities = torch.squeeze(open_gate_probabilities).numpy().tolist()
        chosen_nodes = [node for node, prob in zip(self.components, open_gate_probabilities) if prob >= cutoff]
        return chosen_nodes

    def to_equation(self, node_sampling: str, use_outer_brackets: bool = False) -> str:
        """
        Convert the search node to a string representation of an equation
        args:
            node_sampling: string, in self.node_sampling_strategies
            use_outer_brackets: whether to use outer brackets in the equation
        """
        node_sampling = self.node_sampling_strategies(node_sampling)
        equation = ''
        for node in node_sampling:
            if isinstance(node, SearchNode):
                added_string = node.to_equation(node_sampling, use_outer_brackets=True)
                if not added_string == '':
                    equation += added_string
                    equation += self.reduce_op_notation
            else:
                equation += node.eq_string()
                equation += self.reduce_op_notation

        if len(equation) > 0:
            equation = equation[:-1]  # Remove the last operator
        if use_outer_brackets:
            equation = f'({equation})'
        return equation

    def test_run(self, independent_variables, node_sampling: str):
        """
        Test run of the search node
        args:
            node_sampling: string, in self.node_sampling_strategies
            independent_variables: input data (batched) to be processed by the equation components
        """
        node_sampling = self.node_sampling_strategies(node_sampling)
        output = self.forward(independent_variables)
        return output, self.to_equation(node_sampling)

    def regularization_cost(self):
        gate_probs = self.gates.open_gate_probability()
        gate_probs = torch.squeeze(gate_probs)
        #start with torch zero scalar
        cost_total = torch.zeros((), device=gate_probs.device)
        for i, component in enumerate(self.components):
            if isinstance(component, SearchNode):
                costs = component.regularization_cost()
                cost_total += gate_probs[i] * costs
            else:
                costs = component.cost()
                cost_total += gate_probs[i] * costs
        return cost_total