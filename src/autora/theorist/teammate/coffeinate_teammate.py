import search_node
import torch
import data
import json
from tqdm import tqdm

def split_model_parameters(model):
    gate_parameters = []

    # extract gate parameters
    def extract_all_gaes_from_node(node):
        if isinstance(node, search_node.SearchNode):
            gates = node.gates.parameters()
            gate_parameters.append(gates)
            for child in node.components:
                extract_all_gaes_from_node(child)

    extract_all_gaes_from_node(model)
    assert len(gate_parameters) > 0, "No gate parameters found in the model"

    # extract all other parameters
    model_parameters = [p for p in model.parameters() if p not in gate_parameters]
    return gate_parameters, model_parameters

def naive_optimization(model, dataloader, hparam_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(hparam_file, 'r') as f:
        hparams = json.load(f)
    gate_parameters, model_parameters = split_model_parameters(model)

    num_epochs = hparams['num_epochs']
    epoch_iter = tqdm(range(num_epochs), desc="Training Epochs")
    for epoch in epoch_iter:
        running_loss = 0.0
        for inputs, targets in dataloader:


def bilevel_optimization(model, dataloader, hparam_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(hparam_file, 'r') as f:
        hparams = json.load(f)

    gate_parameters = []
    #extract gate parameters
    def extract_all_gaes_from_node(node):
        if isinstance(node, search_node.SearchNode):
            gates = node.gates.parameters()
            gate_parameters.append(gates)
            for child in node.components:
                extract_all_gaes_from_node(child)

    extract_all_gaes_from_node(model)
    assert len(gate_parameters) > 0, "No gate parameters found in the model"

    #extract all other parameters
    model_parameters = [p for p in model.parameters() if p not in gate_parameters]

    optimizer_gates = torch.optim.Adam(gate_parameters, lr=hparams['learning_rate_gates'])
    optimizer_parameters = torch.optim.Adam(model_parameters, lr=hparams['learning_rate_other'])
    num_epochs = hparams['num_epochs']
    consecutive_gate_epochs = hparams['consecutive_gate_epochs']
    consecutive_params_epochs = hparams['consecutive_params_epochs']
    meta_epochs = hparams['meta_epochs'] #iterations of bilevel optimization
    #tqdm over meta epochs
    meta_epoch_iter = tqdm(range(meta_epochs), desc="Meta Training Epochs")
    for meta_epoch in meta_epoch_iter:
        # train params
        for _ in range(consecutive_params_epochs):




