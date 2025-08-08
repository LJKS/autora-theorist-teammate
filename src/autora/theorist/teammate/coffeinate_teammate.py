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
    optimizer_gates = torch.optim.Adam(gate_parameters, lr=hparams['learning_rate_gates'])
    optimizer_parameters = torch.optim.Adam(model_parameters, lr=hparams['learning_rate_other'])
    criterion = torch.nn.MSELoss()
    num_epochs = hparams['num_epochs']
    epoch_iter = tqdm(range(num_epochs), desc="Training Epochs")
    for epoch in epoch_iter:
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer_gates.zero_grad()
            optimizer_parameters.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer_gates.step()
            optimizer_parameters.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        epoch_iter.set_postfix(loss=epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")



def bilevel_optimization(model, dataloader, hparam_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(hparam_file, 'r') as f:
        hparams = json.load(f)

    gate_parameters, model_parameters = split_model_parameters(model)
    optimizer_gates = torch.optim.Adam(gate_parameters, lr=hparams['learning_rate_gates'])
    optimizer_parameters = torch.optim.Adam(model_parameters, lr=hparams['learning_rate_other'])
    #only estimating regression tasks
    criterion = torch.nn.MSELoss()
    num_meta_epochs = hparams['meta_epochs']
    num_sub_epochs_params = hparams['sub_epochs_params']
    num_sub_epochs_gates = hparams['sub_epochs_gates']
    meta_epoch_iter = tqdm(range(num_meta_epochs), desc="Meta Training Epochs")
    for meta_epoch in meta_epoch_iter:
        running_loss = 0.0
        #iterate over dataset for parameter optimization
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer_parameters.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass and optimization
            loss.backward()
            optimizer_parameters.step()
            running_loss += loss.item()
        epoch_loss_params = running_loss / len(dataloader)
        #reset accumulated gradients for gate optimization
        optimizer_gates.zero_grad()
        print(f'Updating model parameters; Loss: {epoch_loss_params:.4f}')
        #reset running loss for gate optimization
        running_loss = 0.0

        #now iterate over the dataset for gate optimization
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer_gates.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass and optimization
            loss.backward()
            optimizer_gates.step()
            running_loss += loss.item()
        epoch_loss_gates = running_loss / len(dataloader)
        print(f'Updating gate parameters; Loss: {epoch_loss_gates:.4f}')
        meta_epoch_iter.set_postfix(loss_params=epoch_loss_params, loss_gates=epoch_loss_gates)
        print(f"Meta Epoch [{meta_epoch + 1}/{num_meta_epochs}], Loss Params: {epoch_loss_params:.4f}, Loss Gates: {epoch_loss_gates:.4f}")