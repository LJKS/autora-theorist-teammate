import torch
#Dataloader
from torch.utils.data import DataLoader
import json
def np2data(x,y,hparam_file=None):
    """
    Converts numpy arrays to PyTorch dataset.
    """
    assert hparam_file is not None, "hparam_file must be provided"
    with open(hparam_file, 'r') as f:
        hparams = json.load(f)
    batch_size = hparams.get('batch_size', 32)  # Default batch size if not specified in hparams
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    #load hparams
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
