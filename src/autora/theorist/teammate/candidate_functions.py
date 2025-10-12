import pandas as pd
import numpy as np
import torch 
from torch import nn
from utils import get_variable_name


class CustomModule(nn.Module):
    def __init__(self, var_idxs: list):
        super().__init__()
        self.var_idxs = var_idxs  # List of variable indices this module uses

    def eq_string(self):
        return None

    def cost(self, *args):
        return float(len(self.eq_string(*args)))



class Plus(CustomModule):               
    def forward(self, independent_variables):
        arg_1 = independent_variables[self.var_idxs[0]]
        arg_2 = independent_variables[self.var_idxs[1]]
        out = torch.add(arg_1, arg_2)
        return out
    
    def eq_string(self):
        return f"{get_variable_name(self.var_idxs[0])}+{get_variable_name(self.var_idxs[1])}"
    

class Subtract(CustomModule):               
    def forward(self, independent_variables):
        arg_1 = independent_variables[self.var_idxs[0]]
        arg_2 = independent_variables[self.var_idxs[1]]
        out = torch.subtract(arg_1, arg_2)
        return out

    def eq_string(self):
        return f"{get_variable_name(self.var_idxs[0])}-{get_variable_name(self.var_idxs[1])}"



class Mult(CustomModule):
    def forward(self, independent_variables):
        arg_1 = independent_variables[self.var_idxs[0]]
        arg_2 = independent_variables[self.var_idxs[1]]
        out = torch.multiply(arg_1, arg_2)
        return out
    def eq_string(self):
        return f"{get_variable_name(self.var_idxs[0])}*{get_variable_name(self.var_idxs[1])}"



class Div(CustomModule):               
    def forward(self, independent_variables):
        arg_1 = independent_variables[self.var_idxs[0]]
        arg_2 = independent_variables[self.var_idxs[1]]
        out = torch.divide(arg_1, arg_2)
        return out
    
    def eq_string(self):
        return f"{get_variable_name(self.var_idxs[0])}/{get_variable_name(self.var_idxs[1])}"
    
"""
class Ln(CustomModule):
    def __init__(self):
        super().__init__()
               
    def forward(self, a):
        out = torch.log(a)
        return out
    
    def eq_string(self, stringA):
        return f"ln({stringA})"

class Exp(CustomModule):
    def __init__(self):
        super().__init__()
               
    def forward(self, a):
        out = torch.exp(a)
        return out
    
    def eq_string(self, stringA):
        return f"exp{stringA}"    


class Constant(CustomModule):
    def __init__(self, value):
        super().__init__()
        self.value = torch.tensor(value, dtype=torch.float32)
        self.stringValue = str(round(self.value.item(), 1))
               
    def forward(self):
        out = self.value
        return out
    
    def eq_string(self, stringValue):
        return f"{round(stringValue, 1)}"
    

class Pow(CustomModule):
    def __init__(self, value):
        super().__init__()
        self.value = torch.tensor(value, dtype=torch.float32)
        self.stringValue = str(round(self.value.item(), 1))  

    def forward(self, a):
        out = torch.pow(a, self.value)
        return out
    
    def eq_string(self, stringA):
        return f"{stringA}^{self.stringValue}"



dict_mate = {
    "plus": lambda : Plus(),
    "minus": lambda : Subtract(),
    "mult": lambda : Mult(),
    "div": lambda : Div(),
    "ln": lambda: Ln(),
    "exp": lambda: Exp(),
    "const": lambda: Constant(),
    "pow": lambda: Pow()
    }




'''
# Testblock

def test_exp_module():
    model = Exp()
    
    # Test 1: forward() funktioniert
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 1.0, 2.0])
    
    print("Forward output:", model.forward(a))  # Erwartet: [1.0, e^1, e^2]

    # Test 2: eq_string() funktioniert
    print("eq_string output:", model.eq_string("x"))  # Erwartet: "expx"

    # Test 3: cost() verwendet eq_string korrekt
    print("Cost output:", model.cost("x"))  # Erwartet: Länge von "expx" → 4

    # Test 4: isinstance
    print("Ist Exp Instanz von CustomModule?", isinstance(model, CustomModule))  # Erwartet: True

test_exp_module() '''
"""