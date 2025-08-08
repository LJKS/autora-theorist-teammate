import pandas as pd
import numpy as np
import torch 
from torch import nn



class CustomModule(nn.Module):
    def __init__(self, referenced_independent_variables: list = []):
        super().__init__()
        self.referenced_independent_variables = referenced_independent_variables
    
    def eq_string(self):
        return None

    def cost(self, *args):
        return float(len(self.eq_string(*args)))



class Plus(CustomModule):

    def forward(self, independent_variables):
        assert len(self.referenced_independent_variables) == 2, "Plus operation requires exactly two independent variables."
        arg_1 = independent_variables[self.referenced_independent_variables[0]]
        arg_2 = independent_variables[self.referenced_independent_variables[1]]
        out = torch.add(arg_1, arg_2)
        return out
    
    def eq_string(self, stringA, stringB):
        return f"{stringA}+{stringB}"
    

class Subtract(CustomModule):               
    def forward(self, independent_variables):
        assert len(self.referenced_independent_variables) == 2, "Subtract operation requires exactly two independent variables."
        arg_1 = independent_variables[self.referenced_independent_variables[0]]
        arg_2 = independent_variables[self.referenced_independent_variables[1]]
        out = torch.subtract(arg_1, arg_2)
        return out
    
    def eq_string(self, stringA, stringB):
        return f"{stringA}-{stringB}"



class Mult(CustomModule):
    def forward(self, a, b):
        out = torch.multiply(a,b)
        return out
    
    def eq_string(self, stringA, stringB):
        return f"{stringA}*{stringB}"



class Div(CustomModule):               
    def forward(self, a, b):
        out = torch.divide(a, b)
        return out
    
    def eq_string(self, stringA, stringB):
        return f"{stringA}/{stringB}"
    

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