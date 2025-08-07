import pandas as pd
import numpy as np
import torch 
from torch import nn



class CustomModule(nn.Module):
    def __init__(self):
        super().__init__()
    
    def eq_string(self):
        return None

    def cost(self, *args):
        return len(self.eq_string(*args))



class Plus(CustomModule):               
    def forward(self, a, b):
        out = torch.add(a, b)
        return out
    
    def eq_string(self, stringA, stringB):
        return f"{stringA}+{stringB}"
    

class Subtract(CustomModule):               
    def forward(self, a, b):
        out = torch.subtract(a, b)
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
    



dict_mate = {
    "plus": lambda : Plus(),
    "minus": lambda : Subtract(),
    "mult": lambda : Mult(),
    "div": lambda : Div(),
    "ln": lambda: Ln(),
    "exp": lambda: Exp(),
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
