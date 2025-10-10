## Theorist: A Modular Framework for Symbolic Search and Differentiable Equation Discovery

This repository implements **Theorist**, a modular and extensible framework for *symbolic equation discovery* using neural architectures with **L₀ regularized gates**.  

The core idea is to represent mathematical expressions as **search trees** composed of parameterized and gated components (e.g., addition, multiplication, logarithm, power).  
Each component is trainable and selectively activated through **LZeroGate** layers, which enforce sparsity and allow the model to explore compact symbolic representations.


The framework builds symbolic equations by recursively combining **CustomModules** (basic mathematical operations) through **SearchNodes**, each equipped with **stochastic gates** controlling which components are active.  
During training, both the gates and the functional parameters are optimized in tandem to minimize prediction error while promoting parsimonious models.



### 1. `CustomModule` and Mathematical Operators
Each operator (e.g., `Plus`, `Subtract`, `Ln`, `Exp`, `Pow`, etc.) inherits from `CustomModule`, a lightweight `torch.nn.Module` wrapper that defines:

- A `forward()` computation (e.g., `torch.add`, `torch.exp`)
- A symbolic representation via `eq_string()`
- A structural complexity measure via `cost()`

Goodness of an equation is determined by:
- Mean Squared Error between model output and data (fit quality), and 
- regularization cost derived from gate activations and module complexity (model simplicity).

**Example:**
```python
class Plus(CustomModule):               
    def forward(self, a, b):
        return torch.add(a, b)
    
    def eq_string(self, stringA, stringB):
        return f"{stringA}+{stringB}"



2. SearchNode

A SearchNode represents a single compositional layer in the symbolic search tree.
It contains:
- A list of mathematical components (from COMPONENT_DICT)
- A gating layer (LZeroGate) that controls which components are active
- A reduction operation (currently “sum”) to combine outputs

Each node can recursively contain other SearchNode objects, enabling deep compositional hierarchies.

The search is constrained by:
-A fixed maximum tree depth and number of components per node,
- Gate regularization, which penalizes unnecessary components,
- And the optimization scheme, which prunes unlikely nodes by reducing their gate probabilities.


3. LZeroGate

Implements stochastic binary gates using the Stretched Concrete Distribution (Louizos et al., 2018).
This layer learns differentiable switches that control which parts of the search tree are active.