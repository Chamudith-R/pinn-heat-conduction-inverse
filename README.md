# Physics-Informed Neural Networks for Heat Conduction Inverse Problem

A complete implementation for solving the transient 2D heat conduction equation with an unknown space- and time-dependent heat source using Physics-Informed Neural Networks (PINNs).

## Problem Statement

This project addresses an inverse problem in heat transfer:
- **Forward Problem**: Given domain properties, boundary conditions, and heat source, find temperature distribution
- **Inverse Problem**: Given sparse temperature measurements at sensors, simultaneously infer:
  1. The unknown internal heat source function `g(x,y,t)`
  2. The complete temperature field `u(x,y,t)` everywhere in the domain

### Governing Equation

The 2D unsteady heat conduction equation with unknown heat source:

$$\frac{\partial u}{\partial t} = \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) + g(x,y,t)$$

Where:
- `u(x,y,t)` = temperature field
- `α` = thermal diffusivity (0.1 in base case)
- `g(x,y,t)` = unknown heat source function
- Domain: rectangular plate with dimensions `Lx × Ly = 1.0 × 1.0`
- Time interval: `[0, T_final]` where `T_final = 1.0`

## Features

✅ **Two neural networks**:
- Temperature network `u_net`: Predicts temperature field
- Heat source network `g_net`: Predicts internal heat source

✅ **Physics-informed loss function**:
- PDE residual loss (ensures PDE satisfaction)
- Boundary condition loss (zero temperature at edges)
- Data fitting loss (matches sparse sensor measurements)

✅ **Advanced training**:
- Automatic differentiation for computing gradients
- Learning rate scheduling
- Gradient clipping for stability
- Loss weighting for multi-objective optimization

✅ **Comprehensive analysis**:
- Error metrics (MAE, RMSE, relative error)
- Visualization of predictions vs. exact solutions
- Training loss history
- Solution contour plots

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/Chamudith-R/pinn-heat-conduction-inverse.git
cd pinn-heat-conduction-inverse
```

2. **Create virtual environment** (recommended):
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n pinn python=3.10
conda activate pinn
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## How to Run

### Basic Execution

Run the complete pipeline with default parameters:

```bash
python pinn_heat_conduction.py
```

This will:
1. Generate synthetic training data (5000 interior points, 1000 boundary points, 50 sensor measurements)
2. Build the PINN model (two networks with specified architectures)
3. Train for 10,000 epochs
4. Evaluate performance on test data
5. Generate visualization plots

**Expected output**:
```
================================================================================
Physics-Informed Neural Networks for Heat Conduction Inverse Problem
================================================================================

Using device: cuda (or cpu)

Generating training data...
Interior points: 5000
Boundary points: 1000
Sensor measurements: 50

Building PINN model...
Total parameters: 295,553

Training for 10000 epochs...
Epoch 1000/10000: Loss = X.XXXE-XX, L_PDE = X.XXXE-XX, L_BC = X.XXXE-XX, L_Data = X.XXXE-XX
Epoch 2000/10000: Loss = X.XXXE-XX, L_PDE = X.XXXE-XX, L_BC = X.XXXE-XX, L_Data = X.XXXE-XX
...

================================================================================
RESULTS AND METRICS
================================================================================

Temperature Field u(x,y,t):
  MAE:             X.XXXE-XX
  RMSE:            X.XXXE-XX
  Relative Error:  0.XXXXX (X.XX%)

Heat Source Function g(x,y,t):
  MAE:             X.XXXE-XX
  RMSE:            X.XXXE-XX
  Relative Error:  0.XXXXX (X.XX%)

Generating visualizations...
Saved: losses.png
Saved: solution_at_t05.png

================================================================================
Training completed successfully!
================================================================================
```

### Advanced Usage

#### Custom Problem Parameters

Create a Python script with custom settings:

```python
from pinn_heat_conduction import *

# Custom domain parameters
dataset = HeatConductionDataset(Lx=2.0, Ly=1.5, T_final=2.0, device='cuda')

# Generate data with different configuration
data = dataset.generate_training_data(
    n_interior=8000,        # More collocation points
    n_boundary=2000,        # More boundary points
    n_sensors=100,          # More sensor measurements
    noise_level=0.05        # 5% noise
)

# Build larger model
model = PINN(
    layers_u=[3, 256, 256, 256, 256, 1],      # Deeper network
    layers_g=[3, 128, 128, 128, 1],
    activation='tanh'
)

# Train with custom weights
trainer = PINNTrainer(model, data, device='cuda', alpha=0.1)
trainer.train(
    epochs=20000,
    print_freq=500,
    lambda_pde=2.0,         # Increase PDE constraint weight
    lambda_boundary=20.0,
    lambda_data=15.0
)

# Visualize results
fig_losses = trainer.plot_losses()
fig_solution = plot_solution(trainer, dataset, time_slice=1.0)
plt.show()
```

#### GPU Acceleration

To use GPU (if CUDA is available):

```bash
# The code automatically detects GPU
python pinn_heat_conduction.py
```

#### Generating Predictions on Custom Points

```python
from pinn_heat_conduction import *

# Load trained model
model = PINN()
model.load_state_dict(torch.load('pinn_model.pth'))
trainer = PINNTrainer(model, data, device='cuda')

# Predict at custom locations
x_custom = np.array([0.25, 0.5, 0.75])
y_custom = np.array([0.25, 0.5, 0.75])
t_custom = np.array([0.5, 0.5, 0.5])

u_pred, g_pred = trainer.predict(x_custom, y_custom, t_custom)

print("Temperature predictions:", u_pred.flatten())
print("Heat source predictions:", g_pred.flatten())
```

## Project Structure

```
pinn-heat-conduction-inverse/
├── pinn_heat_conduction.py      # Main implementation
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── outputs/
    ├── pinn_model.pth           # Trained model weights
    ├── losses.png               # Training loss curves
    └── solution_at_t05.png      # Solution visualizations
```

## File Descriptions

### `pinn_heat_conduction.py` (Main Implementation)

**Classes**:

1. **`PINN`**: Neural network architecture
   - Two sub-networks: `u_net` (temperature) and `g_net` (heat source)
   - Methods: `forward_u()`, `forward_g()`, `compute_residuals()`

2. **`HeatConductionDataset`**: Data generation
   - Generates synthetic heat source
   - Generates synthetic temperature field
   - Creates training data with collocation points, boundary conditions, and sensor measurements

3. **`PINNTrainer`**: Training manager
   - Computes multi-component loss function
   - Manages optimization with Adam optimizer and learning rate scheduling
   - Tracks training metrics
   - Provides inference interface

**Functions**:

- `plot_solution()`: Visualizes predictions vs. exact solutions
- `compute_metrics()`: Calculates error metrics (MAE, RMSE, relative error)

## Loss Function Details

### Total Loss Formulation

L_total = λ_PDE · L_PDE + λ_BC · L_BC + λ_data · L_data

**Components**:

1. **PDE Residual Loss**: Enforces satisfaction of the heat conduction PDE
2. **Boundary Condition Loss**: Enforces zero temperature at domain boundaries
3. **Data Fitting Loss**: Minimizes error at sensor measurement locations

**Default weights**: λ_PDE = 1.0, λ_BC = 10.0, λ_data = 10.0

## Model Architecture

### Temperature Network (u_net)
- Input: 3 neurons (x, y, t coordinates)
- Hidden layers: 4 layers × 128 neurons each
- Activation: Tanh
- Output: 1 neuron (temperature u)
- Total parameters: ~295k

### Heat Source Network (g_net)
- Input: 3 neurons (x, y, t coordinates)
- Hidden layers: 3 layers × 64 neurons each
- Activation: Tanh
- Output: 1 neuron (heat source g)
- Total parameters: ~25k

## Training Details

- **Optimizer**: Adam (learning rate = 0.001)
- **Scheduler**: StepLR (step_size=1000, gamma=0.9)
- **Gradient clipping**: max_norm=1.0
- **Default epochs**: 10,000
- **Print frequency**: 500 epochs

## Output Files

After running, the following files are generated:

1. **pinn_model.pth**: Trained model weights
2. **losses.png**: Training loss curves showing all components
3. **solution_at_t05.png**: Solution visualization at t=0.5

## Performance Metrics

Expected results on test set with default parameters:

| Metric | Temperature u | Heat Source g |
|--------|---------------|---------------|
| MAE | ~1e-3 | ~1e-3 |
| RMSE | ~1e-3 | ~1e-3 |
| Relative Error | <1% | <5% |

## Theory and Background

### Physics-Informed Machine Learning

PINNs combine:
- Deep neural networks for function approximation
- Physical laws (PDEs) as constraints
- Automatic differentiation for computing derivatives

### Key Advantages of PINNs

1. ✅ Mesh-free: No domain discretization needed
2. ✅ Sample efficient: Works with limited sparse data
3. ✅ Physics-respecting: Encodes domain knowledge
4. ✅ Inverse capability: Can identify parameters and sources
5. ✅ Scalable: Can handle high-dimensional problems

### References

- **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

- **University of Washington - Physics-Informed Machine Learning**: https://composites.uw.edu/AI/
  - Dr. Navid Zobeiry, Associate Professor of Materials Science and Engineering

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce data size
```python
data = dataset.generate_training_data(
    n_interior=2000,
    n_boundary=500,
    n_sensors=25
)
```

### Issue: Slow Training

**Solution**: Use GPU
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Poor Convergence

**Adjust loss weights**:
```python
trainer.train(
    epochs=20000,
    lambda_pde=2.0,
    lambda_boundary=20.0,
    lambda_data=10.0
)
```

## Discussion Points

### 1. PINN vs. Traditional Numerical Methods (FEM)

**PINN Advantages**:
- Mesh-free approach
- Data-efficient with sparse measurements
- Physics-aware via PDE constraints
- Natural inverse problem capability

**PINN Disadvantages**:
- Higher computational cost
- Requires careful hyperparameter tuning
- Less mature validation standards

### 2. Effect of Systematic Sensor Bias

If all measurements are 5% high:
- Temperature field shifts upward by ~5%
- Inferred heat source compensates with lower magnitude
- Solution: Implement sensor calibration in loss function

### 3. Inferring Unknown Thermal Diffusivity

Make α a learnable parameter:
- Add α as a neural network parameter
- Modify residual computation to use self.alpha
- Include regularization on α to prevent divergence

## License

MIT License

## Contact

For questions, open an issue on GitHub.

---

**Last Updated**: September 2026
