"""
Physics-Informed Neural Networks for Transient Heat Conduction Inverse Problem
Solving the 2D unsteady heat equation with unknown space- and time-dependent heat source

∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) + g(x,y,t)

where:
    u(x,y,t) = temperature field
    α = thermal diffusivity
    g(x,y,t) = unknown heat source function
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for heat conduction inverse problem
    
    Architecture:
    - Temperature network u_net: predicts u(x,y,t)
    - Heat source network g_net: predicts g(x,y,t)
    """
    
    def __init__(self, layers_u=[3, 128, 128, 128, 128, 1], 
                 layers_g=[3, 64, 64, 64, 1],
                 activation='tanh'):
        """
        Args:
            layers_u: Network architecture for temperature field
            layers_g: Network architecture for heat source
            activation: Activation function ('tanh' or 'relu')
        """
        super(PINN, self).__init__()
        
        self.activation_name = activation
        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        
        # Build temperature network
        self.u_net = self._build_network(layers_u)
        
        # Build heat source network
        self.g_net = self._build_network(layers_g)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_network(self, layers):
        """Build a fully connected neural network"""
        net = nn.ModuleList()
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i+1]))
        return net
    
    def _initialize_weights(self):
        """Xavier uniform initialization"""
        for layer in self.u_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.g_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward_u(self, x):
        """Forward pass for temperature network
        
        Args:
            x: Input tensor of shape (N, 3) where x[:, 0] = x-coord, 
                                            x[:, 1] = y-coord, 
                                            x[:, 2] = t (time)
        Returns:
            u: Temperature field predictions
        """
        for i, layer in enumerate(self.u_net[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.u_net[-1](x)
        return x
    
    def forward_g(self, x):
        """Forward pass for heat source network
        
        Args:
            x: Input tensor of shape (N, 3)
        Returns:
            g: Heat source predictions
        """
        for i, layer in enumerate(self.g_net[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.g_net[-1](x)
        return x
    
    def compute_residuals(self, x_physics, alpha=0.1):
        """
        Compute PDE residuals
        
        Residual: ∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) - g = 0
        
        Args:
            x_physics: Tensor of shape (N, 3) with domain points
            alpha: Thermal diffusivity
        
        Returns:
            residuals: PDE residuals
        """
        x_physics.requires_grad_(True)
        
        # Predict temperature and heat source
        u = self.forward_u(x_physics)
        g = self.forward_g(x_physics)
        
        # Compute gradients
        u_x = torch.autograd.grad(u.sum(), x_physics, create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u.sum(), x_physics, create_graph=True)[0][:, 1:2]
        u_t = torch.autograd.grad(u.sum(), x_physics, create_graph=True)[0][:, 2:3]
        
        # Compute second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), x_physics, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), x_physics, create_graph=True)[0][:, 1:2]
        
        # PDE residual: ∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²) - g = 0
        residuals = u_t - alpha * (u_xx + u_yy) - g
        
        return residuals, u, g


class HeatConductionDataset:
    """Dataset generation and management for heat conduction problem"""
    
    def __init__(self, Lx=1.0, Ly=1.0, T_final=1.0, device='cpu'):
        """
        Args:
            Lx: Length of domain in x-direction
            Ly: Length of domain in y-direction
            T_final: Final time
            device: torch device
        """
        self.Lx = Lx
        self.Ly = Ly
        self.T_final = T_final
        self.device = device
    
    def generate_synthetic_heat_source(self, x, y, t):
        """
        Generate synthetic heat source function
        g(x,y,t) = sin(πx/Lx) * sin(πy/Ly) * sin(πt/T_final)
        
        Args:
            x, y, t: Coordinate arrays
        
        Returns:
            Heat source values
        """
        return (np.sin(np.pi * x / self.Lx) * 
                np.sin(np.pi * y / self.Ly) * 
                np.sin(np.pi * t / self.T_final))
    
    def generate_synthetic_temperature(self, x, y, t, alpha=0.1):
        """
        Generate synthetic temperature solution
        For the problem with known heat source, we use an analytical approximation
        
        Args:
            x, y, t: Coordinate arrays
            alpha: Thermal diffusivity
        
        Returns:
            Temperature values
        """
        # Simplified solution based on eigenfunction expansion
        u = (np.sin(np.pi * x / self.Lx) * 
             np.sin(np.pi * y / self.Ly) * 
             (1 - np.exp(-alpha * (2 * np.pi**2 / (self.Lx**2 + self.Ly**2)) * t)))
        
        # Add contribution from heat source
        source_contribution = (self.generate_synthetic_heat_source(x, y, t) * t * 0.1)
        
        return u + source_contribution
    
    def generate_training_data(self, n_interior=5000, n_boundary=1000, n_sensors=50, noise_level=0.02):
        """
        Generate training data
        
        Args:
            n_interior: Number of interior (collocation) points
            n_boundary: Number of boundary points
            n_sensors: Number of sensor measurements
            noise_level: Noise standard deviation for sensor data
        
        Returns:
            Dictionary with training data
        """
        # Interior (collocation) points
        np.random.seed(42)
        x_interior = np.random.uniform(0.1, self.Lx - 0.1, n_interior)
        y_interior = np.random.uniform(0.1, self.Ly - 0.1, n_interior)
        t_interior = np.random.uniform(0.01, self.T_final - 0.01, n_interior)
        
        # Boundary points
        x_boundary = np.concatenate([
            np.random.uniform(0, self.Lx, n_boundary // 4),  # x-edges
            np.full(n_boundary // 4, 0),                      # x=0
            np.full(n_boundary // 4, self.Lx),                # x=Lx
            np.random.uniform(0, self.Lx, n_boundary // 4)
        ])
        y_boundary = np.concatenate([
            np.full(n_boundary // 4, 0),                      # y=0
            np.full(n_boundary // 4, self.Ly),                # y=Ly
            np.random.uniform(0, self.Ly, n_boundary // 4),
            np.random.uniform(0, self.Ly, n_boundary // 4)
        ])
        t_boundary = np.random.uniform(0, self.T_final, n_boundary)
        
        # Sensor measurements (sparse observations)
        x_sensors = np.random.uniform(0.2, self.Lx - 0.2, n_sensors)
        y_sensors = np.random.uniform(0.2, self.Ly - 0.2, n_sensors)
        t_sensors = np.random.uniform(0.1, self.T_final, n_sensors)
        
        u_sensors = self.generate_synthetic_temperature(x_sensors, y_sensors, t_sensors)
        u_sensors += np.random.normal(0, noise_level * np.max(np.abs(u_sensors)), n_sensors)
        
        # Boundary condition (zero temperature at boundaries)
        u_boundary = np.zeros_like(t_boundary)
        
        # Convert to tensors
        data = {
            'x_interior': torch.tensor(x_interior, dtype=torch.float32, device=self.device),
            'y_interior': torch.tensor(y_interior, dtype=torch.float32, device=self.device),
            't_interior': torch.tensor(t_interior, dtype=torch.float32, device=self.device),
            'x_boundary': torch.tensor(x_boundary, dtype=torch.float32, device=self.device),
            'y_boundary': torch.tensor(y_boundary, dtype=torch.float32, device=self.device),
            't_boundary': torch.tensor(t_boundary, dtype=torch.float32, device=self.device),
            'u_boundary': torch.tensor(u_boundary, dtype=torch.float32, device=self.device),
            'x_sensors': torch.tensor(x_sensors, dtype=torch.float32, device=self.device),
            'y_sensors': torch.tensor(y_sensors, dtype=torch.float32, device=self.device),
            't_sensors': torch.tensor(t_sensors, dtype=torch.float32, device=self.device),
            'u_sensors': torch.tensor(u_sensors, dtype=torch.float32, device=self.device),
        }
        
        return data


class PINNTrainer:
    """Trainer for PINN model"""
    
    def __init__(self, model, data, device='cpu', alpha=0.1):
        """
        Args:
            model: PINN model
            data: Training data dictionary
            device: torch device
            alpha: Thermal diffusivity
        """
        self.model = model.to(device)
        self.data = data
        self.device = device
        self.alpha = alpha
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        self.losses = {'total': [], 'pde': [], 'boundary': [], 'data': []}
        self.best_loss = float('inf')
    
    def compute_loss(self, lambda_pde=1.0, lambda_boundary=10.0, lambda_data=10.0):
        """
        Compute total loss
        
        L_total = λ_pde * L_pde + λ_boundary * L_boundary + λ_data * L_data
        
        Args:
            lambda_pde: Weight for PDE residual loss
            lambda_boundary: Weight for boundary condition loss
            lambda_data: Weight for data fitting loss
        
        Returns:
            Total loss and component losses
        """
        # PDE residual loss
        x_physics = torch.stack([
            self.data['x_interior'],
            self.data['y_interior'],
            self.data['t_interior']
        ], dim=1)
        
        residuals, _, _ = self.model.compute_residuals(x_physics, self.alpha)
        loss_pde = torch.mean(residuals ** 2)
        
        # Boundary condition loss
        x_boundary = torch.stack([
            self.data['x_boundary'],
            self.data['y_boundary'],
            self.data['t_boundary']
        ], dim=1)
        
        u_boundary_pred = self.model.forward_u(x_boundary)
        loss_boundary = torch.mean((u_boundary_pred - self.data['u_boundary'].unsqueeze(-1)) ** 2)
        
        # Data fitting loss (sensor measurements)
        x_sensors = torch.stack([
            self.data['x_sensors'],
            self.data['y_sensors'],
            self.data['t_sensors']
        ], dim=1)
        
        u_sensors_pred = self.model.forward_u(x_sensors)
        loss_data = torch.mean((u_sensors_pred - self.data['u_sensors'].unsqueeze(-1)) ** 2)
        
        # Total loss
        loss_total = lambda_pde * loss_pde + lambda_boundary * loss_boundary + lambda_data * loss_data
        
        return loss_total, loss_pde, loss_boundary, loss_data
    
    def train_step(self, lambda_pde=1.0, lambda_boundary=10.0, lambda_data=10.0):
        """Single training step"""
        self.optimizer.zero_grad()
        
        loss_total, loss_pde, loss_boundary, loss_data = self.compute_loss(
            lambda_pde, lambda_boundary, lambda_data
        )
        
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss_total, loss_pde, loss_boundary, loss_data
    
    def train(self, epochs=10000, print_freq=500, lambda_pde=1.0, lambda_boundary=10.0, lambda_data=10.0):
        """Train the model"""
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            loss_total, loss_pde, loss_boundary, loss_data = self.train_step(
                lambda_pde, lambda_boundary, lambda_data
            )
            
            self.losses['total'].append(loss_total.item())
            self.losses['pde'].append(loss_pde.item())
            self.losses['boundary'].append(loss_boundary.item())
            self.losses['data'].append(loss_data.item())
            
            if loss_total.item() < self.best_loss:
                self.best_loss = loss_total.item()
            
            if (epoch + 1) % print_freq == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss = {loss_total.item():.6e}, "
                      f"L_PDE = {loss_pde.item():.6e}, "
                      f"L_BC = {loss_boundary.item():.6e}, "
                      f"L_Data = {loss_data.item():.6e}")
            
            self.scheduler.step()
    
    def plot_losses(self):
        """Plot training losses"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.semilogy(self.losses['total'], label='Total Loss', linewidth=2)
        ax.semilogy(self.losses['pde'], label='PDE Loss', linewidth=2)
        ax.semilogy(self.losses['boundary'], label='Boundary Loss', linewidth=2)
        ax.semilogy(self.losses['data'], label='Data Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def predict(self, x, y, t):
        """
        Make predictions on new points
        
        Args:
            x, y, t: Coordinate arrays
        
        Returns:
            u_pred: Predicted temperature
            g_pred: Predicted heat source
        """
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
            t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
            
            x_input = torch.stack([x_tensor, y_tensor, t_tensor], dim=1)
            
            u_pred = self.model.forward_u(x_input).cpu().numpy()
            g_pred = self.model.forward_g(x_input).cpu().numpy()
        
        return u_pred, g_pred


def plot_solution(trainer, dataset, time_slice=0.5):
    """
    Plot solution contours and comparisons
    
    Args:
        trainer: Trained PINNTrainer
        dataset: Dataset object
        time_slice: Time at which to plot solution
    """
    # Create grid for visualization
    x_plot = np.linspace(0, dataset.Lx, 100)
    y_plot = np.linspace(0, dataset.Ly, 100)
    X, Y = np.meshgrid(x_plot, y_plot)
    
    T = np.full_like(X, time_slice)
    
    # Predictions
    u_pred, g_pred = trainer.predict(X.flatten(), Y.flatten(), T.flatten())
    u_pred = u_pred.reshape(X.shape)
    g_pred = g_pred.reshape(X.shape)
    
    # Exact solution
    u_exact = dataset.generate_synthetic_temperature(X, Y, T)
    g_exact = dataset.generate_synthetic_heat_source(X, Y, T)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Temperature predictions
    ax1 = fig.add_subplot(gs[0, 0])
    contour1 = ax1.contourf(X, Y, u_pred, levels=20, cmap='RdYlBu_r')
    ax1.set_title('Predicted Temperature u(x,y,t)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(contour1, ax=ax1)
    
    # Temperature exact
    ax2 = fig.add_subplot(gs[0, 1])
    contour2 = ax2.contourf(X, Y, u_exact, levels=20, cmap='RdYlBu_r')
    ax2.set_title('Exact Temperature u(x,y,t)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(contour2, ax=ax2)
    
    # Temperature error
    ax3 = fig.add_subplot(gs[0, 2])
    u_error = np.abs(u_pred - u_exact)
    contour3 = ax3.contourf(X, Y, u_error, levels=20, cmap='hot')
    ax3.set_title('Temperature Error |u_pred - u_exact|', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plt.colorbar(contour3, ax=ax3)
    
    # Heat source predictions
    ax4 = fig.add_subplot(gs[1, 0])
    contour4 = ax4.contourf(X, Y, g_pred, levels=20, cmap='seismic')
    ax4.set_title('Predicted Heat Source g(x,y,t)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(contour4, ax=ax4)
    
    # Heat source exact
    ax5 = fig.add_subplot(gs[1, 1])
    contour5 = ax5.contourf(X, Y, g_exact, levels=20, cmap='seismic')
    ax5.set_title('Exact Heat Source g(x,y,t)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(contour5, ax=ax5)
    
    # Heat source error
    ax6 = fig.add_subplot(gs[1, 2])
    g_error = np.abs(g_pred - g_exact)
    contour6 = ax6.contourf(X, Y, g_error, levels=20, cmap='hot')
    ax6.set_title('Heat Source Error |g_pred - g_exact|', fontsize=12, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.colorbar(contour6, ax=ax6)
    
    fig.suptitle(f'Solution Comparison at t = {time_slice}', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def compute_metrics(trainer, dataset, x_test, y_test, t_test, u_exact, g_exact):
    """Compute error metrics"""
    u_pred, g_pred = trainer.predict(x_test, y_test, t_test)
    
    u_mae = np.mean(np.abs(u_pred - u_exact.reshape(-1, 1)))
    u_rmse = np.sqrt(np.mean((u_pred - u_exact.reshape(-1, 1)) ** 2))
    u_relative = u_rmse / (np.max(np.abs(u_exact)) + 1e-10)
    
    g_mae = np.mean(np.abs(g_pred - g_exact.reshape(-1, 1)))
    g_rmse = np.sqrt(np.mean((g_pred - g_exact.reshape(-1, 1)) ** 2))
    g_relative = g_rmse / (np.max(np.abs(g_exact)) + 1e-10)
    
    metrics = {
        'u_mae': u_mae,
        'u_rmse': u_rmse,
        'u_relative_error': u_relative,
        'g_mae': g_mae,
        'g_rmse': g_rmse,
        'g_relative_error': g_relative
    }
    
    return metrics


if __name__ == "__main__":
    print("="*80)
    print("Physics-Informed Neural Networks for Heat Conduction Inverse Problem")
    print("="*80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Create dataset
    dataset = HeatConductionDataset(Lx=1.0, Ly=1.0, T_final=1.0, device=device)
    
    # Generate training data
    print("Generating training data...")
    data = dataset.generate_training_data(n_interior=5000, n_boundary=1000, n_sensors=50, noise_level=0.02)
    print(f"Interior points: {len(data['x_interior'])}")
    print(f"Boundary points: {len(data['x_boundary'])}")
    print(f"Sensor measurements: {len(data['x_sensors'])}\n")
    
    # Create model
    print("Building PINN model...")
    model = PINN(layers_u=[3, 128, 128, 128, 128, 1], 
                 layers_g=[3, 64, 64, 64, 1],
                 activation='tanh')
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    # Train model
    trainer = PINNTrainer(model, data, device=device, alpha=0.1)
    trainer.train(epochs=10000, print_freq=1000, 
                  lambda_pde=1.0, lambda_boundary=10.0, lambda_data=10.0)
    
    # Save model
    torch.save(model.state_dict(), 'pinn_model.pth')
    print("\nModel saved as 'pinn_model.pth'")
    
    # Generate test data
    print("\nGenerating test data for evaluation...")
    x_test = np.random.uniform(0.1, 0.9, 500)
    y_test = np.random.uniform(0.1, 0.9, 500)
    t_test = np.random.uniform(0.1, 0.9, 500)
    
    u_exact_test = dataset.generate_synthetic_temperature(x_test, y_test, t_test)
    g_exact_test = dataset.generate_synthetic_heat_source(x_test, y_test, t_test)
    
    # Compute metrics
    metrics = compute_metrics(trainer, dataset, x_test, y_test, t_test, u_exact_test, g_exact_test)
    
    print("\n" + "="*80)
    print("RESULTS AND METRICS")
    print("="*80)
    print(f"\nTemperature Field u(x,y,t):")
    print(f"  MAE:             {metrics['u_mae']:.6e}")
    print(f"  RMSE:            {metrics['u_rmse']:.6e}")
    print(f"  Relative Error:  {metrics['u_relative_error']:.6f} ({metrics['u_relative_error']*100:.2f}%)")
    
    print(f"\nHeat Source Function g(x,y,t):")
    print(f"  MAE:             {metrics['g_mae']:.6e}")
    print(f"  RMSE:            {metrics['g_rmse']:.6e}")
    print(f"  Relative Error:  {metrics['g_relative_error']:.6f} ({metrics['g_relative_error']*100:.2f}%)")
    
    # Visualizations
    print("\nGenerating visualizations...")
    fig_losses = trainer.plot_losses()
    fig_losses.savefig('losses.png', dpi=150, bbox_inches='tight')
    print("Saved: losses.png")
    
    fig_solution = plot_solution(trainer, dataset, time_slice=0.5)
    fig_solution.savefig('solution_at_t05.png', dpi=150, bbox_inches='tight')
    print("Saved: solution_at_t05.png")
    
    plt.show()
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
