"""
PINN inverse heat-conduction demo with diagnostics and output generation.

PDE:
    u_t - alpha (u_xx + u_yy) - g = 0
on [0,Lx] x [0,Ly] x [0,T].

We manufacture a synthetic ground-truth pair (u_true, g_true) so that we can
generate sensor data and then evaluate the inversion quality.
"""

from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def seed_all(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Problem:
    def __init__(self, Lx=1.0, Ly=1.0, T=5.0, alpha=0.10, device="cpu"):
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.alpha = alpha
        self.device = device

    def u_true(self, x, y, t):
        x = np.asarray(x)
        y = np.asarray(y)
        t = np.asarray(t)

        spatial = np.sin(np.pi * x / self.Lx) * np.sin(np.pi * y / self.Ly)
        temporal = np.sin(2 * np.pi * t / self.T) + 0.35 * np.sin(np.pi * t / self.T)
        return spatial * temporal

    def g_true(self, x, y, t):
        x = np.asarray(x)
        y = np.asarray(y)
        t = np.asarray(t)

        spatial = np.sin(np.pi * x / self.Lx) * np.sin(np.pi * y / self.Ly)
        temporal = np.sin(2 * np.pi * t / self.T) + 0.35 * np.sin(np.pi * t / self.T)
        dtemporal = (
            (2 * np.pi / self.T) * np.cos(2 * np.pi * t / self.T)
            + (0.35 * np.pi / self.T) * np.cos(np.pi * t / self.T)
        )
        laplace_factor = -(np.pi ** 2) * (1.0 / self.Lx ** 2 + 1.0 / self.Ly ** 2)
        return spatial * (dtemporal - self.alpha * laplace_factor * temporal)

    def make_data(
        self,
        n_collocation=2500,
        n_boundary=600,
        n_sensors=100,
        n_sensor_times=40,
        noise_std=0.02,
    ):
        rng = np.random.default_rng(1234)

        # Collocation points in the interior
        xi = rng.uniform(0.0, self.Lx, n_collocation)
        yi = rng.uniform(0.0, self.Ly, n_collocation)
        ti = rng.uniform(0.0, self.T, n_collocation)

        # Boundary points over the four edges
        q = n_boundary // 4
        xb = np.r_[
            rng.uniform(0, self.Lx, q),
            rng.uniform(0, self.Lx, q),
            np.zeros(q),
            np.full(q, self.Lx),
        ]
        yb = np.r_[
            np.zeros(q),
            np.full(q, self.Ly),
            rng.uniform(0, self.Ly, q),
            rng.uniform(0, self.Ly, q),
        ]
        tb = rng.uniform(0, self.T, len(xb))

        # Sensor locations and noisy measurements over time
        xs = rng.uniform(0.05 * self.Lx, 0.95 * self.Lx, n_sensors)
        ys = rng.uniform(0.05 * self.Ly, 0.95 * self.Ly, n_sensors)

        ts = np.linspace(0.0, self.T, n_sensor_times)
        Xs, Ts = np.meshgrid(xs, ts, indexing="ij")
        Ys = np.broadcast_to(ys[:, None], Xs.shape)

        Uclean = self.u_true(Xs, Ys, Ts)
        Umeas = Uclean + rng.normal(0.0, noise_std, Uclean.shape)

        sensor_table = np.c_[Xs.ravel(), Ys.ravel(), Ts.ravel(), Umeas.ravel()]

        def to_t(a):
            return torch.tensor(a, dtype=torch.float32, device=self.device)

        return {
            "interior": to_t(np.c_[xi, yi, ti]),
            "boundary": to_t(np.c_[xb, yb, tb]),
            "u_boundary": to_t(self.u_true(xb, yb, tb).reshape(-1, 1)),
            "sensor": to_t(sensor_table),
            "sensor_x": xs,
            "sensor_y": ys,
            "sensor_times": ts,
            "sensor_clean": Uclean,
            "sensor_meas": Umeas,
        }


class MLP(nn.Module):
    def __init__(self, widths):
        super().__init__()
        layers = []
        for a, b in zip(widths[:-2], widths[1:-1]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(widths[-2], widths[-1]))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_net = MLP([3, 96, 96, 96, 1])
        self.g_net = MLP([3, 64, 64, 64, 1])

    def u(self, z):
        return self.u_net(z)

    def g(self, z):
        return self.g_net(z)

    def residual(self, z, alpha):
        z = z.clone().detach().requires_grad_(True)
        u = self.u(z)
        g = self.g(z)

        du = torch.autograd.grad(u.sum(), z, create_graph=True)[0]
        uxx = torch.autograd.grad(du[:, 0].sum(), z, create_graph=True)[0][:, 0:1]
        uyy = torch.autograd.grad(du[:, 1].sum(), z, create_graph=True)[0][:, 1:2]
        ut = du[:, 2:3]

        return ut - alpha * (uxx + uyy) - g


class Trainer:
    def __init__(self, model, problem, data, h1_weight=1e-4, lr=1e-3):
        self.model = model
        self.problem = problem
        self.data = data
        self.h1_weight = h1_weight
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)

        self.history = {
            "total": [],
            "data": [],
            "physics": [],
            "boundary": [],
            "regularization": [],
        }

    def losses(self):
        interior = self.data["interior"]
        boundary = self.data["boundary"]
        sensor = self.data["sensor"]

        # PDE residual
        r = self.model.residual(interior, self.problem.alpha)
        physics_loss = (r ** 2).mean()

        # BC loss: zero temperature on boundaries
        u_boundary_pred = self.model.u(boundary)
        boundary_loss = ((u_boundary_pred - self.data["u_boundary"]) ** 2).mean()

        # Data loss at sensors
        sensor_pred = self.model.u(sensor[:, :3])
        data_loss = ((sensor_pred - sensor[:, 3:4]) ** 2).mean()

        # Sobolev-type regularization to stabilize the inverse estimate
        z = sensor[:, :3].clone().detach().requires_grad_(True)
        u_reg = self.model.u(z)
        grad = torch.autograd.grad(u_reg.sum(), z, create_graph=True)[0]
        reg_loss = (grad[:, :2] ** 2).mean()

        return physics_loss, boundary_loss, data_loss, reg_loss

    def train(self, epochs=3000, weights=(1.0, 10.0, 20.0), print_every=250):
        wp, wb, wd = weights

        for epoch in range(1, epochs + 1):
            self.opt.zero_grad()

            lp, lb, ld, reg = self.losses()
            total = wp * lp + wb * lb + wd * ld + self.h1_weight * reg

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()

            self.history["total"].append(float(total.detach()))
            self.history["data"].append(float(ld.detach()))
            self.history["physics"].append(float(lp.detach()))
            self.history["boundary"].append(float(lb.detach()))
            self.history["regularization"].append(float(reg.detach()))

            if epoch % print_every == 0:
                print(
                    f"epoch {epoch:5d} "
                    f"total={total.item():.3e} "
                    f"data={ld.item():.3e} "
                    f"physics={lp.item():.3e} "
                    f"bc={lb.item():.3e}"
                )

    def predict(self, x, y, t):
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        t = np.asarray(t).ravel()

        z = torch.tensor(
            np.c_[x, y, t], dtype=torch.float32, device=self.problem.device
        )
        with torch.no_grad():
            u = self.model.u(z).cpu().numpy().ravel()
            g = self.model.g(z).cpu().numpy().ravel()
        return u, g


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss = np.sum((y_true - y_true.mean()) ** 2)
    rel_l2 = np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + 1e-12)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    r2 = 1.0 - np.sum((y_pred - y_true) ** 2) / (ss + 1e-12)
    return {
        "relative_l2": float(rel_l2),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def save_heatmap(x, y, z, title, path, cmap="viridis", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        z,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set(xlabel="x", ylabel="y", title=title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_truth_and_prediction_snapshots(model, problem, times):
    x = np.linspace(0, problem.Lx, 80)
    y = np.linspace(0, problem.Ly, 80)
    X, Y = np.meshgrid(x, y, indexing="xy")

    for t in times:
        U_true = problem.u_true(X, Y, t)
        G_true = problem.g_true(X, Y, t)

        U_pred, G_pred = model.predict(X, Y, np.full_like(X, t))
        U_pred = U_pred.reshape(X.shape)
        G_pred = G_pred.reshape(X.shape)

        # Save true/predicted fields and abs errors
        save_heatmap(x, y, U_true, f"u_true_t={t:.2f}", OUT / f"u_true_t{t:.2f}.png", cmap="magma")
        save_heatmap(x, y, G_true, f"g_true_t={t:.2f}", OUT / f"g_true_t{t:.2f}.png", cmap="coolwarm")
        save_heatmap(x, y, U_pred, f"u_pred_t={t:.2f}", OUT / f"u_pred_t{t:.2f}.png", cmap="magma")
        save_heatmap(x, y, G_pred, f"g_pred_t={t:.2f}", OUT / f"g_pred_t{t:.2f}.png", cmap="coolwarm")
        save_heatmap(x, y, np.abs(U_pred - U_true), f"u_error_t={t:.2f}", OUT / f"u_error_t{t:.2f}.png", cmap="hot")
        save_heatmap(x, y, np.abs(G_pred - G_true), f"g_error_t={t:.2f}", OUT / f"g_error_t{t:.2f}.png", cmap="hot")

    np.savez_compressed(
        OUT / "ground_truth_and_predictions.npz",
        x=x,
        y=y,
        times=np.asarray(times),
    )


def plot_training_diagnostics(trainer):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in trainer.history.items():
        ax.semilogy(np.maximum(values, 1e-14), label=name)
    ax.set(xlabel="epoch", ylabel="loss", title="Training diagnostics")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "training_losses.png", dpi=160)
    plt.close(fig)


def plot_sensor_layout_and_readings(data):
    # Sensor layout
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["sensor_x"], data["sensor_y"], s=12)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title="Sensor locations")
    fig.tight_layout()
    fig.savefig(OUT / "sensor_layout.png", dpi=160)
    plt.close(fig)

    # Example sensor readings over time
    fig, ax = plt.subplots(figsize=(8, 5))
    nshow = min(5, len(data["sensor_x"]))
    for i in range(nshow):
        ax.plot(data["sensor_times"], data["sensor_meas"][i], ".-", label=f"sensor {i}")
    ax.set(xlabel="time", ylabel="temperature", title="Noisy sensor readings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "sensor_readings.png", dpi=160)
    plt.close(fig)

    np.savez_compressed(
        OUT / "sensor_data.npz",
        sensor_x=data["sensor_x"],
        sensor_y=data["sensor_y"],
        sensor_times=data["sensor_times"],
        sensor_clean=data["sensor_clean"],
        sensor_meas=data["sensor_meas"],
    )

    np.savetxt(
        OUT / "sensor_training_tuples.csv",
        np.asarray(data["sensor"].cpu()),
        delimiter=",",
        header="x,y,t,u_meas",
        comments="",
    )


def evaluate_snapshots(model, problem, times):
    x = np.linspace(0, problem.Lx, 70)
    y = np.linspace(0, problem.Ly, 70)
    X, Y = np.meshgrid(x, y, indexing="xy")

    rows = []

    for t in times:
        U_true = problem.u_true(X, Y, t)
        G_true = problem.g_true(X, Y, t)

        U_pred, G_pred = model.predict(X, Y, np.full_like(X, t))
        U_pred = U_pred.reshape(X.shape)
        G_pred = G_pred.reshape(X.shape)

        u_metrics = metrics(U_true, U_pred)
        g_metrics = metrics(G_true, G_pred)

        rows.append(
            {
                "time": float(t),
                "u_relative_l2": u_metrics["relative_l2"],
                "u_rmse": u_metrics["rmse"],
                "u_r2": u_metrics["r2"],
                "g_relative_l2": g_metrics["relative_l2"],
                "g_rmse": g_metrics["rmse"],
                "g_r2": g_metrics["r2"],
            }
        )

    # Global metrics across all requested snapshot times
    all_u_true = []
    all_u_pred = []
    all_g_true = []
    all_g_pred = []

    for t in times:
        U_true = problem.u_true(X, Y, t)
        G_true = problem.g_true(X, Y, t)
        U_pred, G_pred = model.predict(X, Y, np.full_like(X, t))
        U_pred = U_pred.reshape(X.shape)
        G_pred = G_pred.reshape(X.shape)

        all_u_true.extend(U_true.ravel())
        all_u_pred.extend(U_pred.ravel())
        all_g_true.extend(G_true.ravel())
        all_g_pred.extend(G_pred.ravel())

    global_u = metrics(all_u_true, all_u_pred)
    global_g = metrics(all_g_true, all_g_pred)

    rows.append(
        {
            "time": "global",
            "u_relative_l2": global_u["relative_l2"],
            "u_rmse": global_u["rmse"],
            "u_r2": global_u["r2"],
            "g_relative_l2": global_g["relative_l2"],
            "g_rmse": global_g["rmse"],
            "g_r2": global_g["r2"],
        }
    )

    with open(OUT / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "u_relative_l2",
                "u_rmse",
                "u_r2",
                "g_relative_l2",
                "g_rmse",
                "g_r2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    # R² vs. time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["time"] for r in rows[:-1]], [r["u_r2"] for r in rows[:-1]], "o-", label="u")
    ax.plot([r["time"] for r in rows[:-1]], [r["g_r2"] for r in rows[:-1]], "o-", label="g")
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set(xlabel="time", ylabel="R²", title="R² vs. time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "r2_vs_time.png", dpi=160)
    plt.close(fig)

    print(json.dumps(rows[-1], indent=2))
    return rows


def main():
    seed_all()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    problem = Problem(T=5.0, device=device)
    data = problem.make_data(
        n_collocation=2500,
        n_boundary=600,
        n_sensors=100,
        n_sensor_times=40,
        noise_std=0.02,
    )

    model = PINN().to(device)
    trainer = Trainer(model, problem, data, h1_weight=1e-4, lr=1e-3)

    # 1) Diagnostics and sensor outputs
    plot_sensor_layout_and_readings(data)

    # 2) Training
    trainer.train(epochs=3000, weights=(1.0, 10.0, 20.0), print_every=250)
    plot_training_diagnostics(trainer)

    # 3) Save trained model
    torch.save(model.state_dict(), OUT / "pinn_model.pt")

    # 4) Ground truth snapshots
    times = [0.0, 1.25, 2.5, 3.75, 5.0]
    save_truth_and_prediction_snapshots(model, problem, times)

    # 5) Evaluation metrics
    evaluate_snapshots(model, problem, times)

    print("Outputs saved in:", OUT.resolve())


if __name__ == "__main__":
    main()
