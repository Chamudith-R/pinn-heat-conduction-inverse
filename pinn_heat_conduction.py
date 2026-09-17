"""PINN inverse heat-conduction demo with reproducible diagnostics and outputs.

PDE: u_t - alpha (u_xx + u_yy) - g = 0, on [0,Lx]x[0,Ly]x[0,T].
The manufactured solution is used only to create synthetic measurements and to
make quantitative evaluation possible; the PINN never sees true g during training.
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


def seed_all(seed=1234):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Problem:
    def __init__(self, Lx=1., Ly=1., T=5., alpha=.10, device="cpu"):
        self.Lx, self.Ly, self.T, self.alpha = Lx, Ly, T, alpha
        self.device = device

    def u_true(self, x, y, t):
        # Zero Dirichlet boundaries and zero initial field.
        spatial = np.sin(np.pi*x/self.Lx) * np.sin(np.pi*y/self.Ly)
        temporal = np.sin(2*np.pi*t/self.T) + .35*np.sin(np.pi*t/self.T)
        return spatial * temporal

    def g_true(self, x, y, t):
        # Derived analytically from u_t-alpha Laplacian(u).
        spatial = np.sin(np.pi*x/self.Lx) * np.sin(np.pi*y/self.Ly)
        temporal = np.sin(2*np.pi*t/self.T) + .35*np.sin(np.pi*t/self.T)
        dtemporal = (2*np.pi/self.T)*np.cos(2*np.pi*t/self.T) + (.35*np.pi/self.T)*np.cos(np.pi*t/self.T)
        laplace_factor = -np.pi**2*(1/self.Lx**2 + 1/self.Ly**2)
        return spatial * (dtemporal - self.alpha*laplace_factor*temporal)

    def make_data(self, n_collocation=2500, n_boundary=600, n_sensors=100,
                  n_sensor_times=40, noise_std=.02):
        r = np.random.default_rng(1234)
        xi = r.uniform(0, self.Lx, n_collocation)
        yi = r.uniform(0, self.Ly, n_collocation)
        ti = r.uniform(0, self.T, n_collocation)
        # Four boundary edges, including t=0 initial condition points.
        q = n_boundary // 4
        xb = np.r_[r.uniform(0,self.Lx,q), r.uniform(0,self.Lx,q),
                   np.zeros(q), np.full(q,self.Lx)]
        yb = np.r_[np.zeros(q), np.full(q,self.Ly), r.uniform(0,self.Ly,q),
                   r.uniform(0,self.Ly,q)]
        tb = r.uniform(0, self.T, len(xb))
        # Fixed spatial sensor locations, time series at each sensor.
        xs = r.uniform(.05*self.Lx, .95*self.Lx, n_sensors)
        ys = r.uniform(.05*self.Ly, .95*self.Ly, n_sensors)
        ts = np.linspace(0, self.T, n_sensor_times)
        Xs, Ts = np.meshgrid(xs, ts, indexing="ij")
        Ys = np.broadcast_to(ys[:,None], Xs.shape)
        Uclean = self.u_true(Xs, Ys, Ts)
        Umeas = Uclean + r.normal(0, noise_std, Uclean.shape)
        # Flattened training tuples (x_i,y_i,t_i,u_meas).
        sensor_table = np.c_[Xs.ravel(), Ys.ravel(), Ts.ravel(), Umeas.ravel()]
        to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)
        return {"interior": to_t(np.c_[xi,yi,ti]), "boundary": to_t(np.c_[xb,yb,tb]),
                "u_boundary": to_t(self.u_true(xb,yb,tb)[:,None]),
                "sensor": to_t(sensor_table), "sensor_x": xs, "sensor_y": ys,
                "sensor_times": ts, "sensor_clean": Uclean, "sensor_meas": Umeas}


class MLP(nn.Module):
    def __init__(self, widths):
        super().__init__(); layers=[]
        for a,b in zip(widths[:-2], widths[1:-1]):
            layers += [nn.Linear(a,b), nn.Tanh()]
        layers.append(nn.Linear(widths[-2], widths[-1])); self.net=nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear): nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, z): return self.net(z)


class PINN(nn.Module):
    def __init__(self):
        super().__init__(); self.u_net=MLP([3,96,96,96,1]); self.g_net=MLP([3,64,64,64,1])
    def u(self,z): return self.u_net(z)
    def g(self,z): return self.g_net(z)
    def residual(self,z,alpha):
        z = z.clone().detach().requires_grad_(True); u=self.u(z); g=self.g(z)
        du=torch.autograd.grad(u.sum(),z,create_graph=True)[0]
        uxx=torch.autograd.grad(du[:,0].sum(),z,create_graph=True)[0][:,0:1]
        uyy=torch.autograd.grad(du[:,1].sum(),z,create_graph=True)[0][:,1:2]
        return du[:,2:3]-alpha*(uxx+uyy)-g


class Trainer:
    def __init__(self, model, data, problem, h1_weight=0., lr=1e-3):
        self.model, self.data, self.p = model, data, problem
        self.h1_weight=h1_weight; self.opt=torch.optim.Adam(model.parameters(),lr=lr)
        self.history={k:[] for k in ["total","data","physics","boundary","regularization"]}
    def losses(self):
        r=self.model.residual(self.data["interior"],self.p.alpha)
        lp=(r*r).mean(); b=self.model.u(self.data["boundary"])
        lb=((b-self.data["u_boundary"])**2).mean()
        s=self.data["sensor"]; pred=self.model.u(s[:,:3])
        ld=((pred-s[:,3:4])**2).mean()
        # Sobolev H1 data regularization: sensor-space gradients are penalized.
        z=s[:,:3].clone().detach().requires_grad_(True)
        grad=torch.autograd.grad(self.model.u(z).sum(),z,create_graph=True)[0]
        reg=(grad[:,:2]**2).mean()
        return lp,lb,ld,reg
    def train(self, epochs=3000, weights=(1.,10.,20.), print_every=250):
        wp,wb,wd=weights
        for e in range(1,epochs+1):
            self.opt.zero_grad(); lp,lb,ld,reg=self.losses()
            total=wp*lp+wb*lb+wd*ld+self.h1_weight*reg
            total.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
            self.opt.step()
            for k,v in zip(self.history,[total,ld,lp,lb,self.h1_weight*reg]): self.history[k].append(float(v.detach()))
            if e%print_every==0: print(f"epoch {e:5d} total={total.item():.3e} data={ld.item():.3e} pde={lp.item():.3e}")
    def predict(self,x,y,t):
        z=torch.tensor(np.c_[np.asarray(x).ravel(),np.asarray(y).ravel(),np.asarray(t).ravel()],dtype=torch.float32,device=self.p.device)
        with torch.no_grad(): u=self.model.u(z).cpu().numpy().ravel(); g=self.model.g(z).cpu().numpy().ravel()
        return u,g


def metrics(y, yp):
    y=np.asarray(y).ravel(); yp=np.asarray(yp).ravel(); ss=np.sum((y-y.mean())**2)
    return {"relative_l2": float(np.linalg.norm(yp-y)/(np.linalg.norm(y)+1e-12)),
            "rmse": float(np.sqrt(np.mean((yp-y)**2))),
            "r2": float(1-np.sum((yp-y)**2)/(ss+1e-12))}


def save_heatmap(x,y,z,title,path,cmap="viridis",vmin=None,vmax=None):
    fig,ax=plt.subplots(figsize=(5,4)); im=ax.imshow(z,origin="lower",extent=[x.min(),x.max(),y.min(),y.max()],aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set(xlabel="x",ylabel="y",title=title); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def make_snapshot_figures(model, p, times, nx=80, ny=80):
    x=np.linspace(0,p.Lx,nx); y=np.linspace(0,p.Ly,ny); X,Y=np.meshgrid(x,y,indexing="xy")
    arrays={"x":x,"y":y,"times":np.asarray(times),"u_true":[],"g_true":[],"u_pred":[],"g_pred":[]}
    for t in times:
        U=p.u_true(X,Y,t); G=p.g_true(X,Y,t); up,gp=Trainer(model,{},p).predict(X,Y,np.full_like(X,t))
        arrays["u_true"].append(U); arrays["g_true"].append(G); arrays["u_pred"].append(up.reshape(Y.shape)); arrays["g_pred"].append(gp.reshape(Y.shape))
        for name,z,cmap in [(f"u_true_t{t:g}",U,"magma"),(f"g_true_t{t:g}",G,"coolwarm"),(f"u_pred_t{t:g}",arrays["u_pred"][-1],"magma"),(f"g_pred_t{t:g}",arrays["g_pred"][-1],"coolwarm")]: save_heatmap(x,y,z,name,OUT/(name+".png"),cmap)
        save_heatmap(x,y,np.abs(arrays["u_pred"][-1]-U),f"u_abs_error_t{t:g}",OUT/f"u_abs_error_t{t:g}.png","hot")
        save_heatmap(x,y,np.abs(arrays["g_pred"][-1]-G),f"g_abs_error_t{t:g}",OUT/f"g_abs_error_t{t:g}.png","hot")
    np.savez_compressed(OUT/"ground_truth_and_predictions.npz",**{k:np.asarray(v) for k,v in arrays.items()})
    return arrays


def diagnostics(trainer, data):
    fig,ax=plt.subplots(figsize=(8,5))
    for k,v in trainer.history.items(): ax.semilogy(np.maximum(v,1e-14),label=k)
    ax.set(xlabel="epoch",ylabel="loss",title="Training diagnostics"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"training_losses.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(data["sensor_x"],data["sensor_y"],s=12); ax.set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",title="100 sensor locations"); fig.tight_layout(); fig.savefig(OUT/"sensor_layout.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for i in range(min(5,len(data["sensor_x"]))): ax.plot(data["sensor_times"],data["sensor_meas"][i],".-",label=f"sensor {i}")
    ax.set(xlabel="time",ylabel="temperature",title="Noisy sensor readings"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"sensor_readings.png",dpi=160); plt.close(fig)
    np.savetxt(OUT/"sensor_training_tuples.csv",np.asarray(data["sensor"].cpu()),delimiter=",",header="x,y,t,u_meas",comments="")


def evaluate(model,p,times,nx=70,ny=70):
    x=np.linspace(0,p.Lx,nx); y=np.linspace(0,p.Ly,ny); X,Y=np.meshgrid(x,y)
    rows=[]; r2u=[]; r2g=[]
    for t in times:
        ut=p.u_true(X,Y,t); gt=p.g_true(X,Y,t); up,gp=Trainer(model,{},p).predict(X,Y,np.full_like(X,t))
        mu=metrics(ut,up); mg=metrics(gt,gp); rows.append({"time":float(t),**{"u_"+k:v for k,v in mu.items()},**{"g_"+k:v for k,v in mg.items()}}); r2u.append(mu["r2"]); r2g.append(mg["r2"])
    global_row={"time":"global",**{"u_"+k:v for k,v in metrics(p.u_true(X,Y[:,None],np.asarray(times)[:,None,None]),np.asarray([])).items()}} if False else None
    # Global metrics over all requested snapshots.
    true_u=[]; pred_u=[]; true_g=[]; pred_g=[]
    for t in times:
        ut=p.u_true(X,Y,t);gt=p.g_true(X,Y,t);up,gp=Trainer(model,{},p).predict(X,Y,np.full_like(X,t));true_u+=list(ut.ravel());pred_u+=list(up);true_g+=list(gt.ravel());pred_g+=list(gp)
    rows.append({"time":"global",**{"u_"+k:v for k,v in metrics(true_u,pred_u).items()},**{"g_"+k:v for k,v in metrics(true_g,pred_g).items()}})
    with open(OUT/"metrics.csv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT/"metrics.json","w") as f: json.dump(rows,f,indent=2)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(times,r2u,"o-",label="u R²");ax.plot(times,r2g,"o-",label="g R²");ax.axhline(0,color="k",lw=.7);ax.set(xlabel="time",ylabel="R²",title="R² versus time");ax.legend();fig.tight_layout();fig.savefig(OUT/"r2_vs_time.png",dpi=160);plt.close(fig)
    print(json.dumps(rows[-1],indent=2)); return rows


def main():
    seed_all(); OUT.mkdir(exist_ok=True); device="cuda" if torch.cuda.is_available() else "cpu"; print("device:",device)
    p=Problem(T=5.,device=device); data=p.make_data(n_collocation=2500,n_boundary=600,n_sensors=100,n_sensor_times=40,noise_std=.02)
    np.savez_compressed(OUT/"sensor_data.npz",sensor_training_tuples=np.asarray(data["sensor"].cpu()),sensor_clean=data["sensor_clean"],sensor_meas=data["sensor_meas"],sensor_x=data["sensor_x"],sensor_y=data["sensor_y"],sensor_times=data["sensor_times"])
    model=PINN().to(device); trainer=Trainer(model,data,p,h1_weight=1e-4); diagnostics(trainer,data)
    trainer.train(epochs=3000,print_every=250); torch.save(model.state_dict(),OUT/"pinn_model.pt")
    diagnostics(trainer,data); times=[0.,1.25,2.5,3.75,5.]; make_snapshot_figures(model,p,times); evaluate(model,p,times)
    print("All outputs saved under outputs/")

if __name__=="__main__": main()
