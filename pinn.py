import numpy as np
import os
import torch
import random
from PINNmodel.iij import PINN_NN
#Lowlouを予測する用のアルゴリズム

def grad(u, x):
    return torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

def curl_and_div(B, x, y, z):
    """Return J=curl(B) and div(B). mu0 is omitted."""
    bx = B[:, 0:1]
    by = B[:, 1:2]
    bz = B[:, 2:3]

    bx_x = grad(bx, x)
    bx_y = grad(bx, y)
    bx_z = grad(bx, z)

    by_x = grad(by, x)
    by_y = grad(by, y)
    by_z = grad(by, z)

    bz_x = grad(bz, x)
    bz_y = grad(bz, y)
    bz_z = grad(bz, z)

    jx = bz_y - by_z
    jy = bx_z - bz_x
    jz = by_x - bx_y

    div_b = bx_x + by_y + bz_z

    J = torch.cat([jx, jy, jz], dim=-1)
    return J, div_b

def current_free_losses(model, x, y, z, eps=1.0e-8):
    B = model(x, y, z)

    abs = 1
    abs = torch.sqrt(B[:, 0:1]**2 + B[:, 1:2]**2 + B[:, 2:3]**2)
    print(abs.shape)

    J, div_b = curl_and_div(B, x, y, z)

    # |J|^2
    loss_current = torch.mean(torch.sum(J**2, dim=-1, keepdim=True)/abs**4)

    # |div B|^2
    loss_div = torch.mean(div_b**2/abs**2)

    return loss_current, loss_div


def sample_periodic_b0(b0_grid, x, y, lx, ly):

    ny, nx = b0_grid.shape

    # Periodic wrap
    xw = torch.remainder(x, lx)
    yw = torch.remainder(y, ly)

    # Convert to grid indices
    gx = xw / lx * nx
    gy = yw / ly * ny

    i0 = torch.floor(gx).long() % nx
    j0 = torch.floor(gy).long() % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny

    tx = gx - torch.floor(gx)
    ty = gy - torch.floor(gy)

    b00 = b0_grid[j0.squeeze(-1), i0.squeeze(-1)].unsqueeze(-1)
    b10 = b0_grid[j0.squeeze(-1), i1.squeeze(-1)].unsqueeze(-1)
    b01 = b0_grid[j1.squeeze(-1), i0.squeeze(-1)].unsqueeze(-1)
    b11 = b0_grid[j1.squeeze(-1), i1.squeeze(-1)].unsqueeze(-1)

    return (
        (1.0 - tx) * (1.0 - ty) * b00
        + tx * (1.0 - ty) * b10
        + (1.0 - tx) * ty * b01
        + tx * ty * b11
    )


def make_points(n, lx, ly, lz, device):
    x = lx * torch.rand(n, 1, device=device, requires_grad=True)
    y = ly * torch.rand(n, 1, device=device, requires_grad=True)
    z = lz * torch.rand(n, 1, device=device, requires_grad=True)
    return x, y, z


def make_points_near_top(n, lx, ly, lz, device, thickness=0.2):
    x = lx * torch.rand(n, 1, device=device, requires_grad=True)
    y = ly * torch.rand(n, 1, device=device, requires_grad=True)

    # z in [(1-thickness)*lz, lz]
    z = lz * (1.0 - thickness * torch.rand(n, 1, device=device))
    z.requires_grad_(True)

    return x, y, z

def Cov_weighting(itr, Lt1, Lt2, Lt3, Lt4,l1, l2, l3, l4):#自動で重みを決定する関数
    if itr == 0:
        return [1/5, 1/5, 1/5, 1/5]
        
    else:
        lt1 = Lt1[itr]/np.mean(Lt1[:itr])
        lt2 = Lt2[itr]/np.mean(Lt2[:itr])
        lt3 = Lt3[itr]/np.mean(Lt3[:itr])
        lt4 = Lt4[itr]/np.mean(Lt4[:itr])


        l1.append(lt1)
        l2.append(lt2)
        l3.append(lt3)
        l4.append(lt4)

        if itr > 10:
            clt1 = np.std(l1)/np.mean(l1)
            clt2 = np.std(l2)/np.mean(l2)
            clt3 = np.std(l3)/np.mean(l3)
            clt4 = np.std(l4)/np.mean(l4)
            zt = clt1 + clt2 + clt3 + clt4 

            weight1 = (1/zt)*clt1
            weight2 = (1/zt)*clt2
            weight3 = (1/zt)*clt3
            weight4 = (1/zt)*clt4
    

            return [weight1, weight2, weight3, weight4]

        else:
            return [1/5, 1/5, 1/5, 1/5]

def save_checkpoint(path, model, optimizer, step):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)




def train(bx0, by0, bz0, lx, ly, lz, steps=20000, n_pde=4096, n_bc=2048, lr=1.0e-3, T_max=3000, eta_min=1e-5, device="cuda", ckpt_dir = None, loss_path = None):

    
    bx0 = torch.as_tensor(bx0, dtype=torch.float32, device=device)
    by0 = torch.as_tensor(by0, dtype=torch.float32, device=device)
    bz0 = torch.as_tensor(bz0, dtype=torch.float32, device=device)
    
    Lt1 = []
    Lt2 = []
    Lt3 = []
    Lt4 = []

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    weight = [1/5, 1/5, 1/5, 1/5]

    model = PINN_NN(lx, ly, lz).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min,)

    for step in range(steps):
        opt.zero_grad()

        # ---- Interior current-free and divB losses ----
        x, y, z = make_points(n_pde, lx, ly, lz, device)
        

        # 上端付近の内部点を少し増やす
        xti, yti, zti = make_points_near_top(n_pde // 2, lx, ly, lz, device)


        # ---- Bottom boundary: Bz = B0 ----
        xb = lx * torch.rand(n_bc, 1, device=device, requires_grad=True)
        yb = ly * torch.rand(n_bc, 1, device=device, requires_grad=True)
        zb = torch.zeros(n_bc, 1, device=device, requires_grad=True)

        B_bottom_pred = model(xb, yb, zb)
        bx_bottom_pred = B_bottom_pred[:, 0:1]
        by_bottom_pred = B_bottom_pred[:, 1:2]        
        bz_bottom_pred = B_bottom_pred[:, 2:3]

        bx_bottom_target = sample_periodic_b0(bx0, xb, yb, lx, ly)
        by_bottom_target = sample_periodic_b0(by0, xb, yb, lx, ly)
        bz_bottom_target = sample_periodic_b0(bz0, xb, yb, lx, ly)


        #すべてのベクトル損失を出す
        loss_bottom_bx = torch.mean((bx_bottom_pred - bx_bottom_target) ** 2)
        loss_bottom_by = torch.mean((by_bottom_pred - by_bottom_target) ** 2)
        loss_bottom_bz = torch.mean((bz_bottom_pred - bz_bottom_target) ** 2)
        loss_bottom = loss_bottom_by + loss_bottom_bz + loss_bottom_bx

        # ---- Top boundary regularization ----
        xt = lx * torch.rand(n_bc, 1, device=device, requires_grad=True)
        yt = ly * torch.rand(n_bc, 1, device=device, requires_grad=True)
        zt = lz * torch.ones(n_bc, 1, device=device, requires_grad=True)

        B_top = model(xt, yt, zt)
        bx_top = B_top[:, 0:1]
        by_top = B_top[:, 1:2]

        # 上端では水平磁場を弱く消す
        loss_top_horizontal = torch.mean(bx_top**2 + by_top**2)

        loss_current, loss_div = current_free_losses(model, x, y, z)
        loss_current_top_i, loss_div_top_i = current_free_losses(
            model, xti, yti, zti
        )

  
        loss_current = loss_current + loss_current_top_i
        loss_div = loss_div + loss_div_top_i

        # ---- Total loss ----
        loss = (1.0 * weight[0] * loss_current+ 1.0 * weight[1] * loss_div+ 10.0 * weight[2] * loss_bottom + 0.1 * weight[3] * loss_top_horizontal)

        Lt1.append(loss_current.item())
        Lt2.append(loss_div.item())
        Lt3.append(loss_bottom_bz.item())
        Lt4.append(loss_top_horizontal.item())
        weight = Cov_weighting(step, Lt1, Lt2, Lt3, Lt4,l1, l2, l3, l4)

        loss.backward()
        opt.step()
        scheduler.step()

        with open(loss_path, "a", encoding="utf-8") as f:
            loss_txt = f"jxb = {loss_current.item()}, div = {loss_div.item()}, loss_bottom = {loss_bottom.item()}, loss_top_horizontal = {loss_top_horizontal.item()}" 
            f.write(loss_txt)
            f.write("\n")

        if step % 500 == 0:
            print(
                step,
                {
                    "loss": float(loss.detach().cpu()),
                    "current": float(loss_current.detach().cpu()),
                    "divB": float(loss_div.detach().cpu()),
                    "bottom_bz": float(loss_bottom_bz.detach().cpu()),
                    "top_h": float(loss_top_horizontal.detach().cpu()),
                },
            )

        if (step + 1) % 50000 == 0:
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step+1}.pt")
                save_checkpoint(ckpt_path, model, opt, step + 1)
                print(f"Checkpoint saved at step {step+1}")

    return model


@torch.no_grad()
def evaluate_field(model, x1d, y1d, z1d, device="cuda"):
    """
    Returns Bx, By, Bz on meshgrid with shape (nz, ny, nx).
    """
    X, Y, Z = torch.meshgrid(
        torch.as_tensor(x1d, dtype=torch.float32, device=device),
        torch.as_tensor(y1d, dtype=torch.float32, device=device),
        torch.as_tensor(z1d, dtype=torch.float32, device=device),
        indexing="xy",
    )

    x = X.reshape(-1, 1)
    y = Y.reshape(-1, 1)
    z = Z.reshape(-1, 1)

    B = model(x, y, z)

    nx = len(x1d)
    ny = len(y1d)
    nz = len(z1d)

    bx = B[:, 0].reshape(ny, nx, nz).permute(0, 1, 2).detach().cpu()
    by = B[:, 1].reshape(ny, nx, nz).permute(0, 1, 2).detach().cpu()
    bz = B[:, 2].reshape(ny, nx, nz).permute(0, 1, 2).detach().cpu()

    return bx, by, bz


def evaluate_residuals(model, x1d, y1d, z1d, device="cuda"):
    """
    Evaluate |J x B| and divB on a grid.
    """
    X, Y, Z = torch.meshgrid(
        torch.as_tensor(x1d, dtype=torch.float32, device=device),
        torch.as_tensor(y1d, dtype=torch.float32, device=device),
        torch.as_tensor(z1d, dtype=torch.float32, device=device),
        indexing="xy",
    )

    x = X.reshape(-1, 1).detach().requires_grad_(True)
    y = Y.reshape(-1, 1).detach().requires_grad_(True)
    z = Z.reshape(-1, 1).detach().requires_grad_(True)

    B = model(x, y, z)
    J, div_b = curl_and_div(B, x, y, z)
    j_cross_b = torch.cross(J, B, dim=-1)

    nx = len(x1d)
    ny = len(y1d)
    nz = len(z1d)

    jxb_abs = torch.sqrt(torch.sum(j_cross_b**2, dim=-1))
    div_b = div_b[:, 0]

    jxb_abs = jxb_abs.reshape(ny, nx, nz).permute(0, 1, 2).detach().cpu()
    div_b = div_b.reshape(ny, nx, nz).permute(0, 1, 2).detach().cpu()

    return jxb_abs, div_b

def main(bx0, by0, bz0, ckpt_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    loss_path = os.path.join(ckpt_dir, "log.txt")
    print("device:", device)

    # domain
    x = np.linspace(0, 1, 64)
    y = np.linspace(0, 1, 64)
    z = np.linspace(0, 1, 64)

    lx = 64
    ly = 64
    lz = 64

    # train
    model = train(bx0 = bx0, by0 = by0, bz0 = bz0, lx=lx, ly=ly, lz=lz, steps=500000, n_pde=4096*4, n_bc=2024, lr=1.0e-3, T_max=150000, eta_min=1e-5, device=device, ckpt_dir = ckpt_dir, loss_path = loss_path)

    # evaluate field
    bx, by, bz = evaluate_field(model, x, y, z, device=device)

    # potential-field reference from FFT
  
    # residuals
    jxb_abs, div_b = evaluate_residuals(model, x, y, z, device=device)
    return bx, by, bz



seed = 1234 
random.seed(seed)          # Python標準乱数
np.random.seed(seed)       # NumPy
torch.manual_seed(seed)    # PyTorch (CPU)


lowlou_f = "data/b_0.210_0.124.npz"
data = np.load(lowlou_f)
Exa_b = data["b"]#(64, 64, 64, 3)
bottom = Exa_b[:, :, 0, :]

Exa_bx = bottom[:, :, 0:1]
Exa_by = bottom[:, :, 1:2]
Exa_bz = bottom[:, :, 2:3]


bx0 = bottom[:, :, 0:1].squeeze(-1)
by0 = bottom[:, :, 1:2].squeeze(-1)
bz0 = bottom[:, :, 2:3].squeeze(-1)



#new model
path = "output"
num  = sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))
os.makedirs(f"{path}/model{num}", exist_ok=True)
model_dir = f"{path}/model{num}"
loss_path = os.path.join(model_dir, "log.txt")
print(model_dir)

main(bx0, by0, bz0, model_dir)

