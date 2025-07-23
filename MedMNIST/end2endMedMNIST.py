import math
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, GaussianBlur
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF

# MedMNIST import
import medmnist
from medmnist import INFO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data setup ---
data_flag = 'chestmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
# Download datasets
train_dataset = DataClass(split='train', download=True)
# test_dataset  = DataClass(split='test' , download=True)   # COMMENTED OUT

# Train / test
numSamples = int(0.8 * len(train_dataset))
# numTest    = len(test_dataset)                           # COMMENTED OUT

# Prepare data tensors --------------------------------------------------------
def _get_tensor(ds, n):
    """Stack first n samples, convert to float32 in [0,1]."""
    imgs = []
    for i in range(n):
        x, _ = ds[i]                 
        if isinstance(x, torch.Tensor):
            t = x.float() / 255.0    
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float() / 255.0
            if t.ndim == 3:          
                t = t.permute(2, 0, 1)
        else:                        
            t = TF.to_tensor(x)      
        imgs.append(t[0])            
    return torch.stack(imgs).to(device)   

X_raw       = _get_tensor(train_dataset, numSamples)        # (N, H, W)
# X_raw_test  = _get_tensor(test_dataset , numTest)         # COMMENTED OUT

dim = X_raw[0].numel()     # 28*28 = 784
r = 200                   # bottleneck / testing rank

# -----------------------------------------------------------------------------
# X (clean)
X = X_raw.view(numSamples, -1).T                           
muX = X.mean(dim=1, keepdim=True)
X  = X - muX
gammaX = (X @ X.T) / (numSamples - 1)
gammaX += 1e-5 * torch.eye(dim, device=device)
L_X = torch.linalg.cholesky(gammaX)

# F matrix (blur)
blurKernelSizeF = 5
blurSigmaF = 1.5
forwardBlur = GaussianBlur(kernel_size=blurKernelSizeF, sigma=blurSigmaF)
basisImages = torch.eye(dim).reshape(dim, 1, 28, 28)
F_cols = []
for j in range(dim):
    F_cols.append(forwardBlur(basisImages[j]).flatten())
F = torch.stack(F_cols, dim=1).to(device)
FX = F @ X

# Y = F(X) + noise
noiseSigma = 0.05
E = torch.randn_like(FX) * noiseSigma
Y = FX + E
muY = Y.mean(dim=1, keepdim=True)
Y = Y - muY

# gammaY and gammaE
gammaE = noiseSigma**2 * torch.eye(dim, device=device)
gammaY = F @ gammaX @ F.T + gammaE 
L_Y = torch.linalg.cholesky(gammaY)

# Theoretical best M_r via SVD
M  = gammaX @ F.T @ torch.linalg.inv(gammaY)
U, S, Vh = torch.linalg.svd(M)

U_r, S_r, Vh_r = U[:, :r], torch.diag(S[:r]), Vh[:r, :]
M_r = U_r @ S_r @ Vh_r

# Theoretical error -----------------------------------------------------------
rel_error_theory_train = (torch.norm(M_r @ Y - X, p="fro") / torch.norm(X, p="fro")).item()
print(f"Theoretical relative training error: {rel_error_theory_train:.4f}")

# --- Test set (COMMENTED OUT) -----------------------------------------------
# X_test = X_raw_test.view(numTest, -1).T
# X_test = X_test - muX

# FX_test = F @ X_test
# E_test  = torch.randn_like(FX_test) * noiseSigma
# Y_test  = FX_test + E_test
# Y_test  = Y_test - muY

# rel_error_theory_test = (torch.norm(M_r @ Y_test - X_test, p="fro") / torch.norm(X_test, p="fro")).item()
# error_theory_test = (torch.norm(M_r @ Y_test - X_test, p="fro") ** 2).item()
# print(f"Theoretical test error: {error_theory_test:.4f}")
# print(f"Theoretical relative test error: {rel_error_theory_test:.4f}")

# ── quick 1×3 preview: (X, F X, F X + E) ─────────────────────────────────────
# place this block **after** Y/E are defined and **before** the big rank-sweep loop
import os
import matplotlib.pyplot as plt

# random training-set index
idx = np.random.randint(0, numSamples)

# (i) original image  →  (H,W)
orig_img = X_raw[idx].cpu()          # already in [0,1]
if orig_img.ndim == 3:               # safety: drop channel dim
    orig_img = orig_img[0]

# (ii) blurred image  F X
blur_img = forwardBlur(orig_img.unsqueeze(0)).squeeze(0).cpu()

# (iii) blurred + noise image  F X + E
noise            = torch.randn_like(blur_img) * noiseSigma
blur_noise_img   = (blur_img + noise).clamp(0, 1).cpu()

# ── plot & save ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
axes[0].imshow(orig_img,        cmap='gray'); axes[0].set_title('Original');      axes[0].axis('off')
axes[1].imshow(blur_img,        cmap='gray'); axes[1].set_title('$F\\!X$');       axes[1].axis('off')
axes[2].imshow(blur_noise_img,  cmap='gray'); axes[2].set_title('$F\\!X+E$');     axes[2].axis('off')
plt.tight_layout()

out_dir = f"end2endpics/MedMNIST"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f"{out_dir}/{data_flag}_mapping{idx}.png", dpi=300)
plt.show()

# --- Autoencoder setup -------------------------------------------------------
batch_size = 128
num_epochs = 350

X_tensor = X.clone().to(torch.float32)
Y_tensor = Y.clone().to(torch.float32)

class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, bottleneck_dim, bias=False)
        self.decoder = nn.Linear(bottleneck_dim, input_dim, bias=False)
    def forward(self, x):
        return self.decoder(self.encoder(x))

model     = LinearAutoencoder(input_dim=dim, bottleneck_dim=r).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop using full Frobenius loss ------------------------------------
train_errors = [] # test_errors removed
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    outputs = model(Y.T)
    target  = X.T
    
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()

    # Evaluate and store error after each epoch
    with torch.no_grad():
        A_learned = model.decoder.weight @ model.encoder.weight

        rel_err_train = (torch.norm(A_learned @ Y - X, p="fro") / torch.norm(X, p='fro'))
        # rel_err_test  = (torch.norm(A_learned @ Y_test - X_test, p="fro") / torch.norm(X_test, p='fro'))  # COMMENTED OUT
        train_errors.append(rel_err_train.item())
        # test_errors .append(rel_err_test .item())                                                    # COMMENTED OUT

# Plot convergence vs. theoretical optimum -----------------------------------
plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs+1), train_errors, label='Training', color='#a7c080')
# plt.plot(range(1, num_epochs+1), test_errors , label='Test  error')                                # COMMENTED OUT
plt.axhline(rel_error_theory_train, ls='--', color='k', label='Optimal')
# plt.axhline(rel_error_theory_test , ls='--', color='r', label='Theoretical optimum (test)')        # COMMENTED OUT
plt.xlabel('Epoch')
plt.ylabel(r'Relative Error $\frac{\|A_{\text{ae}} Y - X\|_{\text{F}}}{\|X\|_{\text{F}}}$')
plt.title(f'Auto-encoder vs. Bayes-optimal ({data_flag})')
plt.legend()
plt.tight_layout()
#plt.savefig(f"end2endpics/MedMNIST/{data_flag}/{data_flag}_loss{numSamples}")
plt.show()

# --- Compact 2×3 gallery with smoothly blended Everforest palette -------------------------
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

# define your Everforest base colors
everforest_colors = [
    "#2d353b",  # dark slate (start)
    "#3d484d",  # 
    "#4f585e",  # slate grays
    "#566358",  # slate-green transition
    "#5a6f47",  # deep forest
    "#7a9c58",  # mid green
    "#a7c080",  # bright green
    "#dbbc7f",  # yellow
    "#e69875",  # orange
    "#e67e80",  # red
    "#dc5456",  # red (end)
]


# build a smooth colormap by interpolating between those
everforest_cmap = LinearSegmentedColormap.from_list(
    'everforest_blend',
    everforest_colors,
    N=256
)

for _ in range(1):
    idx = np.random.randint(0, numSamples)

    orig_img   = X_raw[idx].view(28, 28).cpu()
    y_img      = Y[:, idx].view(28, 28).cpu()
    opt_img    = (M_r @ Y)[:, idx].view(28, 28).cpu()
    learn_img  = (A_learned @ Y)[:, idx].view(28, 28).cpu()

    err_opt   = torch.abs(opt_img   - orig_img)
    err_learn = torch.abs(learn_img - orig_img)
    err_vmin, err_vmax = 0.0, max(err_opt.max(), err_learn.max()).item()

    fig = plt.figure(figsize=(7, 4.2))
    gs  = gridspec.GridSpec(
        2, 4,
        width_ratios=[1, 1, 1, 0.06],
        wspace=0.20, hspace=0.30  # a bit more vertical spacing
    )

    # row 0
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(orig_img, cmap='gray')
    ax.set_title('Original', pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(opt_img, cmap='gray')
    ax.set_title(r'Optimal $M_rY$', pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(err_opt, cmap=everforest_cmap, vmin=0, vmax=1)
    ax.set_title(r'$|M_rY - X|$', pad=4)
    ax.axis('off')

    # row 1
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(y_img, cmap='gray')
    ax.set_title('$Y$', pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(learn_img, cmap='gray')
    ax.set_title(r'Learned $AY$', pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(err_learn, cmap=everforest_cmap, vmin=0, vmax=1)
    ax.set_title(r'$|AY - X|$', pad=4)
    ax.axis('off')

    # color-bar
    cax = fig.add_subplot(gs[:, 3])
    plt.colorbar(im, cax=cax)
    cax.yaxis.tick_right()

    plt.tight_layout()
    plt.show()
    #plt.savefig(f"end2endpics/MedMNIST/{data_flag}/sample{idx}_{numSamples}.png")
    

# ════════════════════════════════════════════════════════════════════════════
# MULTI-DATASET RANK SWEEP  (chestmnist, organamnist, organcmnist,
#                            organsmnist, retinamnist)
# ════════════════════════════════════════════════════════════════════════════
from collections import defaultdict
from copy import deepcopy

# ── pastel-Everforest colours  ───────────────────────────
ef_pastel = ['#a7c080',   # light green
             '#d3869b',   # red/pink
             "#83c0c0",   # aqua
             '#e69875',   # orange
             '#a988b0',   # purple
             "#b8912f"]   # warm yellow/gold
line_styles = {'theory': '-',  'learned': '--'}
markers     = {'theory': 'o',  'learned': 's'}   # ◯ for theory, ■ for learned
lw, ms      = 1.5, 4                            # thin lines, clear markers

# ── helper: run rank sweep on ONE MedMNIST subset ───────────────────────────
def run_rank_sweep(data_flag, ranks, train_epochs=350, lr=1e-3):
    """Return ([theory errors], [AE errors]) for the given dataset."""
    info        = INFO[data_flag]
    DataClass   = getattr(medmnist, info['python_class'])
    train_ds    = DataClass(split='train', download=True)

    N           = int(0.8 * len(train_ds))
    X_raw       = _get_tensor(train_ds, N)                    # uses earlier helper
    dim         = X_raw[0].numel()

    # --- build centred X, Y --------------------------------------------------
    X   = X_raw.view(N, -1).T
    muX = X.mean(dim=1, keepdim=True)
    X   = X - muX
    γX  = (X @ X.T)/(N-1) + 1e-5*torch.eye(dim, device=device)

    blur   = GaussianBlur(kernel_size=5, sigma=1.5)
    basis  = torch.eye(dim).reshape(dim, 1, 28, 28)
    Fcols  = [blur(basis[j]).flatten() for j in range(dim)]
    F      = torch.stack(Fcols, dim=1).to(device)

    FX = F @ X
    σ  = 0.05
    Y  = FX + torch.randn_like(FX)*σ
    Y  = Y - Y.mean(dim=1, keepdim=True)

    γE = σ**2 * torch.eye(dim, device=device)
    γY = F @ γX @ F.T + γE
    M  = γX @ F.T @ torch.linalg.inv(γY)
    U,S,Vh = torch.linalg.svd(M)

    # losses
    theory_err, learned_err = [], []
    for r in ranks:
        # --- Bayes-optimal map (truncated SVD) ------------------------------
        Mr = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
        relOpt = (torch.norm(Mr @ Y - X, p='fro') /
                  torch.norm(X,   p='fro')).item()
        theory_err.append(relOpt)

        # --- linear autoencoder of rank r -----------------------------------
        ae   = LinearAutoencoder(dim, r).to(device)
        opt  = optim.Adam(ae.parameters(), lr=lr)
        for _ in range(train_epochs):
            opt.zero_grad()
            out = ae(Y.T)
            loss = criterion(out, X.T)
            loss.backward(); opt.step()

        with torch.no_grad():
            A   = ae.decoder.weight @ ae.encoder.weight
            rel = (torch.norm(A @ Y - X, p='fro') /
                   torch.norm(X, p='fro')).item()
        learned_err.append(rel)

    return theory_err, learned_err

# ── run all requested datasets ──────────────────────────────────────────────
datasets  = ['breastmnist', 'chestmnist', 'organamnist', 'organcmnist',
             'organsmnist', 'retinamnist']
ranks     = list(range(25, 776, 25))          # 25, 50, … , 775
results   = defaultdict(dict)                 # results[flag]['theory'|'learned']

for flag in datasets:
    print(f"\n▶ Running rank sweep for {flag} …")
    th, le = run_rank_sweep(flag, ranks)
    results[flag]['theory']  = th
    results[flag]['learned'] = le

# ── combined plot ───────────────────────────────────────────────────────────
plt.figure(figsize=(7.8, 4.7))

short_label = {'theory': 'O',    # Optimal (Bayes-optimal map)
               'learned': 'L'}   # Learned (autoencoder)

for i, flag in enumerate(datasets):
    col = ef_pastel[i % len(ef_pastel)]
    for kind in ('theory', 'learned'):
        plt.plot(ranks,
                 results[flag][kind],
                 line_styles[kind],
                 marker=markers[kind],
                 color=col,
                 lw=lw,
                 ms=ms,
                 label=f"{flag} ({short_label[kind]})")

plt.xlabel('Rank $r$')
plt.ylabel(r'Relative Error $\frac{\|A Y - X\|_{\mathrm{F}}}{\|X\|_{\mathrm{F}}}$')
plt.title('Relative Error vs. Rank')
plt.legend(ncol=3, fontsize='small')   # spread legend across 3 columns
plt.tight_layout()
plt.savefig("end2endpics/MedMNIST/medmnist_rank_sweep.png")
plt.show()

