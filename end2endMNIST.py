import math
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import GaussianBlur
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data setup ---
tensorTransform = transforms.ToTensor()
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=tensorTransform)

# Parameters
numSamples = int(0.8 * len(dataset))
dim = 784
r = 200  # bottleneck / testing rank

# X (clean)
X_raw = torch.stack([dataset[i][0] for i in range(numSamples)]).to(device)
X = X_raw.view(numSamples, -1).T
muX = X.mean(dim=1, keepdim=True)
X = X - muX
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
M = gammaX @ F.T @ torch.linalg.inv(gammaY) 
U, S, Vh = torch.linalg.svd(M)

# Compute theoretical total error (for reference)
U_r = U[:, :r]
S_r = torch.diag(S[:r])
Vh_r = Vh[:r, :]
M_r = U_r @ S_r @ Vh_r

# Calculate the theoretical pixel error
error_theory_train = (torch.norm(M_r @ Y - X, p="fro")**2).item()
rel_error_theory_train = (torch.norm(M_r @ Y - X, p="fro") / torch.norm(X, p="fro")).item()
print(f"Theoretical training error: {error_theory_train:.4f}")
print(f"Theoretical relative training error: {rel_error_theory_train:.4f}")

# --- Create the test set (COMMENTED OUT) ---
# numTest = 500
# X_raw_test = torch.stack([dataset[i + numSamples][0] for i in range(numTest)]).to(device)
# X_test = X_raw_test.view(numTest, -1).T
# X_test = X_test - muX

# FX_test = F @ X_test
# E_test  = torch.randn_like(FX_test) * noiseSigma
# Y_test  = FX_test + E_test
# Y_test = Y_test - muY

# error_theory_test = (torch.norm(M_r @ Y_test - X_test, p="fro")**2).item()
# rel_error_theory_test = (torch.norm(M_r @ Y_test - X_test, p="fro") / torch.norm(X_test, p="fro")).item()
# print(f"Theoretical test error: {error_theory_test:.4f}")
# print(f"Theoretical relative test error: {rel_error_theory_test:.4f}")

# --- Autoencoder setup ---
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

# Training loop using full Frobenius loss
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
        
        err_train = torch.norm(A_learned @ Y - X, p="fro")**2
        # err_test  = torch.norm(A_learned @ Y_test - X_test, p="fro")**2  # COMMENTED OUT
        
        rel_err_train = torch.norm(A_learned @ Y - X, p="fro") / torch.norm(X, p='fro')
        # rel_err_test = torch.norm(A_learned @ Y_test - X_test, p="fro") / torch.norm(X_test, p='fro')  # COMMENTED OUT
        
        train_errors.append(rel_err_train.item())
        # test_errors.append(rel_err_test.item())  # COMMENTED OUT

# Plot convergence vs. theoretical optimum
plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs+1), train_errors, label='Training', color='#a7c080')
# plt.plot(range(1, num_epochs+1), test_errors,  label='Test  error')  # COMMENTED OUT
plt.axhline(rel_error_theory_train, ls='--', color='k', label='Optimal')
# plt.axhline(rel_error_theory_test, ls='--', color='r', label='Theoretical optimum (test)')  # COMMENTED OUT
plt.xlabel('Epoch')
plt.ylabel(r'Relative Error $\frac{\|A_{\text{ae}} Y - X\|_{\text{F}}}{\|X\|_{\text{F}}}$')
plt.title(r'Optimal $M_r$ vs. $A_{\text{learned}}$')
plt.legend()
plt.tight_layout()
#plt.savefig(f"end2endpics/MNIST/mnist_loss{numSamples}")
#plt.show()

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
    "#e67e80",  # red (end)
    "#dc5456",  # red (end)
]


# build a smooth colormap by interpolating between those
everforest_cmap = LinearSegmentedColormap.from_list(
    'everforest_blend',
    everforest_colors,
    N=256
)

for _ in range(10):
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

    # colour-bar
    cax = fig.add_subplot(gs[:, 3])
    plt.colorbar(im, cax=cax)
    cax.yaxis.tick_right()

    plt.tight_layout()
    #plt.savefig(f"end2endpics/MNIST/sample{idx}_{numSamples}.png")
    

# ---------------------------------------------------------------------------
# --- Rank sweep experiment: theoretical vs. learned mapping ---------------
# ---------------------------------------------------------------------------
ranks               = list(range(25, 776, 25))          # 0, 25, … , 775
theoryRelErrors     = []                               # optimal ‖·‖ₓ errors
learnedRelErrors    = []                               # AE-learned errors
rankTrainEpochs     = 350                             
rankLR              = 1e-3

for rank in ranks:
    # ------------------ theoretical optimum (truncate SVD) -----------------
    #if rank == 0:
    #    theoryRelErrors.append(1.0)        # zero map ⇒ ‖X‖ / ‖X‖ = 1
    #    learnedRelErrors.append(1.0)       # skip AE training for rank 0
    #    print(f"Rank {rank:3d}: theory=1.0000, learned=1.0000 (skipped)")
    #    continue

    U_r   = U[:, :rank]
    S_r   = torch.diag(S[:rank])
    Vh_r  = Vh[:rank, :]
    M_r   = U_r @ S_r @ Vh_r
    relErrTheory = (torch.norm(M_r @ Y - X, p='fro') /
                    torch.norm(X,  p='fro')).item()
    theoryRelErrors.append(relErrTheory)

    # ------------------ train linear autoencoder of this rank --------------
    model_r = LinearAutoencoder(input_dim=dim, bottleneck_dim=rank).to(device)
    opt_r   = optim.Adam(model_r.parameters(), lr=rankLR)

    for _ in range(rankTrainEpochs):
        model_r.train()
        opt_r.zero_grad()
        out_r = model_r(Y.T)
        loss_r = criterion(out_r, X.T)
        loss_r.backward()
        opt_r.step()

    with torch.no_grad():
        A_r = model_r.decoder.weight @ model_r.encoder.weight
        relErrLearned = (torch.norm(A_r @ Y - X, p='fro') /
                         torch.norm(X, p='fro')).item()
    learnedRelErrors.append(relErrLearned)

    print(f"Rank {rank:3d}: theory={relErrTheory:.4f}, learned={relErrLearned:.4f}")

# -------------------------- plot: error vs. rank ---------------------------
plt.figure(figsize=(6.8, 4.2))

plt.plot(ranks, theoryRelErrors, '-o',
         label='Theoretical optimal', color='#a7c080', lw=2, ms=4)   # EF green
plt.plot(ranks, learnedRelErrors, '-s',
         label='Autoencoder learned', color='#e67e80', lw=2, ms=4)   # EF red

plt.xlabel('Rank $r$')
plt.ylabel(r'Relative Error $\frac{\|A Y - X\|_{\mathrm{F}}}{\|X\|_{\mathrm{F}}}$')
plt.title(r'Relative Error vs. Rank')
plt.legend()
plt.tight_layout()
plt.savefig(f"end2endpics/MNIST/mnist_rank_sweep_{numSamples}.png")
# plt.show()
