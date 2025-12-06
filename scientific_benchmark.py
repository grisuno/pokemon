import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import time
import random
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIGURACIÓN CIENTÍFICA
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuración Base
GLOBAL_SETTINGS = {
    'batch_size': 128,
    'epochs': 100,
    'lr_sgd': 0.1,
    'lr_topo': 0.01,
    'grid_size': 8, # CIFAR 32x32/4
}

# Robustez (L_inf 8/255)
ROBUST_CONFIG = {
    'train_eps': 8/255,
    'test_eps': 8/255,
    'pgd_steps_train': 10,
    'pgd_steps_test': 20,
    'autoattack_n': 1000 
}

# Stats CIFAR-10
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(DEVICE)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(DEVICE)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"⚙️ Hardware: {DEVICE}")
print(f"🧪 Modo: Multi-Seed Ablation Suite (RC1)")

# ==========================================
# 1. COMPONENTES DEL MODELO
# ==========================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(batch_size).view(-1, 1).to(features.device), 0
        )
        mask = mask * logits_mask
        
        anchor_dot = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        logits = anchor_dot - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        return -(mask * log_prob).sum(1) / mask_sum # Loss escalar

class PredictiveErrorCell(nn.Module):
    def __init__(self, dim, use_spectral=True):
        super().__init__()
        layer = nn.Linear(dim * 2, dim)
        self.fusion = spectral_norm(layer) if use_spectral else layer
        self.ln = nn.LayerNorm(dim)

    def forward(self, input_signal, prediction):
        pos = F.relu(input_signal - prediction)
        neg = F.relu(prediction - input_signal)
        return self.ln(self.fusion(torch.cat([pos, neg], dim=-1)))

class LearnableAbsenceGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x_sensory, x_prediction):
        error = torch.abs(x_sensory - x_prediction)
        return x_sensory * self.gate_net(error)

class SymbioticBasisRefinement(nn.Module):
    def __init__(self, dim, num_atoms=64):
        super().__init__()
        self.basis_atoms = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis_atoms)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(self.basis_atoms)
        attn = torch.matmul(Q, K.T) * self.scale
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis_atoms)
        entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=-1).mean()
        return x_clean, entropy

# ==========================================
# 2. ARQUITECTURA TOPOBRAIN
# ==========================================
class CombinatorialComplexLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, config, layer_type='midbrain'):
        super().__init__()
        self.num_nodes = num_nodes
        self.cfg = config
        self.layer_type = layer_type
        
        use_spec = config.get('use_spectral', True)
        self.node_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
        self.cell_mapper = spectral_norm(nn.Linear(in_dim, hid_dim)) if use_spec else nn.Linear(in_dim, hid_dim)
        
        self.symbiotic = SymbioticBasisRefinement(hid_dim)
        
        if layer_type == 'midbrain':
            self.pc_cell = PredictiveErrorCell(hid_dim, use_spec)
        else:
            self.absence_gate = LearnableAbsenceGating(hid_dim)
            
        final = nn.Linear(hid_dim * 2, hid_dim)
        self.final_mix = spectral_norm(final) if use_spec else final
        self.norm = nn.LayerNorm(hid_dim)
        
        # Baseline Mixer (para cuando no hay topología)
        self.baseline_mixer = nn.Linear(in_dim, hid_dim)

    def forward(self, x_nodes, adjacency, incidence):
        # 1. Procesamiento Nodal
        if self.cfg['use_plasticity'] or self.cfg.get('use_static_topo', False):
            h0 = self.node_mapper(x_nodes)
            h0_agg = torch.matmul(adjacency, h0)
        else:
            # Baseline justo: MLP Mixer (sin estructura espacial)
            h0_agg = self.baseline_mixer(x_nodes)

        # 2. Top-Down
        cell_input = torch.matmul(incidence.T, x_nodes)
        h2 = self.cell_mapper(cell_input)
        pred_cells, entropy = self.symbiotic(h2)
        pred_nodes = torch.matmul(incidence, pred_cells)
        
        # 3. MGF & Hierarchy
        if self.cfg['use_mgf']:
            if self.layer_type == 'midbrain':
                processed = self.pc_cell(h0_agg, pred_nodes)
            else:
                processed = self.absence_gate(h0_agg, pred_nodes)
        else:
            processed = h0_agg 
            
        out = self.final_mix(torch.cat([processed, pred_nodes], dim=-1))
        return self.norm(out), entropy

class TopoBrainNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        grid = config['grid_size']
        num_nodes = grid * grid
        
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Topología
        adj, inc = self._init_grid(grid)
        self.register_buffer('adj_mask', adj > 0)
        self.register_buffer('inc_mask', inc > 0)
        
        if config['use_plasticity']:
            self.adj_weights = nn.Parameter(torch.zeros_like(adj))
            self.inc_weights = nn.Parameter(torch.zeros_like(inc))
        else:
            # Estática: Pesos fuertes fijos (sigmoid(5) ~= 1)
            self.register_buffer('adj_weights', torch.ones_like(adj) * 5.0)
            self.register_buffer('inc_weights', torch.ones_like(inc) * 5.0)

        self.layer1 = CombinatorialComplexLayer(64, 128, num_nodes, config, 'midbrain')
        self.layer2 = CombinatorialComplexLayer(128, 256, num_nodes, config, 'thalamus')
        
        self.readout = nn.Linear(256 * num_nodes, 10)
        self.proj_head = nn.Sequential(
            nn.Linear(256 * num_nodes, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def _init_grid(self, N):
        num_nodes = N * N
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            r, c = i // N, i % N
            if r > 0: adj[i, i-N] = 1
            if r < N-1: adj[i, i+N] = 1
            if c > 0: adj[i, i-1] = 1
            if c < N-1: adj[i, i+1] = 1
        
        cells = []
        for r in range(N - 1):
            for c in range(N - 1):
                tl = r * N + c
                cells.append([tl, tl+1, tl+N, tl+N+1])
        
        inc = torch.zeros(num_nodes, len(cells))
        for ci, ni in enumerate(cells):
            for n in ni: inc[n, ci] = 1.0
        return adj, inc

    def get_topology(self):
        adj_sparse = torch.zeros_like(self.adj_mask, dtype=torch.float32)
        adj_w = torch.sigmoid(self.adj_weights)
        adj_sparse = torch.where(self.adj_mask, adj_w, adj_sparse)
        curr_adj = F.normalize(adj_sparse, p=1, dim=-1)
        
        inc_sparse = torch.zeros_like(self.inc_mask, dtype=torch.float32)
        inc_w = torch.sigmoid(self.inc_weights)
        inc_sparse = torch.where(self.inc_mask, inc_w, inc_sparse)
        curr_inc = inc_sparse / (inc_sparse.sum(0, keepdim=True) + 1e-6)
        return curr_adj, curr_inc

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        curr_adj, curr_inc = self.get_topology()
        
        x, ent1 = self.layer1(x, curr_adj, curr_inc)
        x = F.gelu(x)
        x, ent2 = self.layer2(x, curr_adj, curr_inc)
        x = F.gelu(x)
        
        flat = x.reshape(x.shape[0], -1)
        logits = self.readout(flat)
        
        proj = None
        if self.cfg.get('use_supcon', False):
            proj = F.normalize(self.proj_head(flat), dim=1)
            
        return logits, proj, ent1+ent2

# ==========================================
# 3. PGD & EVAL
# ==========================================
def clamp_pgd(x_adv_norm, x_orig_norm, eps):
    x_adv = x_adv_norm * CIFAR_STD + CIFAR_MEAN
    x_orig = x_orig_norm * CIFAR_STD + CIFAR_MEAN
    delta = torch.clamp(x_adv - x_orig, -eps, eps)
    x_adv_clamped = torch.clamp(x_orig + delta, 0.0, 1.0)
    return (x_adv_clamped - CIFAR_MEAN) / CIFAR_STD

def make_adversarial_pgd(model, x, y, eps, steps):
    model.eval()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    x_adv = clamp_pgd(x + delta, x, eps)
    alpha = eps / 4.0
    
    for _ in range(steps):
        x_adv.requires_grad_()
        out = model(x_adv)[0]
        loss = F.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = clamp_pgd(x_adv, x, eps)
    model.train()
    return x_adv

def eval_autoattack(model, test_loader, n_samples=1000):
    try:
        from autoattack import AutoAttack
    except ImportError:
        return -1.0

    model.eval()
    all_x, all_y = [], []
    count = 0
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)
        count += x.size(0)
        if count >= n_samples: break
        
    x_test = torch.cat(all_x)[:n_samples].to(DEVICE)
    y_test = torch.cat(all_y)[:n_samples].to(DEVICE)
    
    class Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)[0]

    adversary = AutoAttack(Wrapper(model), norm='Linf', eps=ROBUST_CONFIG['test_eps'], version='standard', verbose=False)
    # Batch size pequeño para no explotar VRAM
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=100)
    
    with torch.no_grad():
        acc = Wrapper(model)(x_adv).argmax(1).eq(y_test).float().mean().item()
    return acc * 100

# ==========================================
# 4. TOPOLOGY LOGGER
# ==========================================
def save_topology_snapshot(model, epoch, run_name):
    if not model.cfg['use_plasticity']: return
    
    # Guardar matriz en disco
    adj_w = torch.sigmoid(model.adj_weights).detach().cpu().numpy()
    
    os.makedirs(f"results/{run_name}", exist_ok=True)
    
    plt.figure(figsize=(8,6))
    plt.imshow(adj_w, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Topology Epoch {epoch}")
    plt.savefig(f"results/{run_name}/topo_ep{epoch:03d}.png")
    plt.close()

# ==========================================
# 5. EXPERIMENT RUNNER
# ==========================================
def run_training(config_override, run_name):
    cfg = GLOBAL_SETTINGS.copy()
    cfg.update(config_override)
    seed_everything(cfg['seed'])
    
    # Datasets
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=1000, num_workers=2)
    
    model = TopoBrainNet(cfg).to(DEVICE)
    
    # Optimizadores
    topo_params = [p for n,p in model.named_parameters() if 'weights' in n and p.requires_grad]
    main_params = [p for n,p in model.named_parameters() if 'weights' not in n and p.requires_grad]
    
    opt_main = optim.SGD(main_params, lr=cfg['lr_sgd'], momentum=0.9, weight_decay=5e-4)
    
    sched_main = optim.lr_scheduler.MultiStepLR(
        opt_main, milestones=[int(cfg['epochs']*0.5), int(cfg['epochs']*0.75)], gamma=0.1
    )
    
    opt_topo = None
    sched_topo = None
    if cfg['use_plasticity']:
        opt_topo = optim.AdamW(topo_params, lr=cfg['lr_topo'], weight_decay=1e-3)
        # Scheduler Suavizado (Neuro-Evolutivo)
        def lambda_topo(epoch):
            warmup = 5
            plateau = 20
            if epoch < warmup: return epoch / warmup
            if epoch < plateau: return 1.0
            return 0.1 + 0.9 * math.exp(-0.1 * (epoch - plateau))
        sched_topo = optim.lr_scheduler.LambdaLR(opt_topo, lr_lambda=lambda_topo)
        
    supcon = SupConLoss(temperature=0.07)
    
    print(f"\n--- Running: {run_name} ---")
    
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        meters = {'loss': 0, 'acc': 0}
        t0 = time.time()
        eps = ROBUST_CONFIG['train_eps']
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = make_adversarial_pgd(model, x, y, eps, ROBUST_CONFIG['pgd_steps_train'])
            
            if cfg['use_plasticity']: opt_topo.zero_grad()
            opt_main.zero_grad()
            
            logits, proj, entropy = model(x_adv)
            
            loss = F.cross_entropy(logits, y)
            
            # Configurable Lambdas
            l_sup = cfg.get('lambda_supcon', 0.1)
            l_ent = cfg.get('lambda_entropy', 0.01)
            l_spar = cfg.get('lambda_sparsity', 5e-5)
            
            if cfg.get('use_supcon', False):
                loss += l_sup * supcon(proj, y).mean() # Fix: .mean()
                
            loss -= l_ent * entropy 
            
            if cfg['use_plasticity']:
                # Sparsity solo en conexiones válidas
                sparse_loss = torch.norm(torch.sigmoid(model.adj_weights), 1)
                loss += l_spar * sparse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_main.step()
            if cfg['use_plasticity']: opt_topo.step()
            
            meters['loss'] += loss.item()
            meters['acc'] += logits.argmax(1).eq(y).sum().item()
            
        sched_main.step()
        if cfg['use_plasticity']: sched_topo.step()
        
        if epoch % 20 == 0 or epoch == cfg['epochs']:
             save_topology_snapshot(model, epoch, run_name)
             
        if epoch % 10 == 0:
             print(f"Ep {epoch:03d} | Loss: {meters['loss']/len(train_loader):.4f} | Acc: {100*meters['acc']/len(train_ds):.2f}% | T: {time.time()-t0:.1f}s")

    # EVALUACIÓN
    model.eval()
    # Clean
    correct = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad(): correct += model(x)[0].argmax(1).eq(y).sum().item()
    clean_acc = 100 * correct / len(test_ds)
    
    # PGD-20
    correct = 0
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_adv = make_adversarial_pgd(model, x, y, ROBUST_CONFIG['test_eps'], ROBUST_CONFIG['pgd_steps_test'])
        with torch.no_grad(): correct += model(x_adv)[0].argmax(1).eq(y).sum().item()
    pgd_acc = 100 * correct / len(test_ds)
    
    # AutoAttack
    aa_acc = eval_autoattack(model, test_loader, n_samples=ROBUST_CONFIG['autoattack_n'])
    
    return clean_acc, pgd_acc, aa_acc

# ==========================================
# 6. ABLATION SUITE (SCIENTIFIC)
# ==========================================
def run_ablation_suite_scientific():
    # Experimentos clave para el paper
    experiments = [
        {
            'name': 'Baseline (AT)', 
            'config': {'use_supcon': False, 'use_mgf': False, 'use_plasticity': False, 'use_static_topo': False} 
        },
        {
            'name': '+SupCon', 
            'config': {'use_supcon': True, 'use_mgf': False, 'use_plasticity': False, 'use_static_topo': False}
        },
        {
            'name': '+MGF', 
            'config': {'use_supcon': True, 'use_mgf': True, 'use_plasticity': False, 'use_static_topo': True}
        },
        {
            'name': 'TopoBrain v15 (Full)', 
            'config': {'use_supcon': True, 'use_mgf': True, 'use_plasticity': True, 'use_spectral': True}
        }
    ]
    
    seeds = [42, 123, 456] # 3 Seeds para estadística
    results_table = []
    
    print(f"{'='*60}\nSTARTING MULTI-SEED ABLATION SUITE\n{'='*60}")
    
    for exp in experiments:
        print(f"\n>>> Experiment: {exp['name']} <<<")
        clean_accs, pgd_accs, aa_accs = [], [], []
        
        for seed in seeds:
            print(f"  Running Seed {seed}...")
            exp['config']['seed'] = seed
            c, p, a = run_training(exp['config'], f"{exp['name']}_s{seed}".replace(" ", "_"))
            clean_accs.append(c)
            pgd_accs.append(p)
            if a != -1: aa_accs.append(a)
            
        # Estadísticas
        r = {
            'name': exp['name'],
            'clean': (np.mean(clean_accs), np.std(clean_accs)),
            'pgd': (np.mean(pgd_accs), np.std(pgd_accs)),
            'aa': (np.mean(aa_accs), np.std(aa_accs)) if aa_accs else (-1, 0)
        }
        results_table.append(r)
        
    # LaTeX Output
    print(f"\n{'='*60}\nFINAL RESULTS TABLE (LaTeX)\n{'='*60}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Configuration & Clean Acc & PGD-20 & AutoAttack \\\\")
    print("\\midrule")
    for r in results_table:
        c_str = f"{r['clean'][0]:.2f} $\\pm$ {r['clean'][1]:.2f}"
        p_str = f"{r['pgd'][0]:.2f} $\\pm$ {r['pgd'][1]:.2f}"
        a_str = f"{r['aa'][0]:.2f} $\\pm$ {r['aa'][1]:.2f}" if r['aa'][0] != -1 else "N/A"
        print(f"{r['name']} & {c_str} & {p_str} & {a_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__ == "__main__":
    # Para prueba rápida, usar epochs=10 y 1 seed
    GLOBAL_SETTINGS['epochs'] = 10
    run_ablation_suite_scientific()