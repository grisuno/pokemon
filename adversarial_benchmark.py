import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 12
LR = 0.001
LR_TOPOLOGY = 0.02 # Alto inicio para plasticidad rápida
# PGD
TRAIN_EPS_MAX = 0.3
TEST_EPS = 0.3
PGD_STEPS_TRAIN = 7
PGD_STEPS_TEST = 20

# Regularización Balanceada (Basado en la crítica)
LAMBDA_SUPCON = 0.1
LAMBDA_ENTROPY = 0.01
LAMBDA_ORTHO = 1e-4   # Ajustado para no dominar
LAMBDA_SPARSITY = 1e-3 # Suficiente para podar conexiones débiles

print(f"⚙️ Hardware: {DEVICE}")
print(f"🧠 Modelo: TopoBrain v11.1 (Engineered: Fixed Scheduler + Scaled Ortho)")

# ==========================================
# 1. UTILS
# ==========================================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        device = features.device
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask
        
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return -mean_log_prob_pos.mean()

# ==========================================
# 2. CAPAS CCNN (ADEX + SYMBIOTIC + MGF)
# ==========================================
class PredictiveErrorCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Linear(dim * 2, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, input_signal, prediction):
        pos_error = F.relu(input_signal - prediction)
        neg_error = F.relu(prediction - input_signal)
        combined = torch.cat([pos_error, neg_error], dim=-1)
        return self.ln(self.fusion(combined))

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
        gate = self.gate_net(error)
        return x_sensory * gate

class SymbioticBasisRefinement(nn.Module):
    def __init__(self, dim, num_atoms=40):
        super().__init__()
        self.basis_atoms = nn.Parameter(torch.empty(num_atoms, dim))
        nn.init.orthogonal_(self.basis_atoms) # Diversidad inicial máxima
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        Q = self.query_proj(x)
        K = self.key_proj(self.basis_atoms)
        attn = torch.matmul(Q, K.T) * self.scale
        weights = F.softmax(attn, dim=-1)
        x_clean = torch.matmul(weights, self.basis_atoms)
        entropy = -torch.sum(weights * torch.log(weights + 1e-6), dim=-1).mean()
        return x_clean, entropy

class CombinatorialComplexLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, layer_type='midbrain'):
        super().__init__()
        self.num_nodes = num_nodes
        self.layer_type = layer_type
        
        self.node_mapper = nn.Linear(in_dim, hid_dim)
        self.cell_mapper = nn.Linear(in_dim, hid_dim)
        self.symbiotic = SymbioticBasisRefinement(hid_dim, num_atoms=40)
        
        if layer_type == 'midbrain':
            self.pc_cell = PredictiveErrorCell(hid_dim)
        else:
            self.absence_gate = LearnableAbsenceGating(hid_dim)
            
        self.final_mix = nn.Linear(hid_dim * 2, hid_dim)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, x_nodes, adjacency, incidence):
        h0 = self.node_mapper(x_nodes)
        h0_agg = torch.matmul(adjacency, h0)
        
        cell_input = torch.matmul(incidence.T, x_nodes) 
        h2 = self.cell_mapper(cell_input)
        pred_cells, entropy = self.symbiotic(h2)
        pred_nodes = torch.matmul(incidence, pred_cells) 
        
        processed_signal = None
        if self.layer_type == 'midbrain':
            processed_signal = self.pc_cell(h0_agg, pred_nodes)
        elif self.layer_type == 'thalamus':
            processed_signal = self.absence_gate(h0_agg, pred_nodes)
            
        combined = torch.cat([processed_signal, pred_nodes], dim=-1)
        out = self.final_mix(combined)
        return self.norm(out), entropy

# ==========================================
# 3. TOPOBRAIN v11.1 (MASKED PLASTICITY + ORTHO FIX)
# ==========================================
class TopoBrainNet(nn.Module):
    def __init__(self, grid_size=7):
        super().__init__()
        self.grid_size = grid_size
        num_nodes = grid_size * grid_size
        
        # FIX: Conv -> BN -> ReLU (Mejor práctica para adversarial)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # --- TOPOLOGÍA MASKED ---
        adj_init, inc_init = self._init_grid_topology(grid_size)
        self.register_buffer('adj_mask', adj_init > 0)
        self.register_buffer('inc_mask', inc_init > 0)
        
        # Pesos aprendibles
        self.adj_weights = nn.Parameter(torch.zeros_like(adj_init))
        self.inc_weights = nn.Parameter(torch.zeros_like(inc_init))
        
        # Capas
        self.layer1 = CombinatorialComplexLayer(32, 64, num_nodes, 'midbrain')
        self.layer2 = CombinatorialComplexLayer(64, 128, num_nodes, 'thalamus')
        
        self.readout = nn.Linear(128 * num_nodes, 10)
        self.proj_head = nn.Sequential(
            nn.Linear(128 * num_nodes, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def _init_grid_topology(self, N):
        num_nodes = N * N
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            r, c = i // N, i % N
            if r > 0: adj[i, i - N] = 1
            if r < N-1: adj[i, i + N] = 1
            if c > 0: adj[i, i - 1] = 1
            if c < N-1: adj[i, i + 1] = 1
        
        cells = []
        for r in range(N - 1):
            for c in range(N - 1):
                tl = r * N + c
                cells.append([tl, tl + 1, tl + N, tl + N + 1])
        
        num_cells = len(cells)
        inc = torch.zeros(num_nodes, num_cells)
        for ci, ni in enumerate(cells):
            for n in ni: inc[n, ci] = 1.0
        return adj, inc

    def get_topology(self):
        # Sparsity estructural garantizada por máscara
        adj_sparse = torch.zeros_like(self.adj_mask, dtype=torch.float32)
        adj_sparse = torch.where(self.adj_mask, torch.sigmoid(self.adj_weights), adj_sparse)
        curr_adj = F.normalize(adj_sparse, p=1, dim=-1)
        
        inc_sparse = torch.zeros_like(self.inc_mask, dtype=torch.float32)
        inc_sparse = torch.where(self.inc_mask, torch.sigmoid(self.inc_weights), inc_sparse)
        curr_inc = inc_sparse / (inc_sparse.sum(0, keepdim=True) + 1e-6)
        
        return curr_adj, curr_inc

    def calculate_ortho_loss(self):
        # Cálculo quirúrgico de ortogonalidad (Frobenius escalado)
        loss = 0
        mappers = [
            self.layer1.node_mapper.weight, self.layer1.cell_mapper.weight,
            self.layer2.node_mapper.weight, self.layer2.cell_mapper.weight
        ]
        
        for w in mappers:
            rows, cols = w.shape
            if rows >= cols:
                # W^T W = I
                target = torch.eye(cols, device=w.device)
                check = torch.matmul(w.T, w)
            else:
                # W W^T = I
                target = torch.eye(rows, device=w.device)
                check = torch.matmul(w, w.T)
                
            # Norma Frobenius escalada por número de elementos (estabilidad)
            loss += torch.norm(check - target, p='fro') / (rows * cols)
            
        return loss

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        curr_adj, curr_inc = self.get_topology()
        
        x, entropy1 = self.layer1(x, curr_adj, curr_inc)
        x = F.gelu(x)
        x, entropy2 = self.layer2(x, curr_adj, curr_inc)
        x = F.gelu(x)
        
        flat = x.reshape(x.shape[0], -1)
        logits = self.readout(flat)
        proj = F.normalize(self.proj_head(flat), dim=1)
        
        return logits, proj, entropy1 + entropy2, self.adj_weights, self.inc_weights

# ==========================================
# 4. PGD ESTÁNDAR
# ==========================================
def make_adversarial_pgd(model, x, y, eps, steps):
    model.eval()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)
    alpha = eps / 2.5
    
    for _ in range(steps):
        x_adv.requires_grad_()
        out = model(x_adv)[0]
        loss = F.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0, 1)
        
    model.train()
    return x_adv

# ==========================================
# 5. TRAIN LOOP (ENGINEERED SCHEDULER)
# ==========================================
def train_and_eval():
    transform = transforms.ToTensor()
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1000)
    
    model = TopoBrainNet(grid_size=7).to(DEVICE)
    
    # Grupos de parámetros
    topo_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'adj_weights' in name or 'inc_weights' in name:
            topo_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': LR},
        {'params': topo_params, 'lr': LR_TOPOLOGY}
    ], weight_decay=1e-4)
    
    # --- SCHEDULER DE CRISTALIZACIÓN ---
    # Topología: Epoch 0-3 (Learning), Epoch 4+ (Frozen/Fine-tune)
    # Otros: Decay suave
    def lambda_topo(epoch):
        if epoch < 4: return 1.0
        return 0.05 # Caída drástica para cristalizar
    
    def lambda_general(epoch):
        return 0.95 ** epoch

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda_general, lambda_topo])
    
    supcon = SupConLoss(temperature=0.07)
    
    print(f"\n>>> Entrenando TopoBrain v11.1 (Engineered) <<<")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        eps = min(TRAIN_EPS_MAX, 0.05 * epoch)
        
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            x_adv = make_adversarial_pgd(model, x, y, eps, PGD_STEPS_TRAIN)
            
            optimizer.zero_grad()
            logits, proj, entropy, adj_w, inc_w = model(x_adv)
            
            l_ce = F.cross_entropy(logits, y)
            l_sup = supcon(proj, y)
            l_ortho = model.calculate_ortho_loss()
            
            # Penalización L1 sobre los pesos activados (sigmoid)
            l_sparse = torch.norm(torch.sigmoid(adj_w), 1) + torch.norm(torch.sigmoid(inc_w), 1)
            
            # Loss Compuesta Balanceada
            loss = l_ce + \
                   (LAMBDA_SUPCON * l_sup) - \
                   (LAMBDA_ENTROPY * entropy) + \
                   (LAMBDA_ORTHO * l_ortho) + \
                   (LAMBDA_SPARSITY * l_sparse)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)
            
        # Scheduler Step POR ÉPOCA (Fix crítico)
        scheduler.step()
        
        # Logging de Topología
        with torch.no_grad():
            sparsity_adj = (torch.sigmoid(model.adj_weights) > 0.1).float().mean().item()
            sparsity_inc = (torch.sigmoid(model.inc_weights) > 0.1).float().mean().item()
            
        print(f"Ep {epoch:02d} (eps={eps:.2f}) | Loss: {total_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")
        print(f"   ↳ Topology Active: Adj={sparsity_adj:.1%} | Inc={sparsity_inc:.1%} | LR_Topo={optimizer.param_groups[1]['lr']:.5f}")

    # EVALUACIÓN FINAL
    print("\n>>> Evaluación Final <<<")
    model.eval()
    cln_corr, rob_corr = 0, 0
    
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        # Clean
        with torch.no_grad():
            cln_corr += model(x)[0].argmax(1).eq(y).sum().item()
        # Robust
        x_adv = make_adversarial_pgd(model, x, y, TEST_EPS, PGD_STEPS_TEST)
        with torch.no_grad():
            rob_corr += model(x_adv)[0].argmax(1).eq(y).sum().item()
            
    print("="*60)
    print(f"🏆 TopoBrain v11.1 RESULTS")
    print(f"✅ Clean:  {100*cln_corr/len(test_set):.2f}%")
    print(f"🛡️ Robust: {100*rob_corr/len(test_set):.2f}%")
    print("="*60)

if __name__ == "__main__":
    train_and_eval()