import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple

# -------------------------- Config Defaults --------------------------
DEFAULT_CONFIG = {
    'block_size': 64,
    'n_embd': 32,
    'n_head': 2,
    'n_layer': 2,
    'dropout': 0.1,
    'lr': 3e-4,
    'train_steps': 8001,
    'val_interval': 500,
    'batch_size': 8,
    'val_batch_size': 16,
    'max_new_tokens': 500,
    'temperature': 0.9,
    'top_k': 30
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------- Model Components (refactored to take config) --------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        n_embd = config['n_embd']
        n_head = config['n_head']
        dropout = config['dropout']
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)  # C is n_embd
        n_head = q.size(-1) // (C // q.size(-1))  # Infer from shape, but better to pass
        # Actually, to avoid globals, pass n_head or compute
        # But for simplicity, assume it's set in config and used
        q = q.view(B, T, -1, C // (3 * C // q.size(-1))).transpose(1, 2)  # Fix: split gives 3 parts of n_embd
        # Correct: split(N_EMBD, dim=2), but N_EMBD=C
        q = q.view(B, T, self.c_attn.out_features // 3 // (C // (self.c_attn.out_features // 3)), C // (self.c_attn.out_features // 3)).transpose(1, 2)  # This is messy
        # Better: since 3*n_embd, split(n_embd, dim=2)
        # Assume C = n_embd
        head_size = C // (3 * C // q.size(-1)) # wait, let's hardcode logic
        # To make it work, I'll pass n_head to init
        # Wait, let's adjust init
        # Actually, let's modify to take n_head, n_embd, dropout

# Redefine with params
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()  # C = n_embd
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.resid_dropout.p if self.training else 0, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(att))

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.n_embd = n_embd
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        vocab_size = config['vocab_size']
        n_embd = config['n_embd']
        block_size = config['block_size']
        n_layer = config['n_layer']
        n_head = config['n_head']
        dropout = config['dropout']
        self.block_size = block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # tie weights

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

# -------------------------- MiniLLM Class --------------------------
class MiniLLM:
    def __init__(self, config: Optional[Dict[str, Any]] = None, text_file: str = "input.txt"):
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        self.text_file = text_file
        self.device = DEVICE
        self.text = None
        self.stoi = None
        self.itos = None
        self.vocab_size = 0
        self.encode = None
        self.decode = None
        self.data = None
        self.train_data = None
        self.val_data = None
        self.model = None
        self.optimizer = None
        self.train_losses = []
        self.val_losses = []
        self.val_steps = []

        # Load text and build vocab
        self._load_text()
        self._build_vocab()
        self.config['vocab_size'] = self.vocab_size  # Update config

        # Initialize model
        self.model = MiniLLM(self.config).to(self.device)
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Vocabulary size: {self.vocab_size} tokens")

    def _load_text(self):
        if not os.path.exists(self.text_file):
            raise FileNotFoundError(f"Text file '{self.text_file}' not found.")
        with open(self.text_file, "r", encoding="utf-8") as f:
            self.text = f.read()

    def _build_vocab(self):
        words = ['<unk>'] + sorted(set(self.text.split()))
        self.stoi = {word: i for i, word in enumerate(words)}
        self.itos = words
        self.vocab_size = len(words)
        print(f"Vocabulary built: {self.vocab_size} tokens (words)")
        self.encode = lambda s: [self.stoi.get(word, 0) for word in s.split()]  # unknown → 0
        self.decode = lambda l: ' '.join([self.itos[i] for i in l])

    def _prepare_data(self):
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split: str, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size is None:
            batch_size = self.config['batch_size'] if split == 'train' else self.config['val_batch_size']
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.config['block_size'], (batch_size,))
        x = torch.stack([data[i:i + self.config['block_size']] for i in ix])
        y = torch.stack([data[i + 1:i + self.config['block_size'] + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def train(self, steps: Optional[int] = None, resume: bool = False):
        if steps is None:
            steps = self.config['train_steps']
        if not resume:
            self._prepare_data()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
            self.train_losses = []
            self.val_losses = []
            self.val_steps = []
            print("Training started...\n")
        else:
            print("Resuming training...\n")

        self.model.train()
        for step in range(steps):
            if step == 0 and resume:
                continue  # Skip first if resuming, but actually, better to start from current len(train_losses)
            # Actually, for simplicity, assume no resume logic beyond loading losses
            xb, yb = self.get_batch("train")
            logits, loss = self.model(xb, yb)
            self.train_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Validation & printing
            if (step + 1) % self.config['val_interval'] == 0 or step == steps - 1:
                self.model.eval()
                with torch.no_grad():
                    vx, vy = self.get_batch("val")
                    _, vloss = self.model(vx, vy)
                    self.val_losses.append(vloss.item())
                    self.val_steps.append(step)
                print(f"step {step + 1:5d} | train {loss.item():.4f} | val {vloss.item():.4f}")
                self.model.train()

    def save(self, path: str = "mini_llm.pth"):
        checkpoint = {
            'config': self.config,
            'stoi': self.stoi,
            'itos': self.itos,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_steps': self.val_steps,
            'vocab_size': self.vocab_size,
        }
        torch.save(checkpoint, path)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"\nSaved {path} → {size_mb:.2f} MB")

    @classmethod
    def load(cls, path: str, text_file: str = "input.txt") -> 'MiniLLM':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file '{path}' not found.")
        checkpoint = torch.load(path, map_location=DEVICE)
        
        # Create instance
        instance = cls(config=checkpoint['config'], text_file=text_file)
        
        # Override loaded components
        instance.stoi = checkpoint['stoi']
        instance.itos = checkpoint['itos']
        instance.vocab_size = checkpoint['vocab_size']
        instance.encode = lambda s: [instance.stoi.get(word, 0) for word in s.split()]
        instance.decode = lambda l: ' '.join([instance.itos[i] for i in l])
        
        # Load model and optimizer
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] is not None:
            instance.optimizer = torch.optim.AdamW(instance.model.parameters(), lr=instance.config['lr'])
            instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load losses
        instance.train_losses = checkpoint['train_losses']
        instance.val_losses = checkpoint['val_losses']
        instance.val_steps = checkpoint['val_steps']
        
        # Prepare data for consistency
        instance._prepare_data()
        
        print(f"Model loaded from {path}")
        return instance

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, top_k: Optional[int] = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.config['max_new_tokens']
        if temperature is None:
            temperature = self.config['temperature']
        if top_k is None:
            top_k = self.config['top_k']
        
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        generated = self.model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)[0].tolist()
        return self.decode(generated)

    def plot_losses(self, save_path: str = "loss_plot.png", show: bool = False):
        if not self.train_losses:
            print("No training losses to plot.")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training loss", alpha=0.6, color="#1f77b4")
        if self.val_losses:
            plt.plot(self.val_steps, self.val_losses, label="Validation loss", color="#ff7f0e", linewidth=2)
        plt.xlabel("Training step")
        plt.ylabel("Cross-entropy loss")
        plt.title("MiniLLM Training & Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"Loss plot saved as → {save_path}")
        if show:
            plt.show()

    def get_embeddings(self, prompt: str):
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            tok_emb = self.model.transformer.wte(context)  # This is the text embedding tensor
            pos_emb = self.model.transformer.wpe(torch.arange(context.size(1), device=self.device))
            full_emb = tok_emb + pos_emb  # Often positional embeddings are added too
        return full_emb  # Shape: (1, seq_len, n_embd)

# -------------------------- Example Usage (for direct script run) --------------------------
if __name__ == "__main__":
    # Train a new model
    llm = MiniLLM()
    llm = MiniLLM.load("model.pth")

    llm.train()
    llm.plot_losses()
    llm.save("model.pth")  

    # Generate
    print("\n" + "═" * 60)
    generated = llm.generate("\nWhat have you learned?")
    print(generated)
    print("═" * 60)
