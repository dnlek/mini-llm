import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import math
import argparse
import random
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from dsl import DSLDecoder, DSLEncoder
from tools import execute_tool
from history import HybridMemorySystem

# Try to import rating module (optional)
try:
    from rate_inputs import rate_all_files
    RATING_AVAILABLE = True
except ImportError:
    RATING_AVAILABLE = False

# Try to import quantization (available in PyTorch 1.3+)
try:
    from torch.quantization import quantize_dynamic
    QUANTIZATION_AVAILABLE = True
except ImportError:
    try:
        from torch.ao.quantization import quantize_dynamic
        QUANTIZATION_AVAILABLE = True
    except ImportError:
        QUANTIZATION_AVAILABLE = False

torch.set_float32_matmul_precision('medium')  # PyTorch 2.0+


# hyperparameters
batch_size = 256
block_size = 256
max_iters = 10000
eval_interval = 100
learning_rate = 2e-5  # Reduced from 1e-4 to combat overfitting (smaller steps = less memorization)
min_learning_rate = 5e-7  # Minimum learning rate for decay
warmup_iters = 500  # Warmup iterations before decay starts
label_smoothing = 0.05
decay_style = 'cosine'  # 'cosine' or 'linear'
eval_iters = 50
n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.35  # Increased from 0.2 to reduce overfitting
early_stop_patience = 15  # Stop training if val loss doesn't improve for N evaluations
early_stop_min_delta = 0.0005  # Minimum change to qualify as an improvement
weight_decay = 0.03

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = 'mps'
    torch.backends.mps.allow_tf32 = True

print(f"Using device: {device}")

torch.manual_seed(1337)

def normalize_checkpoint_state_dict(state_dict):
    """
    Normalize checkpoint state dict to work with uncompiled models.
    Strips '_orig_mod.' prefix if present (from compiled model checkpoints).
    Also filters out unused keys like inv_freq (from old rotary embedding code).
    """
    normalized = {}
    for k, v in state_dict.items():
        # Strip '_orig_mod.' prefix if present
        if k.startswith('_orig_mod.'):
            key = k[len('_orig_mod.'):]
        else:
            key = k
        
        # Filter out unused keys (like inv_freq from old rotary embedding code)
        if 'inv_freq' in key:
            continue
        
        normalized[key] = v
    
    return normalized

def get_input_files(inputs_dir='inputs'):
    if not os.path.exists(inputs_dir):
        print(f"Warning: {inputs_dir} directory not found, creating it...")
        os.makedirs(inputs_dir, exist_ok=True)
        return []
    
    txt_files = []
    for filename in os.listdir(inputs_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(inputs_dir, filename)
            txt_files.append(filepath)
    
    txt_files.sort()
    return txt_files

def train_tokenizer(input_files=None, vocab_size=50000):
    if input_files is None:
        input_files = get_input_files()
    
    if not input_files:
        raise ValueError("No .txt files found in inputs directory")
    
    print(f"Training Byte-Level BPE tokenizer on {len(input_files)} file(s)...")
    for f in input_files:
        print(f"  - {f}")
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        show_progress=True
    )
    
    tokenizer.train(input_files, trainer)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    
    tokenizer_path = 'tokenizer_inputs.json'
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer

def load_or_train_tokenizer(inputs_dir='inputs', vocab_size=50000):
    tokenizer_path = 'tokenizer_inputs.json'
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        input_files = get_input_files(inputs_dir)
        tokenizer = train_tokenizer(input_files, vocab_size)
    return tokenizer

tokenizer = load_or_train_tokenizer()

def encode(text):
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return encoding.ids

def decode(token_ids):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, list):
        token_ids = list(token_ids)
    
    return tokenizer.decode(token_ids, skip_special_tokens=True)

vocab_size = tokenizer.get_vocab_size()
# Pad token ID: used for padding sequences to same length (not zero, as zero might be a valid token)
try:
    pad_token_id = tokenizer.token_to_id("<pad>")
    if pad_token_id is None:
        pad_token_id = 1  # Fallback to 1 if not found
except:
    pad_token_id = 1  # Fallback
print(f"Vocabulary size: {vocab_size}, Pad token ID: {pad_token_id}")

def load_all_input_files(inputs_dir='inputs'):
    """Load and parse question/answer pairs from input files, grouped by source file"""
    # Rate input files before loading (if rating module is available)
    if RATING_AVAILABLE:
        print("\n" + "="*70)
        print("RATING INPUT FILES BEFORE TRAINING")
        print("="*70)
        try:
            rating_summary = rate_all_files(inputs_dir, verbose=True)
            if 'error' not in rating_summary:
                avg_score = rating_summary.get('avg_score', 0)
                min_score = rating_summary.get('min_score', 100)
                if avg_score < 60:
                    print(f"\n⚠️  WARNING: Average quality score is {avg_score:.1f}/100 (below 60)")
                    print("   Consider reviewing and improving input files before training.")
                elif min_score < 40:
                    print(f"\n⚠️  WARNING: Some files have very low quality scores (min: {min_score:.1f}/100)")
                    print("   Consider reviewing and improving low-scoring files before training.")
                else:
                    print(f"\n✓ Input files quality check passed (avg: {avg_score:.1f}/100)")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not rate input files: {e}")
            print("   Continuing with data loading...")
        print("="*70 + "\n")
    
    input_files = get_input_files(inputs_dir)
    
    if not input_files:
        print(f"Warning: No .txt files found in {inputs_dir}, falling back to input.txt")
        if os.path.exists('input.txt'):
            input_files = ['input.txt']
        else:
            raise ValueError(f"No training files found in {inputs_dir} or input.txt")
    
    print(f"Loading {len(input_files)} file(s) from {inputs_dir}:")
    all_pairs = []
    pairs_by_file = {}
    
    for filepath in input_files:
        print(f"  Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        file_pairs = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line and not line.startswith(('S:', 'T:', 'C:', 'E:')) and line != 'CL':
                question = line
                if i + 1 < len(lines):
                    answer = lines[i + 1].strip()
                    if answer.startswith(('S:', 'T:', 'C:', 'E:')) or answer == 'CL':
                        pair_text = f"{question}\n{answer}\n\n"
                        all_pairs.append(pair_text)
                        file_pairs.append(pair_text)
                        i += 2
                        continue
            i += 1
        
        pairs_by_file[filepath] = file_pairs
        print(f"    Loaded {len(file_pairs)} question/answer pairs")
    
    random.shuffle(all_pairs)
    combined_text = ''.join(all_pairs)
    print(f"Total: {len(all_pairs)} pairs, {len(combined_text)} characters")
    return combined_text, all_pairs, pairs_by_file

train_pairs = None
val_pairs = None
train_pairs_by_file = None
val_pairs_by_file = None

def get_batch(split, debug=False):
    """Sample batches with stratified sampling from all files for balanced representation"""
    if train_pairs is None or val_pairs is None:
        raise RuntimeError("Training data not loaded. Call train_model() to load data.")
    
    pairs = train_pairs if split == 'train' else val_pairs
    pairs_by_file = train_pairs_by_file if split == 'train' else val_pairs_by_file
    
    selected_pairs = []
    if pairs_by_file and len(pairs_by_file) > 1:
        samples_per_file = max(1, batch_size // len(pairs_by_file))
        remaining = batch_size
        selected_ids = set()
        
        file_list = list(pairs_by_file.items())
        random.shuffle(file_list)
        
        for filepath, file_pairs in file_list:
            if remaining <= 0:
                break
            n_samples = min(samples_per_file, len(file_pairs), remaining)
            if n_samples > 0:
                sampled = random.sample(file_pairs, n_samples)
                for pair in sampled:
                    pair_id = id(pair)
                    if pair_id not in selected_ids:
                        selected_pairs.append(pair)
                        selected_ids.add(pair_id)
                        remaining -= 1
                        if remaining <= 0:
                            break
        
        if remaining > 0:
            all_file_pairs = [p for file_pairs in pairs_by_file.values() for p in file_pairs]
            available = [p for p in all_file_pairs if id(p) not in selected_ids]
            if available:
                additional = random.sample(available, min(remaining, len(available)))
                selected_pairs.extend(additional)
        
        random.shuffle(selected_pairs)
    else:
        selected_pairs = random.sample(pairs, min(batch_size, len(pairs)))
    
    x_batch = []
    y_batch = []
    pair_info = []
    
    for pair_tokens in selected_pairs:
        non_pad_mask = (pair_tokens != pad_token_id)
        non_pad_indices = torch.nonzero(non_pad_mask, as_tuple=False).squeeze(-1)
        
        if len(non_pad_indices) > 1:
            last_non_pad_idx = non_pad_indices[-1].item()
            x_seq = pair_tokens[:last_non_pad_idx]
            y_seq = pair_tokens[1:last_non_pad_idx+1]
        elif len(non_pad_indices) == 1:
            idx = non_pad_indices[0].item()
            x_seq = pair_tokens[:idx+1]
            y_seq = pair_tokens[1:idx+2] if idx+1 < len(pair_tokens) else pair_tokens[1:]
        else:
            continue
        
        pad_length = block_size - len(x_seq)
        if pad_length > 0:
            pad_tensor = torch.full((pad_length,), pad_token_id, dtype=torch.long)
            x_seq = torch.cat([x_seq, pad_tensor])
            y_seq_pad = torch.full((pad_length,), pad_token_id, dtype=torch.long)
            y_seq = torch.cat([y_seq, y_seq_pad])
        
        if len(x_seq) != block_size or len(y_seq) != block_size:
            continue
        
        if (y_seq != pad_token_id).sum().item() == 0:
            continue
        
        x_batch.append(x_seq)
        y_batch.append(y_seq)
        
        if debug:
            decoded = decode(pair_tokens.tolist())
            parts = decoded.split('\n', 1)
            question = parts[0] if parts else ""
            answer = parts[1].strip() if len(parts) > 1 else ""
            pair_info.append({
                'question': question,
                'answer': answer[:80] + '...' if len(answer) > 80 else answer
            })
    
    x = torch.stack(x_batch)
    y = torch.stack(y_batch)
    
    if debug and pair_info:
        print(f"\n[DEBUG] Batch sample ({split}):")
        for i, info in enumerate(pair_info[:3]):
            print(f"  Pair {i+1}:")
            print(f"    Q: {repr(info['question'])}")
            print(f"    A: {repr(info['answer'])}")
    
    return x.to(device), y.to(device)

class ProgressBar:
    """Simple progress bar for training iterations"""
    def __init__(self, total, desc="Training", width=50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.last_update = 0
        self.update_frequency = max(1, total // 50)  # Update ~50 times per epoch
        
    def update(self, current, loss=None, lr=None):
        """Update progress bar to specific value"""
        self.current = current
        progress = min(self.current / self.total, 1.0) if self.total > 0 else 0.0
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)
        
        # Update only periodically to reduce I/O overhead
        if self.current - self.last_update >= self.update_frequency or self.current >= self.total:
            info_parts = [f"{self.desc}: |{bar}| {self.current}/{self.total} ({100*progress:.1f}%)"]
            if loss is not None:
                info_parts.append(f"loss={loss:.4f}")
            if lr is not None:
                info_parts.append(f"lr={lr:.2e}")
            info = " ".join(info_parts)
            
            # Use \r to overwrite same line (avoids scrolling, more efficient)
            sys.stdout.write(f"\r{info}")
            sys.stdout.flush()
            self.last_update = self.current
            
    def close(self):
        """Close progress bar and move to next line"""
        sys.stdout.write("\n")
        sys.stdout.flush()

def get_lr(iter):
    """
    Get learning rate for current iteration with warmup and decay.
    
    Warmup: linearly increase from 0 to learning_rate (helps stabilize training)
    Decay: smoothly decrease from learning_rate to min_learning_rate (helps fine-tuning)
    """
    if iter < warmup_iters:
        # Warmup phase: gradually increase LR to prevent early instability
        return learning_rate * (iter / warmup_iters)
    
    # Decay phase: reduce LR for fine-tuning
    decay_iters = max_iters - warmup_iters
    progress = (iter - warmup_iters) / decay_iters
    
    if decay_style == 'cosine':
        # Cosine decay: smooth S-curve from max to min LR
        lr = min_learning_rate + (learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
    elif decay_style == 'linear':
        # Linear decay: straight line from max to min LR
        lr = learning_rate - (learning_rate - min_learning_rate) * progress
    else:
        lr = learning_rate
    
    return max(lr, min_learning_rate)

@torch.no_grad()
def estimate_loss(model):
    """Estimate loss on train/val splits (averaged over eval_iters batches)"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        non_pad_counts = []
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            non_pad = (Y != pad_token_id).sum().item()
            non_pad_counts.append(non_pad)
        out[split] = losses.mean()
        avg_non_pad = sum(non_pad_counts) / len(non_pad_counts) if non_pad_counts else 0
        out[f'{split}_non_pad_tokens'] = avg_non_pad
    model.train()
    return out

def save_checkpoint(model, optimizer, tokenizer, iter, best_loss, model_path='model.pt', is_best=False):
    # Compiled models wrap the original model in _orig_mod - we need the unwrapped state dict
    if hasattr(model, '_orig_mod'):
        model_state_dict = model._orig_mod.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'iter': iter,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'best_loss': best_loss,
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout,
        'learning_rate': learning_rate,
    }
    torch.save(checkpoint, model_path)
    
    # Also save best model separately
    if is_best:
        torch.save(checkpoint, 'best_model.pt')
        print(f"Best model saved (val loss: {best_loss:.4f})")
    
    print(f"\nCheckpoint saved at iteration {iter} to {model_path}")

def load_checkpoint(model_path='model.pt'):
    if not os.path.exists(model_path):
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint loaded from {model_path}")
    if 'iter' in checkpoint:
        print(f"Resuming from iteration {checkpoint['iter']}")
    return checkpoint

def quantize_model_int8(model):
    """
    Apply int8 dynamic quantization to Linear layers.
    Keeps embeddings as float16 (quantization doesn't work well for embeddings).
    Reduces model size from ~31MB (float16) to ~15MB (int8).
    """
    if not QUANTIZATION_AVAILABLE:
        print("Warning: PyTorch quantization not available, skipping int8 quantization")
        return model
    
    # Quantize Linear layers (attention, feedforward, lm_head)
    # Embeddings stay float16 (they're not quantized)
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8
    )
    
    return quantized_model

class Head(nn.Module):
    """One head of self-attention - learns what to pay attention to"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask: lower triangular matrix prevents looking at future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        # Attention scores: how much each token attends to each other token
        # Scale by sqrt(head_size) to prevent softmax saturation
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5  # (B, T, T)
        # Apply causal mask: set future positions to -inf (they become 0 after softmax)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T) - probabilities
        wei = self.dropout(wei)
        
        # Weighted aggregation: combine values based on attention weights
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """SwiGLU feedforward network: gate * up, then down projection"""
    def __init__(self, n_embd):
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, 4 * n_embd)
        self.up_proj = nn.Linear(n_embd, 4 * n_embd)
        self.down_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: SiLU(gate) * up, then down project
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        return self.dropout(self.down_proj(x))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)

class Block(nn.Module):
    """Transformer block: self-attention (communication) + feedforward (computation)"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = RMSNorm(n_embd)  # Pre-norm: normalize before attention/ffwd
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        # Residual connections: allow gradients to flow through
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Token embeddings: each token ID maps to a learned vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Position embeddings: each position gets a learned vector (tells model token order)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        # Language model head: projects embeddings to vocabulary logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Weight initialization: smaller weights prevent large initial activations"""
        if isinstance(module, nn.Linear):
            # Xavier-like init scaled down for transformer stability
            fan_in = module.weight.size(1)
            std = (2.0 / fan_in) ** 0.5 * 0.1
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Output layer: even smaller init to prevent extreme logits
        if isinstance(module, nn.Linear) and module is self.lm_head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C) - add token and position embeddings
        x = self.blocks(x)  # (B,T,C) - process through transformer blocks
        x = self.ln_f(x)  # (B,T,C) - final layer norm
        logits = self.lm_head(x)  # (B,T,vocab_size) - project to vocabulary

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=pad_token_id, label_smoothing=label_smoothing)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens autoregressively: predict next token, append, repeat"""
        original_length = idx.shape[1]
        with torch.inference_mode():
            for i in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]  # Only last position (B, C)
                
                # Temperature: >1 = more random, <1 = more deterministic
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Top-k: only consider top k most likely tokens (reduces nonsense)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)  # (B, C)
                
                # Don't sample padding tokens
                probs[:, pad_token_id] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
                
                # Early stopping: check if we've generated a complete DSL response
                if i % 5 == 0:
                    new_tokens = idx[:, original_length:]
                    if new_tokens.shape[1] > 0:
                        decoded_new = decode(new_tokens[0].tolist())
                        if decoded_new:
                            # Look for DSL patterns (S:, T:, C:, E:, CL)
                            if any(pattern in decoded_new for pattern in ['S:', 'T:', 'C:', 'E:', '\nCL', 'CL\n']):
                                if decoded_new.strip().endswith(('\n', 'CL')):
                                    break
                            # Stop on double newline (end of response)
                            if decoded_new.count('\n') >= 2:
                                break

        return idx[:, original_length:]

def train_model(model_path='model.pt', checkpoint_interval=100):
    global train_pairs, val_pairs, train_pairs_by_file, val_pairs_by_file
    
    checkpoint = load_checkpoint(model_path)
    
    # Lazy loading: only load data when training starts (saves memory during inference)
    if train_pairs is None or val_pairs is None:
        if checkpoint is not None:
            print("Resuming training - reloading and shuffling pairs...")
        else:
            print("Starting new training - loading pairs...")
        
        text, pairs, pairs_by_file = load_all_input_files()
        print("Encoding and preprocessing pairs...")
        
        encoded_pairs = []
        pairs_by_file_encoded = {}
        pair_lengths = []
        
        for filepath, file_pairs_list in pairs_by_file.items():
            file_encoded = []
            for pair in file_pairs_list:
                tokens = encode(pair)
                if len(tokens) > 0:
                    if len(tokens) > block_size:
                        tokens = tokens[:block_size]
                    
                    if len(tokens) < block_size:
                        pad_length = block_size - len(tokens)
                        pad_tensor = torch.full((pad_length,), pad_token_id, dtype=torch.long)
                        tokens = torch.cat([torch.tensor(tokens, dtype=torch.long), pad_tensor])
                    else:
                        tokens = torch.tensor(tokens, dtype=torch.long)
                    
                    encoded_pairs.append(tokens)
                    file_encoded.append(tokens)
                    pair_lengths.append(len(tokens))
            
            pairs_by_file_encoded[filepath] = file_encoded
        
        if pair_lengths:
            avg_len = sum(pair_lengths) / len(pair_lengths)
            max_len = max(pair_lengths)
            min_len = min(pair_lengths)
            print(f"Pair length stats - Avg: {avg_len:.1f}, Min: {min_len}, Max: {max_len}, Block size: {block_size}")
            if max_len > block_size:
                print(f"  Warning: {sum(1 for l in pair_lengths if l > block_size)} pairs exceed block_size and will be truncated")
        
        random.shuffle(encoded_pairs)
        
        n = int(0.9 * len(encoded_pairs))
        train_pairs = encoded_pairs[:n]
        val_pairs = encoded_pairs[n:]
        
        train_pairs_by_file = {}
        val_pairs_by_file = {}
        for filepath, file_encoded in pairs_by_file_encoded.items():
            n_file = int(0.9 * len(file_encoded))
            train_pairs_by_file[filepath] = file_encoded[:n_file]
            val_pairs_by_file[filepath] = file_encoded[n_file:]
        
        print(f"Shuffled pairs - Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")
        
        print("\n[Validation] Sample pairs being trained:")
        for i, pair_tokens in enumerate(train_pairs[:3]):
            decoded = decode(pair_tokens.tolist())
            parts = decoded.split('\n', 1)
            question = parts[0] if parts else ""
            answer = parts[1].strip() if len(parts) > 1 else ""
            print(f"  Train pair {i+1} ({len(pair_tokens)} tokens):")
            print(f"    Q: {repr(question)}")
            print(f"    A: {repr(answer[:100])}")
        for i, pair_tokens in enumerate(val_pairs[:2]):
            decoded = decode(pair_tokens.tolist())
            parts = decoded.split('\n', 1)
            question = parts[0] if parts else ""
            answer = parts[1].strip() if len(parts) > 1 else ""
            print(f"  Val pair {i+1} ({len(pair_tokens)} tokens):")
            print(f"    Q: {repr(question)}")
            print(f"    A: {repr(answer[:100])}")
    else:
        print("Using already loaded training data")
    
    model = GPTLanguageModel()
    m = model.to(device)
    
    if device == 'mps':
        torch.mps.empty_cache()
    
    start_iter = 0
    best_loss = float('inf')
    optimizer = None
    saved_checkpoint = checkpoint
    
    if checkpoint is not None:
        if checkpoint['vocab_size'] != vocab_size:
            print(f"Warning: Saved vocab_size ({checkpoint['vocab_size']}) != current vocab_size ({vocab_size})")
            print("Reinitializing model with new vocab_size...")
            model = GPTLanguageModel()
            m = model.to(device)
        else:
            # Normalize checkpoint keys: compiled models wrap in _orig_mod, we need unwrapped state
            normalized_state_dict = normalize_checkpoint_state_dict(checkpoint['model_state_dict'])
            m.load_state_dict(normalized_state_dict)
            start_iter = checkpoint.get('iter', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resuming training from iteration {start_iter}")
    
    # CRITICAL: Compile AFTER loading weights (compilation changes parameter names)
    try:
        compile_mode = 'reduce-overhead' if device == 'mps' else 'default'
        m = torch.compile(m, mode=compile_mode)
        print(f"Model compiled with torch.compile() (mode: {compile_mode})")
    except:
        print("torch.compile() not available (requires PyTorch 2.0+)")
    
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Increased weight decay to combat severe overfitting
    
    if checkpoint is not None and checkpoint['vocab_size'] == vocab_size:
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # CRITICAL: Override LR from checkpoint (prevents loss spikes when resuming)
            checkpoint_lr = checkpoint.get('learning_rate', learning_rate)
            if abs(checkpoint_lr - learning_rate) > 1e-8:
                print(f"Learning rate override: checkpoint had {checkpoint_lr:.2e}, overriding to {learning_rate:.2e}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            
            print("Optimizer state loaded (learning rate overridden to current script value)")
    
    loss_history = []
    last_loss = None
    val_loss_history = []  # Track validation loss for early stopping
    no_improve_count = 0  # Count evaluations without improvement
    # Initialize best_val_loss from checkpoint if resuming
    best_val_loss = checkpoint.get('best_loss', float('inf')) if checkpoint is not None else float('inf')
    
    print(f"\nLearning rate schedule:")
    print(f"  Initial LR: {learning_rate:.2e}")
    print(f"  Min LR: {min_learning_rate:.2e}")
    print(f"  Warmup iterations: {warmup_iters}")
    print(f"  Decay style: {decay_style}")
    print(f"  Max iterations: {max_iters}")
    print(f"  Final LR (at max_iters): {get_lr(max_iters-1):.2e}")
    print(f"  Early stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta}\n")
    
    pbar = None
    eval_start_iter = start_iter
    
    if start_iter % eval_interval != 0:
        next_eval = ((start_iter // eval_interval) + 1) * eval_interval
        next_eval = min(next_eval, max_iters - 1)
        pbar = ProgressBar(next_eval - start_iter, desc=f"Epoch {start_iter//eval_interval + 1}")
        eval_start_iter = start_iter
    
    for iter in range(start_iter, max_iters):
        current_lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        debug_batch = (iter == start_iter) or (iter % 1000 == 0)
        xb, yb = get_batch('train', debug=debug_batch)
        
        non_pad_count = (yb != pad_token_id).sum().item()
        if non_pad_count == 0:
            print(f"\nWARNING: All padding tokens in batch at iteration {iter}! Skipping.")
            continue
        
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = m(xb, yb)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWARNING: NaN/Inf loss at iteration {iter}!")
            print(f"  Loss value: {loss.item()}")
            print(f"  Learning rate: {current_lr:.2e}")
            
            has_nan_params = False
            model_to_check = m._orig_mod if hasattr(m, '_orig_mod') else m
            for name, param in model_to_check.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    print(f"  Model parameter '{name}' contains NaN/Inf!")
                    has_nan_params = True
                    break
            
            non_pad_tokens = (yb != pad_token_id).sum().item()
            print(f"  Batch stats: non-pad tokens={non_pad_tokens}/{yb.numel()}, pad_percent={(1-non_pad_tokens/yb.numel())*100:.1f}%")
            print(f"  Logits stats: min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}")
            if torch.isnan(logits).any():
                print(f"  Logits contain NaN!")
            if torch.isinf(logits).any():
                print(f"  Logits contain Inf!")
            
            if has_nan_params:
                print(f"  Model weights corrupted! Reloading from checkpoint...")
                if saved_checkpoint is not None:
                    model_to_reload = m._orig_mod if hasattr(m, '_orig_mod') else m
                    normalized_state_dict = normalize_checkpoint_state_dict(saved_checkpoint['model_state_dict'])
                    model_to_reload.load_state_dict(normalized_state_dict)
                    if saved_checkpoint.get('optimizer_state_dict') is not None:
                        optimizer.load_state_dict(saved_checkpoint['optimizer_state_dict'])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                    print(f"  Reloaded from checkpoint at iteration {saved_checkpoint.get('iter', 0)}")
                else:
                    print(f"  ERROR: No checkpoint available to reload from!")
                    break
            
            nan_count = sum(1 for _ in range(iter - start_iter) if iter > start_iter + 10)
            if nan_count > 10:
                print(f"  Stopping training - NaN persists after 10 attempts. Reload checkpoint manually.")
                break
            
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        optimizer.step()
        
        loss_value = loss.item()
        if not (math.isnan(loss_value) or math.isinf(loss_value)):
            loss_history.append(loss_value)
            if len(loss_history) > 100:
                loss_history.pop(0)
        
        if pbar is not None:
            iter_in_epoch = iter - eval_start_iter
            if iter_in_epoch % max(1, eval_interval // 50) == 0:
                if loss_history:
                    avg_loss = sum(loss_history[-10:]) / min(10, len(loss_history))
                    if math.isnan(avg_loss) or math.isinf(avg_loss):
                        avg_loss = loss_value if not (math.isnan(loss_value) or math.isinf(loss_value)) else 0.0
                else:
                    avg_loss = loss_value if not (math.isnan(loss_value) or math.isinf(loss_value)) else 0.0
                pbar.update(iter_in_epoch, loss=avg_loss, lr=current_lr)
        
        if iter % 100 == 0:
            if loss_history:
                avg_loss = sum(loss_history[-10:]) / min(10, len(loss_history))
                if math.isnan(avg_loss) or math.isinf(avg_loss):
                    avg_loss = loss_value if not (math.isnan(loss_value) or math.isinf(loss_value)) else 0.0
            else:
                avg_loss = loss_value if not (math.isnan(loss_value) or math.isinf(loss_value)) else 0.0
            print(f"\niter {iter}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
        
        if device == 'mps' and iter % 100 == 0:
            torch.mps.empty_cache()
        
        if iter % checkpoint_interval == 0 and iter > 0:
            save_checkpoint(m, optimizer, tokenizer, iter, best_loss, model_path)
        
        if iter % (len(train_pairs) // batch_size) == 0:
            random.shuffle(train_pairs)
            print(f"\nReshuffled training pairs at iteration {iter}")
        
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"\nEvaluating at iteration {iter}")
            if pbar is not None:
                pbar.close()
                pbar = None
            if iter < max_iters - 1 and iter % eval_interval == 0:
                next_eval = min(iter + eval_interval, max_iters - 1)
                pbar = ProgressBar(next_eval - iter, desc=f"Epoch {iter//eval_interval + 1}")
                eval_start_iter = iter
            losses = estimate_loss(m)
            train_loss = losses['train']
            val_loss = losses['val']
            
            if len(loss_history) >= 10:
                recent_avg = sum(loss_history[-10:]) / 10
                older_avg = sum(loss_history[:10]) / 10 if len(loss_history) >= 20 else recent_avg
                trend = "↓" if recent_avg < older_avg else "↑" if recent_avg > older_avg else "→"
            else:
                trend = "?"
            
            if last_loss is not None:
                improvement = last_loss - val_loss
                improvement_str = f" ({improvement:+.4f})" if abs(improvement) > 0.0001 else ""
            else:
                improvement_str = ""
            
            train_non_pad = losses.get('train_non_pad_tokens', 0)
            val_non_pad = losses.get('val_non_pad_tokens', 0)
            pad_percentage = (1 - train_non_pad / (batch_size * block_size)) * 100 if train_non_pad > 0 else 0
            
            lr_phase = "warmup" if iter < warmup_iters else "decay"
            print(f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}{improvement_str} {trend} | LR: {current_lr:.2e} ({lr_phase})")
            print(f"  Non-pad tokens: train={train_non_pad:.0f}, val={val_non_pad:.0f}, padding={pad_percentage:.1f}%")
            
            # Overfitting detection
            if val_loss > train_loss * 1.5:
                gap_ratio = val_loss / train_loss
                print(f"  ⚠️  OVERFITTING WARNING: Val loss is {gap_ratio:.1f}x higher than train loss!")
                if gap_ratio > 2.0:
                    print(f"  ⚠️  SEVERE overfitting detected! Consider:")
                    print(f"      - Increasing dropout (current: {dropout})")
                    print(f"      - Reducing learning rate (current: {current_lr:.2e})")
                    print(f"      - Adding more training data")
                    print(f"      - Early stopping if val loss doesn't improve")
            
            if iter % 1000 == 0 and iter > 0:
                print(f"  [Sample generation at iter {iter}]:")
                with torch.no_grad():
                    test_prompts = ["hello", "what is 2+2", "go to kitchen"]
                    for sample_prompt in test_prompts:
                        sample_tokens = encode(sample_prompt + "\n")
                        sample_context = torch.tensor([sample_tokens], dtype=torch.long, device=device)
                        sample_gen = m.generate(sample_context, max_new_tokens=30, temperature=0.8, top_k=50)
                        sample_text = decode(sample_gen[0].tolist())
                        print(f"    Q: '{sample_prompt}'")
                        print(f"    A: '{sample_text[:100]}...'")
            
            last_loss = val_loss
            
            # Early stopping logic
            val_loss_history.append(val_loss)
            should_stop = False
            if val_loss < best_val_loss - early_stop_min_delta:
                # Validation loss improved significantly
                best_val_loss = val_loss
                no_improve_count = 0
                print(f"  ✓ Validation loss improved to {val_loss:.4f} (best: {best_val_loss:.4f})")
            else:
                # No improvement
                no_improve_count += 1
                if no_improve_count >= early_stop_patience:
                    print(f"\n  ⚠️  EARLY STOPPING: No improvement for {no_improve_count} evaluations")
                    print(f"  Best validation loss: {best_val_loss:.4f}")
                    print(f"  Current validation loss: {val_loss:.4f}")
                    print(f"  Stopping training to prevent overfitting...")
                    should_stop = True
                else:
                    print(f"  ⚠️  No improvement ({no_improve_count}/{early_stop_patience}), best: {best_val_loss:.4f}")
            
            if device == 'mps':
                if hasattr(torch.mps, 'current_allocated_memory'):
                    allocated = torch.mps.current_allocated_memory() / 1024**3
                    print(f"MPS allocated memory: {allocated:.2f} GB")
            
            is_best = losses['val'] < best_loss
            if is_best:
                best_loss = losses['val']
            
            save_checkpoint(m, optimizer, tokenizer, iter, best_loss, model_path, is_best)
            
            # Break after saving checkpoint if early stopping triggered
            if should_stop:
                break
    
    # Close progress bar if still open
    if pbar is not None:
        pbar.close()
    
    print("Training completed!")
    return m

def inference_model(model_path='model.pt', prompt="", num_tokens=500, output_file=None, use_int8=False):
    checkpoint = load_checkpoint(model_path)
    
    if checkpoint is None:
        print(f"Error: No checkpoint found at {model_path}")
        print("Please train the model first using: python gpt.py train")
        return
    
    if checkpoint['vocab_size'] != vocab_size:
        print(f"Error: Vocab size mismatch!")
        print(f"  Checkpoint: {checkpoint['vocab_size']}, Current: {vocab_size}")
        return
    
    model = GPTLanguageModel()
    m = model.to(device)
    
    # CRITICAL: Load weights BEFORE compiling (compilation changes parameter names)
    normalized_state_dict = normalize_checkpoint_state_dict(checkpoint['model_state_dict'])
    m.load_state_dict(normalized_state_dict)
    m.eval()
    
    # Convert to float16 first (embeddings stay float16 even with int8 quantization)
    m = m.half()
    
    # Apply int8 quantization if requested (quantizes Linear layers, keeps embeddings float16)
    if use_int8:
        m = quantize_model_int8(m)
    
    try:
        # MPS needs 'reduce-overhead', CUDA can use 'max-autotune' for better optimization
        compile_mode = 'reduce-overhead' if device == 'mps' else 'max-autotune'
        m = torch.compile(m, mode=compile_mode, fullgraph=True)
        print(f"Model compiled with torch.compile() (mode: {compile_mode})")
    except:
        print("torch.compile() not available (requires PyTorch 2.0+)")
    
    total_params = sum(p.numel() for p in m.parameters())
    if use_int8:
        model_size_mb = total_params * 1.2 / 1e6  # ~1 byte/param average with some overhead
    else:
        model_size_mb = total_params * 2 / 1e6  # float16 = 2 bytes/param
    
    print(f"Model loaded: {total_params/1e6:.2f} M parameters (~{model_size_mb:.2f} MB)")
    if 'iter' in checkpoint:
        print(f"Trained for {checkpoint['iter']} iterations")
    
    if prompt:
        # Training format: "Question\nAnswer\n\n", so prompt should be "Question\n"
        formatted_prompt = prompt if prompt.endswith('\n') else prompt + '\n'
        context_tokens = encode(formatted_prompt)
        context = torch.tensor([context_tokens], dtype=torch.long, device=device)
        print(f"Prompt: {prompt}")
        print(f"Formatted prompt (for model): {formatted_prompt!r}")
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print("Generating from empty context...")
    
    print("\n" + "="*50)
    print("Generated text:")
    print("="*50)
    
    with torch.no_grad():
        generated_tokens = m.generate(context, max_new_tokens=num_tokens, temperature=0.8, top_k=50)
        generated_text = decode(generated_tokens[0].tolist())
        print(generated_text)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nOutput saved to {output_file}")
    
    return generated_text

# Global memory system instance (initialized on first use)
_memory_system = None

def get_memory_system(model=None, tokenizer_encode=None, dev_mode=False):
    """Get or initialize the global memory system"""
    global _memory_system
    if _memory_system is None:
        _memory_system = HybridMemorySystem(
            model=model,
            tokenizer_encode=encode if tokenizer_encode is None else tokenizer_encode,
            max_entries=1000,
            summary_threshold=50,
            embedding_dim=128,
            retrieval_k=5,
            storage_file=None,
            dev_mode=dev_mode
        )
    elif model is not None:
        # Update model reference if provided
        _memory_system.model = model
    return _memory_system

def compress_context(context_tokens, max_tokens, context_prefix_tokens=None):
    if len(context_tokens) <= max_tokens:
        return context_tokens
    
    if context_prefix_tokens:
        prefix_len = len(context_prefix_tokens)
        prompt_tokens = context_tokens[prefix_len:]
        
        if prefix_len >= max_tokens:
            return context_prefix_tokens[:max_tokens]
        
        available_for_prompt = max_tokens - prefix_len
        if len(prompt_tokens) > available_for_prompt:
            return context_prefix_tokens + prompt_tokens[-available_for_prompt:]
    
    return context_tokens[-max_tokens:]

def dispatch(model, prompt="", max_new_tokens=200, max_recursions=5, recursion_depth=0, tool_call_count=0, memory_system=None):
    """
    Dispatch method that handles model responses and tool execution.
    Supports recursive tool calling up to max_recursions times.
    Falls back to cloud (CL) if more than 3 tool calls are needed.
    Uses memory system for RAG-based history retrieval.
    
    Args:
        model: The GPT model instance
        prompt: Initial user prompt
        max_new_tokens: Maximum tokens to generate per call
        max_recursions: Maximum depth of recursive tool calls (default 5)
        recursion_depth: Current recursion depth (internal use)
        tool_call_count: Total number of tool calls made so far (internal use)
        memory_system: Optional memory system instance (uses global if None)
    
    Returns:
        Final response string after all tool calls are resolved
    """
    if memory_system is None:
        memory_system = get_memory_system(model=model, tokenizer_encode=encode)
    if recursion_depth >= max_recursions:
        return "E:Max recursions reached"
    
    if tool_call_count > 3:
        return "CL"
    
    context_prefix = ""
    single_line_prompt = ""
    if prompt and memory_system and recursion_depth == 0:
        clean_prompt = prompt.split('|R:')[0].strip()
        context_prefix = memory_system.get_relevant_context(clean_prompt, k=1)
        if context_prefix:
            print(f"[History] Retrieved context: {context_prefix[:100]}...")
    
    if prompt:
        single_line_prompt = prompt.replace('\n', ' ').strip()
        if context_prefix:
            full_prompt = context_prefix + single_line_prompt + '\n'
        else:
            full_prompt = single_line_prompt + '\n'
        
        context_tokens = encode(full_prompt)
        
        if len(context_tokens) > block_size:
            context_prefix_tokens = None
            if context_prefix:
                prefix_only_tokens = encode(context_prefix)
                if len(prefix_only_tokens) <= len(context_tokens):
                    if context_tokens[:len(prefix_only_tokens)] == prefix_only_tokens:
                        context_prefix_tokens = prefix_only_tokens
            
            original_len = len(context_tokens)
            context_tokens = compress_context(context_tokens, max_tokens=block_size, 
                                              context_prefix_tokens=context_prefix_tokens)
            print(f"[Dispatch] Compressed: {original_len} -> {len(context_tokens)} tokens")
        
        context = torch.tensor([context_tokens], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
        generated_text = decode(generated_tokens[0].tolist())
    
    import re
    
    generated_text_clean = generated_text
    generated_text_clean = re.sub(r'CXT:[^\|]+\|\s*', '', generated_text_clean)
    
    if single_line_prompt and single_line_prompt in generated_text_clean:
        parts = generated_text_clean.rsplit(single_line_prompt, 1)
        if len(parts) > 1:
            generated_text_clean = parts[-1].strip()
    
    if context_prefix:
        context_marker = context_prefix.split('|')[0] if '|' in context_prefix else context_prefix
        if context_marker in generated_text_clean:
            generated_text_clean = generated_text_clean.replace(context_marker, '').strip()
    
    dsl_response = None
    lines = generated_text_clean.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(('S:', 'T:', 'C:', 'E:')) or line == 'CL':
            dsl_response = line
            break
        if ';' in line and any(line.startswith(prefix) for prefix in ['S:', 'T:', 'C:']):
            dsl_response = line
            break
    
    if not dsl_response:
        dsl_pattern = r'(?:^|\n)([STCE]:[^\n]+|CL)(?:\n|$)'
        match = re.search(dsl_pattern, generated_text_clean)
        if match:
            dsl_response = match.group(1).strip()
    
    if not dsl_response:
        # Try to find any DSL pattern in the cleaned text
        dsl_pattern = r'([STCE]:[^\n;]+|CL)'
        match = re.search(dsl_pattern, generated_text_clean)
        if match:
            dsl_response = match.group(1).strip()
        else:
            dsl_response = generated_text_clean.strip().split('\n')[0] if generated_text_clean.strip() else generated_text_clean
    
    # Debug: show what was extracted
    if not dsl_response or dsl_response.strip() == "":
        print(f"[Dispatch] Warning: Empty DSL response")
        print(f"[Dispatch] Full generated text ({len(generated_text)} chars): {repr(generated_text[:300])}")
        print(f"[Dispatch] Cleaned text ({len(generated_text_clean)} chars): {repr(generated_text_clean[:300])}")
        print(f"[Dispatch] Prompt was: {repr(single_line_prompt)}")
    
    if DSLDecoder.is_tool(dsl_response):
        tool_chain = DSLDecoder.extract_tool_chain(dsl_response)
        
        if tool_chain and len(tool_chain) > 1:
            # Check if total tool calls would exceed 3
            new_tool_count = tool_call_count + len(tool_chain)
            if new_tool_count > 3:
                return "CL"
            
            # Execute chained tools sequentially (e.g., T:date,tomorrow;T:wthr,<date>)
            print(f"[Recursion {recursion_depth + 1}] Executing tool chain: {len(tool_chain)} tools")
            results = []
            
            for i, (tool_name, tool_args) in enumerate(tool_chain):
                print(f"  Tool {i+1}/{len(tool_chain)}: {tool_name}({tool_args})")
                
                # Replace <date> placeholder with previous tool result
                if '<date>' in tool_args and results:
                    tool_args = tool_args.replace('<date>', results[-1])
                
                if '<res>' in tool_args and results:
                    tool_args = tool_args.replace('<res>', results[-1])
                
                tool_result = execute_tool(tool_name, tool_args)
                
                if tool_result.startswith("E:"):
                    return DSLEncoder.encode_error(tool_result[2:])
                
                results.append(tool_result)
                print(f"    Result: {tool_result}")
            
            # Recursively call model with all tool results as context (DSL format: |R:result)
            # Append |R: results without line breaks, unified separator format
            prompt_clean = prompt.replace('\n', ' ').strip()
            results_str = "|".join([f"R:{r}" for r in results])
            new_prompt = f"{prompt_clean} |{results_str}"  # Unified |R: format
            return dispatch(model, new_prompt, max_new_tokens, max_recursions, recursion_depth + 1, new_tool_count, memory_system)
        
        else:
            # Single tool call
            tool_info = DSLDecoder.extract_tool(dsl_response)
            if tool_info:
                # Check if total tool calls would exceed 3
                new_tool_count = tool_call_count + 1
                if new_tool_count > 3:
                    return "CL"
                
                tool_name, tool_args = tool_info
                print(f"[Recursion {recursion_depth + 1}] Executing tool: {tool_name}({tool_args})")
                
                tool_result = execute_tool(tool_name, tool_args)
                
                if tool_result.startswith("E:"):
                    return DSLEncoder.encode_error(tool_result[2:])
                
                # Recursively call model with tool result as context (DSL format: |R:result)
                # Append |R: result without line breaks, unified separator format
                prompt_clean = prompt.replace('\n', ' ').strip()
                new_prompt = f"{prompt_clean} |R:{tool_result}"  # Unified |R: format
                return dispatch(model, new_prompt, max_new_tokens, max_recursions, recursion_depth + 1, new_tool_count, memory_system)
    
    # Commands (C:) are executed but don't require recursive model calls
    if dsl_response:
        print(f"[Dispatch] DSL response: {dsl_response}")
    else:
        print(f"[Dispatch] Warning: No DSL response extracted from: {generated_text_clean[:100]}")
    
    decoded = DSLDecoder.decode(dsl_response) if dsl_response else None
    if decoded and decoded['type'] == 'response_command':
        for item in decoded['content']:
            if item.get('type') == 'command':
                cmd = item.get('command', '')
                args = item.get('args', '')
                print(f"[Command] {cmd}({args})")
    
    # Store conversation in memory system (only for original prompt, not recursive calls)
    if memory_system and recursion_depth == 0 and prompt:
        # Clean prompt (remove history context and tool results)
        # First remove any history context (lines that start with previous prompts)
        clean_prompt = prompt.split('\n')[-1]  # Get last line (current prompt)
        clean_prompt = clean_prompt.split('|R:')[0].strip()  # Remove tool results if present
        
        # Only store if we have both prompt and response
        if clean_prompt and dsl_response and clean_prompt != dsl_response:
            memory_system.add_entry(clean_prompt, dsl_response)
            print(f"[History] Stored entry: Q='{clean_prompt[:50]}...' A='{dsl_response[:50]}...'")
    
    return dsl_response

def dispatch_inference(model_path='model.pt', prompt="", max_new_tokens=200, max_recursions=5, use_int8=False, dev_mode=False):
    """Convenience function to load model and run dispatch with tool execution"""
    checkpoint = load_checkpoint(model_path)
    
    if checkpoint is None:
        print(f"Error: No checkpoint found at {model_path}")
        return None
    
    if checkpoint['vocab_size'] != vocab_size:
        print(f"Error: Vocab size mismatch!")
        return None
    
    model = GPTLanguageModel()
    m = model.to(device)
    
    # Load weights BEFORE compiling
    normalized_state_dict = normalize_checkpoint_state_dict(checkpoint['model_state_dict'])
    m.load_state_dict(normalized_state_dict)
    m.eval()
    
    m = m.half()
    
    # Apply int8 quantization
    if use_int8:
        m = quantize_model_int8(m)
    
    compile_mode = 'reduce-overhead' if device == 'mps' else 'max-autotune'
    try:
        m = torch.compile(m, mode=compile_mode, fullgraph=True)
    except:
        pass
    
    total_params = sum(p.numel() for p in m.parameters())
    if use_int8:
        model_size_mb = total_params * 1.2 / 1e6
    else:
        model_size_mb = total_params * 2 / 1e6
    
    print(f"Model loaded: {total_params/1e6:.2f} M parameters (~{model_size_mb:.2f} MB)")
    
    # Initialize memory system with model
    memory_system = get_memory_system(model=m, tokenizer_encode=encode, dev_mode=dev_mode)
    stats = memory_system.get_stats()
    print(f"Memory system initialized: {stats['total_entries']} entries")
    print(f"Storage mode: {stats['storage_mode']} ({stats['storage_file']})")
    
    print(f"Prompt: {prompt}")
    print("="*50)
    
    result = dispatch(m, prompt, max_new_tokens, max_recursions, memory_system=memory_system)
    
    print("="*50)
    print(f"Final response: {result}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='GPT Language Model - Training and Inference')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train or resume training')
    train_parser.add_argument('--checkpoint', type=str, default='model.pt',
                              help='Path to model checkpoint (default: model.pt)')
    train_parser.add_argument('--checkpoint_interval', type=int, default=200,
                              help='Save checkpoint every N iterations (default: 200)')
    
    # Infer subcommand
    infer_parser = subparsers.add_parser('infer', help='Generate text from trained model')
    infer_parser.add_argument('--checkpoint', type=str, default='model.pt',
                              help='Path to model checkpoint (default: model.pt)')
    infer_parser.add_argument('--int8', action='store_true',
                              help='Use int8 quantization (reduces memory from ~31MB to ~15MB)')
    infer_parser.add_argument('--prompt', type=str, default='',
                              help='Prompt text (default: empty, generates randomly)')
    infer_parser.add_argument('--num_tokens', type=int, default=200,
                              help='Number of tokens to generate (default: 200)')
    infer_parser.add_argument('--output', type=str, default=None,
                              help='Output file for generated text (default: None)')
    infer_parser.add_argument('--dev', action='store_true',
                                help='Use JSON storage format (readable, slower) instead of Pickle')
    
    # Dispatch subcommand
    dispatch_parser = subparsers.add_parser('dispatch', help='Run dispatch with tool execution')
    dispatch_parser.add_argument('--checkpoint', type=str, default='model.pt',
                                 help='Path to model checkpoint (default: model.pt)')
    dispatch_parser.add_argument('--prompt', type=str, required=True,
                                 help='User prompt (required)')
    dispatch_parser.add_argument('--max_tokens', type=int, default=200,
                                help='Max tokens per generation (default: 200)')
    dispatch_parser.add_argument('--max_recursions', type=int, default=5,
                                 help='Max tool call recursions (default: 5)')
    dispatch_parser.add_argument('--int8', action='store_true',
                                 help='Use int8 quantization (reduces memory from ~31MB to ~15MB)')
    dispatch_parser.add_argument('--dev', action='store_true',
                                 help='Use JSON storage format (readable, slower) instead of Pickle')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("="*50)
        print("TRAINING MODE")
        print("="*50)
        train_model(args.checkpoint, args.checkpoint_interval)
    elif args.command == 'infer':
        print("="*50)
        print("INFERENCE MODE")
        print("="*50)
        inference_model(args.checkpoint, args.prompt, args.num_tokens, args.output, use_int8=args.int8)
    elif args.command == 'dispatch':
        print("="*50)
        print("DISPATCH MODE")
        print("="*50)
        dispatch_inference(args.checkpoint, args.prompt, args.max_tokens, args.max_recursions, use_int8=args.int8, dev_mode=args.dev)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
