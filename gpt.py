import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import math
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from dsl import DSLDecoder, DSLEncoder
from tools import execute_tool

torch.set_float32_matmul_precision('medium')  # PyTorch 2.0+


# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
elif torch.backends.mps.is_available():
    device = 'mps'
    torch.backends.mps.allow_tf32 = True

print(f"Using device: {device}")

eval_iters = 100
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

def normalize_checkpoint_state_dict(state_dict):
    """
    Normalize checkpoint state dict to work with uncompiled models.
    Strips '_orig_mod.' prefix if present (from compiled model checkpoints).
    """
    # Check if any keys have '_orig_mod.' prefix
    has_orig_mod = any(k.startswith('_orig_mod.') for k in state_dict.keys())
    
    if has_orig_mod:
        # Strip '_orig_mod.' prefix from all keys
        normalized = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                normalized[k[len('_orig_mod.'):]] = v
            else:
                normalized[k] = v
        return normalized
    else:
        # Already normalized, return as-is
        return state_dict

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
    
    txt_files.sort()  # Sort for consistent ordering
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
    
    # Train on all text files
    tokenizer.train(input_files, trainer)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    
    # Save tokenizer
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

# Load or train tokenizer
tokenizer = load_or_train_tokenizer()

# Create encode/decode functions using the tokenizer
def encode(text):
    encoding = tokenizer.encode(text, add_special_tokens=False)
    return encoding.ids

# def decode(token_ids):
#     """Decode token IDs back to text"""
#     return tokenizer.decode(token_ids, skip_special_tokens=True)

def decode(token_ids):
    # Convert to list if tensor
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    elif not isinstance(token_ids, list):
        token_ids = list(token_ids)
    
    # Decode with proper handling
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # The decoder should handle Ġ and Ċ automatically, but if not:
    # decoded = decoded.replace('Ġ', ' ').replace('Ċ', '\n')
    
    return decoded

# Get vocabulary size from tokenizer
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

# Load all text files from inputs directory
def load_all_input_files(inputs_dir='inputs'):
    input_files = get_input_files(inputs_dir)
    
    if not input_files:
        print(f"Warning: No .txt files found in {inputs_dir}, falling back to input.txt")
        if os.path.exists('input.txt'):
            input_files = ['input.txt']
        else:
            raise ValueError(f"No training files found in {inputs_dir} or input.txt")
    
    print(f"Loading {len(input_files)} file(s) from {inputs_dir}:")
    all_texts = []
    for filepath in input_files:
        print(f"  Loading {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            all_texts.append(text)
            print(f"    Loaded {len(text)} characters")
    
    # Combine all texts
    combined_text = '\n'.join(all_texts)
    print(f"Total combined text: {len(combined_text)} characters")
    return combined_text

# Load all training data
text = load_all_input_files()

# Encode and create data tensor
print("Encoding text...")
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Encoded to {len(data)} tokens")

# Shuffle data before splitting (for better training)
print("Shuffling data...")
shuffle_indices = torch.randperm(len(data))
data = data[shuffle_indices]

# Train and test splits (will be updated in train_model when resuming)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print(f"Train data: {len(train_data)} tokens, Val data: {len(val_data)} tokens")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate all indices at once
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Use advanced indexing (faster)
    x = torch.stack([data[i:i+block_size] for i in ix.tolist()])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix.tolist()])
    
    return x.to(device), y.to(device)

# data loading
# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_checkpoint(model, optimizer, tokenizer, iter, best_loss, model_path='model.pt', is_best=False):
    # Get the underlying model's state dict (strip _orig_mod if compiled)
    # Compiled models wrap the original model in _orig_mod
    if hasattr(model, '_orig_mod'):
        # Model is compiled, get the original model's state dict
        model_state_dict = model._orig_mod.state_dict()
    else:
        # Model is not compiled, use state_dict directly
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
    
    print(f"Checkpoint saved at iteration {iter} to {model_path}")

def load_checkpoint(model_path='model.pt'):
    if not os.path.exists(model_path):
        return None
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Checkpoint loaded from {model_path}")
    if 'iter' in checkpoint:
        print(f"Resuming from iteration {checkpoint['iter']}")
    return checkpoint

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        
        # Precompute rotary frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
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
    def __init__(self, n_embd):
        super().__init__()
        # SwiGLU: gate and up projections
        self.gate_proj = nn.Linear(n_embd, 4 * n_embd)
        self.up_proj = nn.Linear(n_embd, 4 * n_embd)
        self.down_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gate = self.gate_proj(x)  # (B, T, 4*n_embd)
        up = self.up_proj(x)      # (B, T, 4*n_embd)
        # SwiGLU: silu(gate) * up
        x = F.silu(gate) * up     # (B, T, 4*n_embd)
        return self.dropout(self.down_proj(x))

# class FeedForward(nn.Module):
#     def __init__(self, n_embd):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embd, 4 * n_embd),
#             nn.ReLU(),
#             nn.Linear(4 * n_embd, n_embd),
#             nn.Dropout(dropout),
#         )
    
#     def forward(self, x):
#         return self.net(x)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # self.ln1 = nn.LayerNorm(n_embd)
        # self.ln2 = nn.LayerNorm(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2  # silu = swish

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Improved weight initialization following GPT-2 and LLaMA practices"""
        if isinstance(module, nn.Linear):
            # Calculate proper std based on fan-in with GPT-2 scaling
            fan_in = module.weight.size(1)
            std = (2.0 / fan_in) ** 0.5
            std = std * 0.1  # GPT-2 scaling factor for transformer stability
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings: standard initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # Output layer: smaller initialization to prevent large initial logits
        if isinstance(module, nn.Linear) and module is self.lm_head:
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
            if decode(idx[0].tolist())[-1] == '\n':
                break

        return idx[:-1] if len(idx) > 1 else idx

def train_model(model_path='model.pt', checkpoint_interval=100):
    global train_data, val_data
    
    # Try to load existing checkpoint
    checkpoint = load_checkpoint(model_path)
    
    # Shuffle data when resuming training (or on first run)
    if checkpoint is not None:
        print("Resuming training - shuffling data for better training...")
    else:
        print("Starting new training - data already shuffled")
    
    # Reload and shuffle data
    text = load_all_input_files()
    print("Re-encoding and shuffling data...")
    data = torch.tensor(encode(text), dtype=torch.long)
    shuffle_indices = torch.randperm(len(data))
    data = data[shuffle_indices]
    
    # Recreate train/val splits
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Shuffled data - Train: {len(train_data)} tokens, Val: {len(val_data)} tokens")
    
    # Create model
    model = GPTLanguageModel()
    m = model.to(device)
    
    # Clear MPS cache
    if device == 'mps':
        torch.mps.empty_cache()
    
    # Initialize training state
    start_iter = 0
    best_loss = float('inf')
    optimizer = None
    
    if checkpoint is not None:
        # Verify vocab_size matches
        if checkpoint['vocab_size'] != vocab_size:
            print(f"Warning: Saved vocab_size ({checkpoint['vocab_size']}) != current vocab_size ({vocab_size})")
            print("Reinitializing model with new vocab_size...")
            model = GPTLanguageModel()
            m = model.to(device)
        else:
            # Normalize checkpoint keys (strip _orig_mod. if present)
            normalized_state_dict = normalize_checkpoint_state_dict(checkpoint['model_state_dict'])
            m.load_state_dict(normalized_state_dict)
            start_iter = checkpoint.get('iter', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resuming training from iteration {start_iter}")
    
    # Compile model AFTER loading weights
    try:
        m = torch.compile(m, mode='reduce-overhead')
        print("Model compiled with torch.compile()")
    except:
        print("torch.compile() not available (requires PyTorch 2.0+)")
    
    # Print model info
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    
    # Load optimizer state if resuming
    if checkpoint is not None and checkpoint['vocab_size'] == vocab_size:
        if checkpoint.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded")
    
    # Training loop
    for iter in range(start_iter, max_iters):
        # Sample a batch of data
        xb, yb = get_batch('train')
        
        # # Evaluate the loss
        # logits, loss = m(xb, yb)
        # optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        # # Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        # optimizer.step()
        
        # # Evaluate the loss
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='mps', dtype=torch.float16):
            logits, loss = m(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
        optimizer.step()
        
        if device == 'mps' and iter % 100 == 0:
            torch.mps.empty_cache()
        
        # Periodic checkpoint saving
        if iter % checkpoint_interval == 0 and iter > 0:
            save_checkpoint(m, optimizer, tokenizer, iter, best_loss, model_path)
        
        # Evaluation and full checkpoint
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if device == 'mps':
                # Check MPS memory usage
                if hasattr(torch.mps, 'current_allocated_memory'):
                    allocated = torch.mps.current_allocated_memory() / 1024**3  # GB
                    print(f"MPS allocated memory: {allocated:.2f} GB")
            
            # Save checkpoint with best model tracking
            is_best = losses['val'] < best_loss
            if is_best:
                best_loss = losses['val']
            
            save_checkpoint(m, optimizer, tokenizer, iter, best_loss, model_path, is_best)
    
    print("Training completed!")
    return m

def inference_model(model_path='model.pt', prompt="", num_tokens=500, output_file=None):
    checkpoint = load_checkpoint(model_path)
    
    if checkpoint is None:
        print(f"Error: No checkpoint found at {model_path}")
        print("Please train the model first using: python gpt.py --mode train")
        return
    
    # Verify vocab_size matches
    if checkpoint['vocab_size'] != vocab_size:
        print(f"Error: Vocab size mismatch!")
        print(f"  Checkpoint: {checkpoint['vocab_size']}, Current: {vocab_size}")
        return
    
    # Create model
    model = GPTLanguageModel()
    m = model.to(device)
    
    # Normalize checkpoint keys (strip _orig_mod. if present) and load weights BEFORE compiling
    normalized_state_dict = normalize_checkpoint_state_dict(checkpoint['model_state_dict'])
    m.load_state_dict(normalized_state_dict)
    m.eval()  # Set to evaluation mode
    
    # Compile after loading weights
    try:
        # Use 'reduce-overhead' for MPS compatibility, 'max-autotune' for CUDA
        compile_mode = 'reduce-overhead' if device == 'mps' else 'max-autotune'
        m = torch.compile(m, mode=compile_mode, fullgraph=True)
        print(f"Model compiled with torch.compile() (mode: {compile_mode})")
    except:
        print("torch.compile() not available (requires PyTorch 2.0+)")
    
    print(f"Model loaded: {sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")
    if 'iter' in checkpoint:
        print(f"Trained for {checkpoint['iter']} iterations")
    
    # Prepare context
    if prompt:
        # Encode the prompt
        context_tokens = encode(prompt)
        context = torch.tensor([context_tokens], dtype=torch.long, device=device)
        print(f"Prompt: {prompt}")
    else:
        # Empty context (random generation)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        print("Generating from empty context...")
    
    # Generate
    print("\n" + "="*50)
    print("Generated text:")
    print("="*50)
    
    with torch.no_grad():
        generated_tokens = m.generate(context, max_new_tokens=num_tokens)
        generated_text = decode(generated_tokens[0].tolist())
        print(generated_text)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nOutput saved to {output_file}")
    
    return generated_text

def dispatch(model, prompt="", max_new_tokens=200, max_recursions=5, recursion_depth=0):
    """
    Dispatch method that handles model responses and tool execution.
    Supports recursive tool calling up to max_recursions times.
    
    Args:
        model: The GPT model instance
        prompt: Initial user prompt
        max_new_tokens: Maximum tokens to generate per call
        max_recursions: Maximum depth of recursive tool calls (default 5)
        recursion_depth: Current recursion depth (internal use)
    
    Returns:
        Final response string after all tool calls are resolved
    """
    if recursion_depth >= max_recursions:
        return "E:Max recursions reached"
    
    # Prepare context
    if prompt:
        context_tokens = encode(prompt)
        context = torch.tensor([context_tokens], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate response
    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)
        generated_text = decode(generated_tokens[0].tolist())
    
    # Clean up generated text - extract first complete DSL response
    # Remove prompt from generated text if present
    if prompt and prompt in generated_text:
        generated_text = generated_text.split(prompt, 1)[-1].strip()
    
    # Find first valid DSL pattern
    dsl_response = None
    lines = generated_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if it looks like DSL (starts with S:, T:, C:, E:, or CL)
        if line.startswith(('S:', 'T:', 'C:', 'E:')) or line == 'CL':
            dsl_response = line
            break
        # Also check for combined format S:text;C:cmd,args
        if ';' in line and any(line.startswith(prefix) for prefix in ['S:', 'T:', 'C:']):
            dsl_response = line
            break
    
    # If no DSL pattern found, try to find it anywhere in the text
    if not dsl_response:
        import re
        # Look for DSL patterns in the entire text
        dsl_pattern = r'(?:^|\n)([STCE]:[^\n]+|CL)(?:\n|$)'
        match = re.search(dsl_pattern, generated_text)
        if match:
            dsl_response = match.group(1).strip()
    
    # If still no DSL pattern found, use first non-empty line or full text
    if not dsl_response:
        dsl_response = generated_text.strip().split('\n')[0] if generated_text.strip() else generated_text
    
    # Check if response contains a tool call
    if DSLDecoder.is_tool(dsl_response):
        # Check for tool chain (chained tool calls)
        tool_chain = DSLDecoder.extract_tool_chain(dsl_response)
        
        if tool_chain and len(tool_chain) > 1:
            # Execute chained tools sequentially
            print(f"[Recursion {recursion_depth + 1}] Executing tool chain: {len(tool_chain)} tools")
            results = []
            
            for i, (tool_name, tool_args) in enumerate(tool_chain):
                print(f"  Tool {i+1}/{len(tool_chain)}: {tool_name}({tool_args})")
                
                # Replace <date> placeholder with previous result if present
                if '<date>' in tool_args and results:
                    tool_args = tool_args.replace('<date>', results[-1])
                
                # Execute tool
                tool_result = execute_tool(tool_name, tool_args)
                
                # If tool result is an error, return it
                if tool_result.startswith("E:"):
                    return DSLEncoder.encode_error(tool_result[2:])
                
                results.append(tool_result)
                print(f"    Result: {tool_result}")
            
            # Use the last result for the final response
            final_result = results[-1]
            
            # Create new prompt with all tool results and recursively call dispatch
            results_str = " | ".join(results)
            new_prompt = f"{prompt}\nTool results: {results_str}"
            
            # Recursive call
            return dispatch(model, new_prompt, max_new_tokens, max_recursions, recursion_depth + 1)
        
        else:
            # Single tool call
            tool_info = DSLDecoder.extract_tool(dsl_response)
            if tool_info:
                tool_name, tool_args = tool_info
                print(f"[Recursion {recursion_depth + 1}] Executing tool: {tool_name}({tool_args})")
                
                # Execute tool
                tool_result = execute_tool(tool_name, tool_args)
                
                # If tool result is an error, return it
                if tool_result.startswith("E:"):
                    return DSLEncoder.encode_error(tool_result[2:])
                
                # Create new prompt with tool result and recursively call dispatch
                # Format: original prompt + tool result as context
                new_prompt = f"{prompt}\nTool result: {tool_result}"
                
                # Recursive call
                return dispatch(model, new_prompt, max_new_tokens, max_recursions, recursion_depth + 1)
    
    # Check if response contains a command (but not a tool)
    decoded = DSLDecoder.decode(dsl_response)
    if decoded['type'] == 'response_command':
        # Extract command if present
        for item in decoded['content']:
            if item.get('type') == 'command':
                cmd = item.get('command', '')
                args = item.get('args', '')
                print(f"[Command] {cmd}({args})")
                # Commands are executed but don't require recursive model calls
                # They're actions to be performed
    
    return dsl_response

def dispatch_inference(model_path='model.pt', prompt="", max_new_tokens=200, max_recursions=5):
    """
    Convenience function to load model and run dispatch.
    
    Args:
        model_path: Path to model checkpoint
        prompt: User prompt
        max_new_tokens: Maximum tokens per generation
        max_recursions: Maximum tool call recursions
    
    Returns:
        Final response string
    """
    # Load checkpoint
    checkpoint = load_checkpoint(model_path)
    
    if checkpoint is None:
        print(f"Error: No checkpoint found at {model_path}")
        return None
    
    # Verify vocab_size matches
    if checkpoint['vocab_size'] != vocab_size:
        print(f"Error: Vocab size mismatch!")
        return None
    
    # Create model
    model = GPTLanguageModel()
    m = model.to(device)
    
    # Normalize checkpoint keys (strip _orig_mod. if present) and load weights BEFORE compiling
    normalized_state_dict = normalize_checkpoint_state_dict(checkpoint['model_state_dict'])
    m.load_state_dict(normalized_state_dict)
    m.eval()
    
    # Compile after loading weights
    compile_mode = 'reduce-overhead' if device == 'mps' else 'max-autotune'
    try:
        m = torch.compile(m, mode=compile_mode, fullgraph=True)
    except:
        pass  # Compilation is optional
    
    print(f"Model loaded: {sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")
    print(f"Prompt: {prompt}")
    print("="*50)
    
    # Run dispatch
    result = dispatch(m, prompt, max_new_tokens, max_recursions)
    
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
    infer_parser.add_argument('--prompt', type=str, default='',
                              help='Prompt text (default: empty, generates randomly)')
    infer_parser.add_argument('--num_tokens', type=int, default=200,
                              help='Number of tokens to generate (default: 200)')
    infer_parser.add_argument('--output', type=str, default=None,
                              help='Output file for generated text (default: None)')
    
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
        inference_model(args.checkpoint, args.prompt, args.num_tokens, args.output)
    elif args.command == 'dispatch':
        print("="*50)
        print("DISPATCH MODE")
        print("="*50)
        dispatch_inference(args.checkpoint, args.prompt, args.max_tokens, args.max_recursions)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
