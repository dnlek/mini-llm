"""
History system for conversation memory using RAG (Retrieval-Augmented Generation)
Implements vector search and summarization for efficient memory management
"""

import json
import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F


class HistoryEntry:
    """Single conversation entry with embedding"""
    
    def __init__(self, prompt: str, response: str, timestamp: float = None):
        self.prompt = prompt
        self.response = response
        self.timestamp = timestamp or datetime.now().timestamp()
        self.embedding = None  # Will be computed as float16 numpy array
        self.is_summary = False  # True if this is a compressed summary
        self.original_count = 1  # How many entries this summarizes
    
    def to_dsl(self) -> str:
        """Convert to training format for history: prompt\nresponse"""
        # Skip empty entries
        if not self.prompt or not self.response:
            return ""
        # Match training format: Question\nAnswer\n
        return f"{self.prompt}\n{self.response}\n"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'prompt': self.prompt,
            'response': self.response,
            'timestamp': self.timestamp,
            'is_summary': self.is_summary,
            'original_count': self.original_count,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create HistoryEntry from dictionary"""
        entry = cls(data['prompt'], data['response'], data.get('timestamp'))
        entry.is_summary = data.get('is_summary', False)
        entry.original_count = data.get('original_count', 1)
        if data.get('embedding') is not None:
            entry.embedding = np.array(data['embedding'], dtype=np.float16)
        return entry


class HybridMemorySystem:
    """
    Hybrid memory system combining RAG (vector search) and summarization.
    Uses embeddings extracted from the trained model for semantic search.
    """
    
    def __init__(self, 
                 model=None,
                 tokenizer_encode=None,
                 max_entries: int = 1000,
                 summary_threshold: int = 50,
                 embedding_dim: int = 128,
                 retrieval_k: int = 5,
                 storage_file: str = None,
                 dev_mode: bool = False):
        """
        Initialize memory system.
        
        Args:
            model: GPTLanguageModel instance (for extracting embeddings)
            tokenizer_encode: Function to encode text to token IDs
            max_entries: Maximum number of entries before sliding window
            summary_threshold: Summarize every N entries
            embedding_dim: Dimension of embeddings (will be reduced from model dim)
            retrieval_k: Number of top-K entries to retrieve
            storage_file: Path to storage file (auto-determined if None)
            dev_mode: If True, use JSON format (readable, slower). If False, use Pickle (faster, binary)
        """
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.max_entries = max_entries
        self.summary_threshold = summary_threshold
        self.embedding_dim = embedding_dim
        self.retrieval_k = retrieval_k
        self.dev_mode = dev_mode
        
        # Auto-determine storage file based on mode
        if storage_file is None:
            self.storage_file = "history.json" if dev_mode else "history.pkl"
        else:
            self.storage_file = storage_file
        
        self.entries: List[HistoryEntry] = []
        self._embedding_proj = None  # Projection layer to reduce embedding dim
        
        # Load existing history if available
        self.load()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using model's token embeddings.
        Uses mean pooling of token embeddings, then projects to embedding_dim.
        """
        if self.model is None or self.tokenizer_encode is None:
            # Fallback: return zero embedding if model not available
            return np.zeros(self.embedding_dim, dtype=np.float16)
        
        # Encode text to tokens
        try:
            token_ids = self.tokenizer_encode(text)
            if not token_ids:
                return np.zeros(self.embedding_dim, dtype=np.float16)
            
            # Get token embeddings from model
            with torch.no_grad():
                # Access token embedding table
                if hasattr(self.model, '_orig_mod'):
                    # Compiled model
                    token_emb_table = self.model._orig_mod.token_embedding_table
                else:
                    token_emb_table = self.model.token_embedding_table
                
                # Convert token IDs to tensor
                token_tensor = torch.tensor([token_ids], dtype=torch.long, device=token_emb_table.weight.device)
                
                # Get embeddings (B, T, C)
                embeddings = token_emb_table(token_tensor)
                
                # Mean pooling: average over sequence length
                pooled = embeddings.mean(dim=1).squeeze(0)  # (C,)
                
                # Project to embedding_dim if needed
                if pooled.shape[0] != self.embedding_dim:
                    if self._embedding_proj is None:
                        # Create projection layer (lazy initialization)
                        self._embedding_proj = torch.nn.Linear(
                            pooled.shape[0], 
                            self.embedding_dim, 
                            bias=False
                        ).to(pooled.device)
                        # Initialize with small weights
                        torch.nn.init.normal_(self._embedding_proj.weight, mean=0.0, std=0.02)
                    
                    pooled = self._embedding_proj(pooled)
                
                # Normalize and convert to float16 numpy
                pooled_norm = F.normalize(pooled, p=2, dim=0)
                embedding = pooled_norm.cpu().numpy().astype(np.float16)
                
                return embedding
        except Exception as e:
            # Fallback on error
            print(f"Warning: Failed to generate embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float16)
    
    def add_entry(self, prompt: str, response: str):
        """Add new conversation entry and trigger summarization if needed"""
        # Skip empty entries
        if not prompt or not response:
            return
        
        entry = HistoryEntry(prompt, response)
        entry.embedding = self._get_embedding(f"{prompt} {response}")
        
        self.entries.append(entry)
        
        # Summarization: compress old entries periodically
        if len(self.entries) % self.summary_threshold == 0:
            self._summarize_old_entries()
        
        # Sliding window: remove oldest if over limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        # Save after each addition
        self.save()
    
    def _summarize_old_entries(self):
        """Compress old entries into summaries, keeping recent entries uncompressed"""
        # Keep recent entries (last 20) + summaries
        recent_count = 20
        if len(self.entries) <= recent_count:
            return
        
        # Get old entries to summarize (everything except recent)
        old_entries = self.entries[:-recent_count]
        recent_entries = self.entries[-recent_count:]
        
        # Group old entries into chunks for summarization
        chunk_size = 10
        summaries = []
        
        for i in range(0, len(old_entries), chunk_size):
            chunk = old_entries[i:i+chunk_size]
            
            # Create summary entry
            summary_prompt = self._extract_summary([e.prompt for e in chunk], [e.response for e in chunk])
            summary_response = "S:Summarized previous conversations"
            
            summary_entry = HistoryEntry(summary_prompt, summary_response)
            summary_entry.is_summary = True
            summary_entry.original_count = len(chunk)
            summary_entry.embedding = self._get_embedding(summary_prompt)
            
            summaries.append(summary_entry)
        
        # Replace old entries with summaries
        self.entries = summaries + recent_entries
    
    def _extract_summary(self, prompts: List[str], responses: List[str]) -> str:
        """Extract key topics from conversation chunk for summarization"""
        all_text = " ".join(prompts + responses).lower()
        
        # Extract DSL tool calls and commands
        keywords = []
        for text in prompts + responses:
            text_lower = text.lower()
            if "T:" in text:
                keywords.append("tools")
            if "C:" in text:
                keywords.append("commands")
            if any(word in text_lower for word in ["weather", "rain", "sunny", "wthr"]):
                keywords.append("weather")
            if any(word in text_lower for word in ["math", "calculate", "+", "-", "*", "/"]):
                keywords.append("math")
            if any(word in text_lower for word in ["time", "date", "when", "today", "tomorrow"]):
                keywords.append("time")
            if any(word in text_lower for word in ["move", "go", "kitchen", "room"]):
                keywords.append("motion")
        
        # Create summary prompt
        unique_keywords = list(set(keywords))[:5]  # Top 5 topics
        if unique_keywords:
            return f"Previous conversations about {', '.join(unique_keywords)}"
        return "Previous general conversations"
    
    def retrieve_relevant(self, query: str, k: int = None) -> List[HistoryEntry]:
        """
        Vector search: find most relevant history entries using cosine similarity.
        Returns top-K most similar entries.
        """
        if not self.entries:
            return []
        
        k = k or self.retrieval_k
        query_embedding = self._get_embedding(query)
        
        # Compute similarities
        similarities = []
        for entry in self.entries:
            if entry.embedding is None:
                continue
            
            # Cosine similarity
            dot_product = np.dot(query_embedding, entry.embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_entry = np.linalg.norm(entry.embedding)
            
            if norm_query > 0 and norm_entry > 0:
                similarity = dot_product / (norm_query * norm_entry)
            else:
                similarity = 0.0
            
            similarities.append((similarity, entry))
        
        # Sort by similarity and return top-K
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in similarities[:k]]
    
    def get_relevant_context(self, query: str, k: int = 1) -> str:
        """
        Retrieve relevant history and return DSL-formatted context string.
        Format: CXT:context_text | (DSL compatible)
        Returns empty string if no relevant context found.
        """
        relevant = self.retrieve_relevant(query, k=k)
        
        if not relevant:
            return ""
        
        # Extract context from most relevant entry
        entry = relevant[0]
        
        # Extract key information from response
        context_parts = []
        
        if entry.response.startswith('S:'):
            # Speech response - extract the actual text
            speech_text = entry.response[2:].strip()
            # Limit length to keep it compact
            if len(speech_text) > 80:
                speech_text = speech_text[:80] + "..."
            context_parts.append(speech_text)
        
        elif entry.response.startswith('T:'):
            # Tool response - include tool info
            tool_info = entry.response[2:].strip()
            if len(tool_info) > 50:
                tool_info = tool_info[:50] + "..."
            context_parts.append(f"used {tool_info}")
        
        # Also include prompt topic if relevant
        prompt_lower = entry.prompt.lower()
        if any(word in prompt_lower for word in ['weather', 'math', 'time', 'date']):
            # Extract topic
            for word in ['weather', 'math', 'time', 'date']:
                if word in prompt_lower:
                    context_parts.append(f"about {word}")
                    break
        
        if context_parts:
            context_text = " ".join(context_parts)
            # Format as DSL: CXT:context | 
            return f"CXT:{context_text} | "
        
        return ""
    
    def save(self):
        """Save history to file (JSON for dev, Pickle for prod)"""
        try:
            data = {
                'entries': [entry.to_dict() for entry in self.entries],
                'config': {
                    'max_entries': self.max_entries,
                    'summary_threshold': self.summary_threshold,
                    'embedding_dim': self.embedding_dim,
                    'retrieval_k': self.retrieval_k,
                    'dev_mode': self.dev_mode
                }
            }
            
            if self.dev_mode:
                # JSON format (readable, slower)
                with open(self.storage_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            else:
                # Pickle format (binary, faster)
                with open(self.storage_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to save history: {e}")
    
    def load(self):
        """Load history from file (auto-detect format)"""
        try:
            if not os.path.exists(self.storage_file):
                return
            
            # Auto-detect format based on file extension
            if self.storage_file.endswith('.json'):
                # JSON format
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.dev_mode = True
            elif self.storage_file.endswith('.pkl'):
                # Pickle format
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                self.dev_mode = False
            else:
                # Try both formats (for migration)
                try:
                    with open(self.storage_file, 'rb') as f:
                        data = pickle.load(f)
                    self.dev_mode = False
                except:
                    with open(self.storage_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.dev_mode = True
            
            # Load entries
            self.entries = [HistoryEntry.from_dict(e) for e in data.get('entries', [])]
            
            # Update config if present
            if 'config' in data:
                config = data['config']
                self.max_entries = config.get('max_entries', self.max_entries)
                self.summary_threshold = config.get('summary_threshold', self.summary_threshold)
                self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
                self.retrieval_k = config.get('retrieval_k', self.retrieval_k)
                # Update dev_mode from config if present
                if 'dev_mode' in config:
                    self.dev_mode = config.get('dev_mode', self.dev_mode)
        except Exception as e:
            print(f"Warning: Failed to load history: {e}")
            self.entries = []
    
    def clear(self):
        """Clear all history"""
        self.entries = []
        self.save()
    
    def export_json(self, output_file: str = "history_export.json"):
        """Export history to JSON format for debugging (works in both dev and prod mode)"""
        try:
            data = {
                'entries': [entry.to_dict() for entry in self.entries],
                'config': {
                    'max_entries': self.max_entries,
                    'summary_threshold': self.summary_threshold,
                    'embedding_dim': self.embedding_dim,
                    'retrieval_k': self.retrieval_k,
                    'dev_mode': self.dev_mode
                }
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"History exported to {output_file}")
        except Exception as e:
            print(f"Warning: Failed to export history: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about stored history"""
        total_entries = len(self.entries)
        summaries = sum(1 for e in self.entries if e.is_summary)
        regular = total_entries - summaries
        
        return {
            'total_entries': total_entries,
            'summaries': summaries,
            'regular_entries': regular,
            'max_entries': self.max_entries,
            'memory_usage_mb': (total_entries * self.embedding_dim * 2) / (1024 * 1024),  # float16 = 2 bytes
            'storage_mode': 'JSON (dev)' if self.dev_mode else 'Pickle (prod)',
            'storage_file': self.storage_file
        }

