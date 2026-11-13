# ============================================================================
# INSTALLATION & IMPORTS
# ============================================================================

# Install required packages (run once)
import subprocess
import sys

def install_packages():
    packages = [
        'transformers',
        'datasets',
        'torch',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'tqdm',
        'scikit-learn',
        'sentencepiece'
    ]
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

try:
    import torch
except ImportError:
    print("Installing required packages...")
    install_packages()

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from tqdm.auto import tqdm
import random
from copy import deepcopy
from typing import List, Dict, Tuple, Optional
import warnings
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import time

from curriculum_dacl import (
    SelfPacedDifficultyEstimator,
    CompetenceScheduler,
    AlignmentGate,
    spearman_drift,
)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration matching paper specifications exactly"""
    
    # Model (Section 4.1)
    model_name = 'microsoft/mdeberta-v3-base'
    num_labels = 3  # NLI: entailment, neutral, contradiction
    max_length = 128
    
    # Training mode
    debug_mode = False  # Set True for quick testing
    debug_samples = 50000  # Samples for debug mode
    
    # Warmup phase (Appendix A.1)
    warmup_epochs = 4
    warmup_batch_size = 32  # Memory-optimized batch size
    warmup_lr = 2e-5
    warmup_gradient_accumulation = 2  # Use gradient accumulation
    # Pipeline-level ablation only (Sec. results-ablation-pipeline): when True,
    # run_complete_experiment() skips warmup_finetune() entirely and
    # meta-training starts directly from the pretrained backbone + a
    # randomly-initialized classification head. Default False everywhere else.
    skip_warmup = False

    # NOTE: a discriminative LR (head_lr_multiplier > 1) combined with a
    # head-only probing/freeze phase previously caused training to DIVERGE
    # (head LR of 2e-4 blew the loss up to 167, gradients pinned at the clip
    # ceiling) and the model collapsed to predicting a single constant class
    # (accuracy frozen at exactly 1/3). mDeBERTa-v3-base fine-tunes stably on
    # XNLI with a single learning rate around 2e-5; no head boost is needed.
    # Kept at 1 (no-op) so any code path that still multiplies by it is safe.
    head_lr_multiplier = 1
    warmup_clip_max_norm = 1.0
    # Same rationale as head_lr_multiplier above: kept at 1 (no-op) for the
    # MAML inner-loop manual SGD update.
    inner_lr_head_multiplier = 1
    # Diagnostic: print head vs. backbone vs. total-unclipped grad norms during
    # warmup (and reusable for meta-training). Set False once convergence is
    # confirmed to remove the extra one-batch forward/backward passes.
    debug_grad_norms = True
    # NEW: print predicted-class distribution during evaluation so a collapse
    # (model predicting a single constant class for every input) is caught
    # immediately instead of just observing a frozen loss/accuracy number.
    debug_prediction_distribution = True

    # Meta-learning (Appendix A.1)
    meta_episodes = 400  # 400 in paper
    # inner_steps=3 / inner_lr=1e-5 produced near-zero post-adaptation query
    # loss (0.0001-0.0005 throughout 400 episodes): 5 steps at 1e-5 barely
    # moves the fast weights off an already-strong warmup model, so the
    # meta-gradient carried almost no signal. inner_lr=1e-3 is the standard
    # MAML inner-LR range and, over only 5 steps, is safe (first-order MAML,
    # fast weights are discarded after each task).
    inner_steps = 5
    inner_lr = 1e-3
    outer_lr = 1e-5  # meta/outer LR stays small
    # k_shot (support) and query_size (query) are now explicit and separate
    # (previously k_shot*2 was split evenly). Larger + separate sizes make
    # each task less trivial for an already-strong warmup model.
    k_shot = 16
    query_size = 16
    tasks_per_episode = 4  # average over several tasks per meta-update for a stabler signal
    # Stratified (label-balanced) support/query draw -- prevents the inner
    # loop from over-adapting to a class-skewed sample, which was driving
    # the neutral-class prediction drift observed during meta-training.
    stratified_sampling = True
    # 'hard_first': invert the competence gate so an already-strong model is
    # trained on the windows it currently gets wrong (informative gradient)
    # instead of the easiest windows first (near-zero gradient). 'easy_first'
    # keeps the original Platanios-style ordering for ablation.
    curriculum_direction = 'hard_first'
    # Track the best avg zero-shot accuracy seen during periodic meta-eval
    # and report/checkpoint from that model rather than the last episode's
    # weights, so a late-episode wobble can't determine the reported number.
    select_best_meta_checkpoint = True

    # Curriculum learning (Section 3.2) - baseline: static multi-teacher cross-review
    n_teachers = 3 if debug_mode else 5  # Reduced from 10 to 5 for memory
    n_buckets = 10 if debug_mode else 20  # 20 in paper (m=20)
    window_step = 3 if debug_mode else 5  # Window size (t=5)
    teacher_epochs = 2  # Reduced from 3 to 2 for memory
    teacher_batch_size = 16  # Batch size used when training/evaluating teachers

    # ------------------------------------------------------------------
    # DACL (novelty) settings: Dynamic Alignment-gated Curriculum Learning
    # See NOVELTY.md for the full method description and equations.
    # ------------------------------------------------------------------
    curriculum_mode = 'dacl_dynamic'   # 'cross_review' | 'dacl_static' | 'dacl_dynamic' | 'none'
    window_scheduling = 'competence'   # 'random' | 'competence'
    alignment_gating = True            # gate meta-updates by probe-gradient alignment

    # Component 1: self-paced difficulty estimation
    lambda_loss = 1.0                  # weight on z-normalized CE loss term
    lambda_ent = 0.5                   # weight on z-normalized predictive entropy term
    rescore_every_n_episodes = 50      # re-estimate difficulty every N episodes (dynamic mode)

    # Component 2: competence-based window scheduling
    initial_competence = 0.1           # c0 in the competence formula

    # Component 3: alignment-gated meta-updates
    probe_size = 64                    # fixed probe set size, sampled once from en_val
    probe_refresh_every = 10           # recompute cached probe gradient every N episodes
    gate_min = 0.1                     # minimum gate value (never fully zero an update)

    # All 15 languages from XNLI (Table 1 & Table 2)
    all_languages = ['en', 'fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 
                     'ar', 'vi', 'th', 'zh', 'hi', 'sw', 'ur']
    
    # Paper setup: en for warmup, ru+el auxiliary, rest for zero-shot
    high_resource_lang = 'en'
    aux_langs = ['ru', 'el']
    target_langs = ['fr', 'es', 'de', 'bg', 'tr', 'ar', 'vi', 'th', 'zh', 'hi', 'sw', 'ur']
    
    # Evaluation
    eval_batch_size = 8  # Memory-optimized eval batch size
    
    # System
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpointing & Logging
    checkpoint_dir = './checkpoints'
    results_dir = './results'
    plots_dir = './plots'
    save_every_n_episodes = 50  # Save checkpoint every N episodes
    
    # Reproducibility seeds for 5 runs (Table 1)
    seeds = [42, 26, 11, 2000, 10]

config = Config()

# Create directories
for dir_path in [config.checkpoint_dir, config.results_dir, config.plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(config.seed)

def get_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json(data, filepath):
    """Save data as JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Saved: {filepath}")

def load_json(filepath):
    """Load JSON data"""
    with open(filepath, 'r') as f:
        return json.load(f)

# ============================================================================
# DATA LOADING
# ============================================================================

class DatasetManager:
    """Manage XNLI dataset loading for all 15 languages"""
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        
    def load_all_datasets(self):
        """Load all XNLI datasets as per paper"""
        print("\n" + "="*70)
        print("LOADING XNLI DATASETS (15 LANGUAGES)")
        print("="*70)
        
        # English: train + validation (for warmup)
        print(f"\n[High-Resource Language]")
        self.datasets['en_train'] = self._load_dataset('en', 'train')
        self.datasets['en_val'] = self._load_dataset('en', 'validation')
        
        # Auxiliary languages: validation sets (for meta-training)
        print(f"\n[Auxiliary Languages - for meta-training]")
        for lang in self.config.aux_langs:
            self.datasets[f'{lang}_val'] = self._load_dataset(lang, 'validation')
        
        # All languages: test sets (for zero-shot evaluation)
        print(f"\n[All Languages - test sets for evaluation]")
        for lang in self.config.all_languages:
            self.datasets[f'{lang}_test'] = self._load_dataset(lang, 'test')
        
        print(f"\n✓ Total datasets loaded: {len(self.datasets)}")
        return self.datasets
    
    def _load_dataset(self, language, split):
        """Load single dataset"""
        try:
            # Remove trust_remote_code parameter
            dataset = load_dataset('facebook/xnli', language, split=split)
            
            # Debug mode: use subset
            if self.config.debug_mode and len(dataset) > self.config.debug_samples:
                indices = random.sample(range(len(dataset)), self.config.debug_samples)
                dataset = dataset.select(indices)
            
            print(f"  ✓ {language} ({split}): {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            print(f"  ✗ Failed to load {language} ({split}): {e}")
            return None

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class DataProcessor:
    """Process XNLI data for model input"""
    
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def prepare_batch(self, examples):
        """Convert examples to model inputs"""
        if isinstance(examples, dict):
            premises = examples['premise']
            hypotheses = examples['hypothesis']
            labels = examples['label']
        else:
            premises = [ex['premise'] for ex in examples]
            hypotheses = [ex['hypothesis'] for ex in examples]
            labels = [ex['label'] for ex in examples]
        
        encoded = self.tokenizer(
            premises,
            hypotheses,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoded['labels'] = torch.tensor(labels)
        return encoded

# ============================================================================
# MODEL
# ============================================================================

class MetaNLIModel(nn.Module):
    """mDeBERTa model for NLI"""
    
    def __init__(self, model_name=config.model_name, num_labels=config.num_labels):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            dtype=torch.float32,  # mDeBERTa's checkpoint config stores torch_dtype=float16;
                                   # newer transformers defaults to that and mDeBERTa's
                                   # disentangled attention overflows in fp16, collapsing
                                   # training to a constant-class prediction. Force fp32.
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# ============================================================================
# DISCRIMINATIVE-LR / GRAD-NORM HELPERS
# ============================================================================

def is_head_param_name(name):
    """A parameter belongs to the (randomly-initialized) classification head if
    its name contains 'classifier' or 'pooler'; everything else is the
    pretrained backbone. Matches the convention used by
    DebertaV2ForSequenceClassification (and AlignmentGate in curriculum_dacl)."""
    return 'classifier' in name or 'pooler' in name


def split_head_backbone_params(model):
    """Split model.named_parameters() into (backbone_params, head_params).

    head_params: the freshly-initialized classifier/pooler head.
    backbone_params: the pretrained mDeBERTa-v3 (the vast majority of ~280M).
    """
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if is_head_param_name(name):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def print_param_split_sanity(model, tag=""):
    """Sanity-check the head/backbone split; if head group is empty, dump the
    parameter name prefixes so the string match can be corrected."""
    backbone_params, head_params = split_head_backbone_params(model)
    head_count = sum(p.numel() for p in head_params)
    backbone_count = sum(p.numel() for p in backbone_params)
    label = f" {tag}" if tag else ""
    print(f"[param-split{label}] backbone: {backbone_count:,} params in "
          f"{len(backbone_params)} tensors | head: {head_count:,} params in "
          f"{len(head_params)} tensors")
    if len(head_params) == 0:
        prefixes = sorted({'.'.join(name.split('.')[:3])
                           for name, _ in model.named_parameters()})
        print("[param-split] WARNING: head group is EMPTY. "
              "Parameter name prefixes are:")
        for pfx in prefixes:
            print(f"    {pfx}")
        raise RuntimeError(
            "Head/backbone parameter split matched no head params; "
            "fix is_head_param_name() using the prefixes printed above.")
    return backbone_params, head_params


def print_grad_norm_diagnostics(model, tag=""):
    """Print head vs. backbone grad norms and the total UNCLIPPED grad norm.

    Assumes .backward() has already populated .grad. The total unclipped norm is
    obtained via clip_grad_norm_(..., max_norm=1e9): the huge threshold means no
    real rescaling happens, but the function still returns the pre-clip total.

    NOTE: if the backbone is currently frozen (requires_grad=False), its
    parameters have .grad == None and are correctly excluded (backbone_norm
    will show as 0.0 during the freeze window -- this is expected)."""
    head_sq, backbone_sq = 0.0, 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.detach().norm().item()
        if is_head_param_name(name):
            head_sq += n * n
        else:
            backbone_sq += n * n
    head_norm = head_sq ** 0.5
    backbone_norm = backbone_sq ** 0.5
    total_unclipped = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9))
    label = f" {tag}" if tag else ""
    print(f"[grad-norm{label}] head={head_norm:.4e} | "
          f"backbone={backbone_norm:.4e} | total_unclipped={total_unclipped:.4e}")


# ============================================================================
# CURRICULUM LEARNING
# ============================================================================

class CurriculumGenerator:
    """
    Implements curriculum learning with:
    1. Cross-review evaluation (Section 3.2.1)
    2. Annealing arrangement (Section 3.2.2)
    """
    
    def __init__(self, n_teachers=10, n_buckets=20, window_step=5, teacher_epochs=3,
                 teacher_lr=2e-5, head_lr_multiplier=1):
        self.n_teachers = n_teachers
        self.n_buckets = n_buckets
        self.window_step = window_step
        self.teacher_epochs = teacher_epochs
        self.teacher_lr = teacher_lr
        # NOTE: teachers previously used a discriminative LR (head_lr_multiplier)
        # but that caused divergence (see Config.head_lr_multiplier). Unused now;
        # kept as a constructor arg so callers don't break.
        self.head_lr_multiplier = head_lr_multiplier
    
    def cross_review_evaluate(self, dataset, tokenizer, device):
        """
        Section 3.2.1: Cross-Review Difficulty Evaluation
        Returns difficulty scores for each sample
        MEMORY OPTIMIZED: Process teachers one at a time
        """
        print(f"\n[Curriculum] Cross-review with {self.n_teachers} teachers...")
        
        dataset_size = len(dataset)
        subset_size = dataset_size // self.n_teachers
        
        # Split dataset into n subsets
        indices = list(range(dataset_size))
        random.shuffle(indices)
        
        subsets = []
        for i in range(self.n_teachers):
            start_idx = i * subset_size
            end_idx = start_idx + subset_size if i < self.n_teachers - 1 else dataset_size
            subset_indices = indices[start_idx:end_idx]
            subsets.append((dataset.select(subset_indices), subset_indices))
        
        processor = DataProcessor(tokenizer)
        
        # Initialize score storage
        all_scores = {i: [] for i in range(dataset_size)}
        
        # Train and evaluate teachers ONE AT A TIME to save memory
        for teacher_idx in tqdm(range(self.n_teachers), desc="Training teachers"):
            print(f"\n[Teacher {teacher_idx+1}/{self.n_teachers}]")
            
            # Create and train this teacher (discriminative LR: fresh head needs
            # a larger effective LR than the pretrained backbone)
            teacher = MetaNLIModel().to(device)
            teacher.train()
            optimizer = AdamW(teacher.parameters(), lr=self.teacher_lr, eps=1e-6)
            
            train_subset = subsets[teacher_idx][0]
            
            # Train teacher on its subset
            for epoch in range(self.teacher_epochs):
                for batch_start in range(0, len(train_subset), 16):
                    batch_end = min(batch_start + 16, len(train_subset))
                    batch_data = train_subset.select(range(batch_start, batch_end))
                    batch = processor.prepare_batch(batch_data)
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    optimizer.zero_grad()
                    logits = teacher(input_ids, attention_mask)
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Clear cache periodically
                    if batch_start % 100 == 0:
                        torch.cuda.empty_cache()
            
            teacher.eval()
            
            # Cross-review: evaluate all OTHER subsets with this teacher
            print(f"  Evaluating other subsets...")
            for eval_idx, (eval_subset, eval_indices) in enumerate(subsets):
                if eval_idx != teacher_idx:  # Don't evaluate own subset
                    with torch.no_grad():
                        for batch_start in range(0, len(eval_subset), 16):
                            batch_end = min(batch_start + 16, len(eval_subset))
                            batch_data = eval_subset.select(range(batch_start, batch_end))
                            batch = processor.prepare_batch(batch_data)
                            
                            input_ids = batch['input_ids'].to(device)
                            attention_mask = batch['attention_mask'].to(device)
                            labels = batch['labels'].to(device)
                            
                            logits = teacher(input_ids, attention_mask)
                            losses = F.cross_entropy(logits, labels, reduction='none')
                            
                            # Store scores for each sample
                            for local_idx, loss_val in enumerate(losses.cpu().numpy()):
                                global_idx = eval_indices[batch_start + local_idx]
                                all_scores[global_idx].append(float(loss_val))
            
            # Delete teacher and free memory before next iteration
            del teacher
            del optimizer
            torch.cuda.empty_cache()
            print(f"  ✓ Teacher {teacher_idx+1} completed and memory freed")
        
        # Average scores across teachers for each sample
        print("\n[Curriculum] Computing final difficulty scores...")
        final_scores = []
        for i in range(dataset_size):
            if all_scores[i]:  # If sample was evaluated
                final_scores.append(np.mean(all_scores[i]))
            else:  # Fallback for samples not evaluated (shouldn't happen)
                final_scores.append(1.0)
        
        return final_scores
    
    def annealing_arrangement(self, dataset, difficulty_scores):
        """
        Section 3.2.2: Annealing Arrangement
        Returns list of annealed batch indices
        """
        print(f"[Curriculum] Annealing: {self.n_buckets} buckets, window={self.window_step}")
        
        # Sort indices by difficulty
        sorted_indices = np.argsort(difficulty_scores)
        
        # Organize into buckets
        bucket_size = len(sorted_indices) // self.n_buckets
        buckets = []
        
        for i in range(self.n_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < self.n_buckets - 1 else len(sorted_indices)
            buckets.append(sorted_indices[start:end].tolist())
        
        # Generate annealed batches with sliding window
        annealed_batches = []
        
        for i in range(self.n_buckets - self.window_step + 1):
            window_buckets = buckets[i:i + self.window_step]
            
            # Flatten all indices in current window
            window_indices = []
            for bucket in window_buckets:
                window_indices.extend(bucket)
            
            annealed_batches.append(window_indices)
        
        print(f"[Curriculum] Generated {len(annealed_batches)} annealed windows")
        return annealed_batches, buckets

# ============================================================================
# MAML IMPLEMENTATION
# ============================================================================

class MAML:
    """Model-Agnostic Meta-Learning - Memory Optimized Version"""
    
    def __init__(self, model, inner_lr=1e-5, outer_lr=1e-5, inner_steps=3,
                 head_lr_multiplier=1, inner_lr_head_multiplier=1,
                 clip_max_norm=1.0):
        self.model = model
        self.inner_lr = inner_lr
        # NOTE: discriminative LR (head_lr_multiplier / inner_lr_head_multiplier
        # > 1) previously caused training divergence and a collapse to a single
        # constant prediction (see Config.head_lr_multiplier). Both default to
        # 1 (no-op) now; kept as constructor args so callers don't break.
        self.inner_lr_head_multiplier = inner_lr_head_multiplier
        self.inner_steps = inner_steps
        self.clip_max_norm = clip_max_norm
        # Single LR for the outer/meta optimizer -- no head param-group boost.
        self.meta_optimizer = AdamW(model.parameters(), lr=outer_lr, eps=1e-6)
        self.use_first_order = True  # Use first-order MAML to save memory
    
    def inner_loop(self, support_batch, query_batch, device):
        """Inner loop: adapt on support set, evaluate on query set
        MEMORY OPTIMIZED: Using first-order MAML approximation

        Returns (post_adaptation_query_loss, pre_adaptation_query_loss). The
        pre-adaptation number is a diagnostic: if post is barely below pre,
        the task's support set carried no learning signal for the inner loop
        (e.g. because the model already solves it), and the resulting
        meta-gradient will be tiny/noisy."""

        # Clone model parameters
        fast_weights = {}
        for name, param in self.model.named_parameters():
            fast_weights[name] = param.clone()

        q_input_ids = query_batch['input_ids'].to(device)
        q_attention_mask = query_batch['attention_mask'].to(device)
        q_labels = query_batch['labels'].to(device)

        # Pre-adaptation query loss: current meta-parameters, before any
        # inner-loop step. Diagnostic only -- no grad needed.
        with torch.no_grad():
            pre_logits = self.model(q_input_ids, q_attention_mask)
            pre_query_loss = F.cross_entropy(pre_logits, q_labels).item()
            del pre_logits

        # Inner adaptation on support set (without computation graph)
        for step in range(self.inner_steps):
            input_ids = support_batch['input_ids'].to(device)
            attention_mask = support_batch['attention_mask'].to(device)
            labels = support_batch['labels'].to(device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            # Compute gradients (no create_graph to save memory)
            grads = torch.autograd.grad(
                loss, 
                self.model.parameters(), 
                create_graph=False,  # First-order approximation
                retain_graph=False
            )
            
            # Manual SGD update with detached gradients; head params get a
            # larger step so the fresh head can actually move.
            with torch.no_grad():
                for (name, param), grad in zip(self.model.named_parameters(), grads):
                    lr = self.inner_lr * (
                        self.inner_lr_head_multiplier if is_head_param_name(name) else 1.0)
                    param.data = param.data - lr * grad.detach()
                    
            # Clear intermediate tensors
            del loss, logits, input_ids, attention_mask, labels, grads
            torch.cuda.empty_cache()
        
        # Evaluate on query set with the adapted (fast) weights -- this
        # creates the graph for the meta-update.
        logits = self.model(q_input_ids, q_attention_mask)
        query_loss = F.cross_entropy(logits, q_labels)

        # Restore original parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = fast_weights[name].data

        # Clean up
        del fast_weights, q_input_ids, q_attention_mask, q_labels

        return query_loss, pre_query_loss

    def meta_update(self, task_batch, device, alignment_gate=None):
        """Outer loop: meta-update across multiple tasks
        MEMORY OPTIMIZED: Process tasks one at a time

        If `alignment_gate` (curriculum_dacl.AlignmentGate) is provided, the
        accumulated gradient is scaled by its alignment with a cached probe
        gradient before the optimizer step (Component 3 of DACL)."""

        self.meta_optimizer.zero_grad()

        # Accumulate gradients from each task
        total_loss = 0
        total_pre_loss = 0
        for task_idx, (support_batch, query_batch) in enumerate(task_batch):
            task_loss, pre_loss = self.inner_loop(support_batch, query_batch, device)

            # Normalize by number of tasks for gradient accumulation
            normalized_loss = task_loss / len(task_batch)
            normalized_loss.backward()

            total_loss += task_loss.item()
            total_pre_loss += pre_loss

            # Clear task-specific tensors
            del task_loss, normalized_loss
            if task_idx % 1 == 0:  # Clear cache after each task
                torch.cuda.empty_cache()

        gate_value, cos_sim = 1.0, None
        if alignment_gate is not None:
            task_head_grad = alignment_gate.get_task_head_grad_flat()
            if task_head_grad is not None:
                gate_value, cos_sim = alignment_gate.compute_gate(task_head_grad)
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(gate_value)
                del task_head_grad

        # Clip gradients and update (relaxed clip so the head grad survives)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_max_norm)
        self.meta_optimizer.step()

        avg_loss = total_loss / len(task_batch)
        avg_pre_loss = total_pre_loss / len(task_batch)
        return avg_loss, avg_pre_loss, gate_value, cos_sim

# ============================================================================
# STRATIFIED TASK SAMPLING
# ============================================================================

def stratified_task_sample(labels_array, window_indices, k_shot, query_size, rng, num_labels=3):
    """Draw k_shot support + query_size query indices from window_indices,
    balanced as evenly as possible across `num_labels` classes.

    Without this, a uniform random draw of a small (k_shot=10-16) support
    set has no guarantee of balanced labels; the MAML inner loop then
    over-adapts toward whichever class happens to dominate that draw, which
    is what was driving the neutral-class prediction drift observed during
    meta-training. `rng` just needs .shuffle/.choice (the `random` module or
    a random.Random instance both work).
    """
    by_label = {c: [] for c in range(num_labels)}
    for idx in window_indices:
        label = int(labels_array[idx])
        if label in by_label:
            by_label[label].append(idx)
    for c in by_label:
        rng.shuffle(by_label[c])

    total_needed = k_shot + query_size
    per_class = total_needed // num_labels
    remainder = total_needed - per_class * num_labels

    pooled, leftover = [], []
    for c in range(num_labels):
        take = per_class + (1 if c < remainder else 0)
        pooled.extend(by_label[c][:take])
        leftover.extend(by_label[c][take:])

    if len(pooled) < total_needed:
        rng.shuffle(leftover)
        pooled.extend(leftover[:total_needed - len(pooled)])

    if len(pooled) < total_needed:
        # Window smaller than requested total (e.g. a very early, narrow
        # curriculum window) -- fall back to sampling with replacement so
        # support/query shapes stay fixed.
        pooled = [rng.choice(window_indices) for _ in range(total_needed)]
    else:
        rng.shuffle(pooled)

    support_indices = pooled[:k_shot]
    query_indices = pooled[k_shot:k_shot + query_size]
    return support_indices, query_indices

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class CMLTrainer:
    """Complete Curriculum Meta-Learning Training Pipeline"""
    
    def __init__(self, config, dataset_manager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.device = torch.device(config.device)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = MetaNLIModel().to(self.device)
        self.processor = DataProcessor(self.tokenizer, config.max_length)
        
        # Training state
        self.training_history = {
            'warmup_losses': [],
            'warmup_val_accs': [],
            'meta_losses': [],
            'meta_query_losses': [],  # per-episode {episode, pre_adapt, post_adapt}
            'episode_eval_results': {},
            # DACL logging
            'curriculum_schedule': [],
            'alignment_stats': [],
            'difficulty_drift': [],
            'compute_cost': {}
        }
        self.best_val_acc = 0.0
        # Best avg zero-shot accuracy seen during periodic meta-eval; used
        # to select best_meta_model.pt so a late-episode wobble (the
        # mid-run degradation MAML can cause) doesn't determine the
        # reported final number.
        self.best_meta_avg_acc = -1.0

        print(f"✓ Initialized trainer on device: {self.device}")

    def _sanity_check_data(self, dataset, n=2):
        """Decode a couple of raw samples to rule out a data/label
        misalignment bug before spending time training. Run once."""
        print("\n[Sanity Check] Sample training examples:")
        for i in range(min(n, len(dataset))):
            ex = dataset[i]
            print(f"  premise='{ex['premise'][:80]}' | "
                  f"hypothesis='{ex['hypothesis'][:80]}' | label={ex['label']}")

    def _run_grad_norm_diagnostic(self, dataset, tag=""):
        """One forward+backward pass on a single batch to inspect head vs.
        backbone gradient magnitudes. Does NOT step the optimizer; clears grads
        afterwards so it has no effect on training. Gated by debug_grad_norms."""
        was_training = self.model.training
        self.model.train()
        n = min(self.config.warmup_batch_size, len(dataset))
        batch_data = dataset.select(range(n))
        batch = self.processor.prepare_batch(batch_data)

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        self.model.zero_grad()
        logits = self.model(input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        print(f"[grad-norm {tag}] loss={loss.item():.4f}")
        print_grad_norm_diagnostics(self.model, tag=tag)
        self.model.zero_grad()

        del logits, loss, input_ids, attention_mask, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not was_training:
            self.model.eval()

    def warmup_finetune(self):
        """Phase 1: Warmup fine-tuning on English"""
        print("\n" + "="*70)
        print("PHASE 1: WARMUP FINE-TUNING (English)")
        print("="*70)

        train_dataset = self.dataset_manager.datasets['en_train']
        val_dataset = self.dataset_manager.datasets['en_val']

        self._sanity_check_data(train_dataset)

        self.model.train()

        if self.config.debug_grad_norms:
            print_param_split_sanity(self.model, tag="warmup")
            self._run_grad_norm_diagnostic(train_dataset, tag="warmup step 0 (pre-train)")

        # Plain fine-tune: the whole model (backbone + head) trains from step
        # 0 with a single learning rate. eps=1e-6 (not the 1e-8 default) is
        # the recommended AdamW epsilon for DeBERTa-v3 -- the smaller default
        # underflows and destabilizes updates.
        optimizer = AdamW(self.model.parameters(), lr=self.config.warmup_lr, eps=1e-6)

        total_steps = (len(train_dataset) // self.config.warmup_batch_size) * self.config.warmup_epochs
        # mDeBERTa-v3 is unstable without LR warmup; use a 10% warmup ratio.
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        global_step = 0
        for epoch in range(self.config.warmup_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.warmup_epochs}")
            
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)
            
            epoch_loss = 0
            steps = 0
            
            pbar = tqdm(range(0, len(indices), self.config.warmup_batch_size), desc="Training")
            for i in pbar:
                batch_indices = indices[i:i+self.config.warmup_batch_size]
                batch_data = train_dataset.select(batch_indices)
                batch = self.processor.prepare_batch(batch_data)
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)

                # Guard against a single bad batch poisoning all weights with NaN.
                if not torch.isfinite(loss):
                    print(f"⚠️  Non-finite loss ({loss.item()}) at step {steps}; skipping batch")
                    optimizer.zero_grad()
                    continue

                loss.backward()

                # Diagnostic checkpoints early in training.
                if self.config.debug_grad_norms and global_step in (1, 50, 200):
                    print_grad_norm_diagnostics(self.model, tag=f"warmup step {global_step}")

                # Gradient clipping is essential for mDeBERTa-v3 stability — without
                # it, one exploding gradient turns the weights to NaN permanently.
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.warmup_clip_max_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                steps += 1
                global_step += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = epoch_loss / steps
            self.training_history['warmup_losses'].append(avg_loss)
            
            # Validation
            val_acc = self.evaluate(val_dataset, tag=f"epoch {epoch+1}")
            self.training_history['warmup_val_accs'].append(val_acc)
            
            print(f"Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_warmup_model.pt', phase='warmup', epoch=epoch+1, val_acc=val_acc)
        
        print(f"\n✓ Warmup completed! Best val acc: {self.best_val_acc:.4f}")

    def _generate_curriculum_cross_review(self, curriculum, compute_cost):
        """Baseline curriculum: static multi-teacher cross-review (Doan & Le)."""
        annealed_tasks = {}
        for lang in self.config.aux_langs:
            print(f"\n[{lang.upper()}] Generating curriculum (cross_review)...")
            dataset = self.dataset_manager.datasets[f'{lang}_val']

            t0 = time.time()
            difficulty_scores = curriculum.cross_review_evaluate(dataset, self.tokenizer, self.device)
            elapsed = time.time() - t0

            # n_teachers models trained (teacher_epochs full passes each) plus
            # cross-evaluation of the other (n_teachers - 1) subsets per teacher.
            n = self.config.n_teachers
            subset_size = len(dataset) // n
            steps_per_epoch = max(1, subset_size // self.config.teacher_batch_size)
            train_passes = n * self.config.teacher_epochs * steps_per_epoch
            eval_passes = n * (n - 1) * steps_per_epoch

            annealed_batches, buckets = curriculum.annealing_arrangement(dataset, difficulty_scores)
            annealed_tasks[lang] = {
                'dataset': dataset,
                'annealed_batches': annealed_batches,
                'difficulty_scores': difficulty_scores,
                'buckets': buckets,
                # Precompute a plain label array once (one columnar Arrow
                # access) so stratified_task_sample doesn't row-index the
                # HF Dataset per-episode, per-example.
                'labels': np.asarray(dataset['label'])
            }

            compute_cost.setdefault('per_language', {})[lang] = {
                'mode': 'cross_review',
                'wall_clock_seconds': elapsed,
                'forward_backward_passes': train_passes + eval_passes
            }
            torch.cuda.empty_cache()
        return annealed_tasks

    def _generate_curriculum_dacl(self, curriculum, estimator, compute_cost):
        """DACL curriculum: teacher-free self-paced difficulty scoring."""
        annealed_tasks = {}
        for lang in self.config.aux_langs:
            print(f"\n[{lang.upper()}] Generating curriculum ({self.config.curriculum_mode})...")
            dataset = self.dataset_manager.datasets[f'{lang}_val']

            t0 = time.time()
            passes_before = estimator.forward_pass_count
            difficulty_scores = estimator.score(self.model, dataset, self.processor, self.device)
            elapsed = time.time() - t0
            passes = estimator.forward_pass_count - passes_before

            annealed_batches, buckets = curriculum.annealing_arrangement(dataset, difficulty_scores)
            annealed_tasks[lang] = {
                'dataset': dataset,
                'annealed_batches': annealed_batches,
                'difficulty_scores': difficulty_scores,
                'buckets': buckets,
                'labels': np.asarray(dataset['label'])
            }

            compute_cost.setdefault('per_language', {})[lang] = {
                'mode': self.config.curriculum_mode,
                'wall_clock_seconds': elapsed,
                'forward_backward_passes': passes
            }
            torch.cuda.empty_cache()
        return annealed_tasks

    def _generate_no_curriculum(self, compute_cost):
        """Pipeline-level ablation only (Sec. results-ablation-pipeline): no
        difficulty scoring, no buckets, no windows -- each language's entire
        auxiliary validation set is exposed as a single 'window' spanning all
        of it, so downstream task sampling draws uniformly at random from the
        whole set every episode. Not a curriculum_mode a DACL/A-B-C-D config
        should ever use; distinct from 'cross_review' (config A), which still
        curricularizes via teacher difficulty, just with a different
        estimator."""
        annealed_tasks = {}
        for lang in self.config.aux_langs:
            print(f"\n[{lang.upper()}] No curriculum (curriculum_mode='none'): "
                  f"uniform random sampling over the full set...")
            dataset = self.dataset_manager.datasets[f'{lang}_val']
            annealed_tasks[lang] = {
                'dataset': dataset,
                'annealed_batches': [list(range(len(dataset)))],
                'difficulty_scores': None,
                'buckets': None,
                'labels': np.asarray(dataset['label'])
            }
            compute_cost.setdefault('per_language', {})[lang] = {
                'mode': 'none',
                'wall_clock_seconds': 0.0,
                'forward_backward_passes': 0
            }
        return annealed_tasks

    def meta_train_with_curriculum(self):
        """Phase 2: Meta-learning with curriculum (DACL or cross-review baseline)"""
        print("\n" + "="*70)
        print("PHASE 2: META-LEARNING WITH CURRICULUM")
        print(f"curriculum_mode={self.config.curriculum_mode} | "
              f"window_scheduling={self.config.window_scheduling} | "
              f"curriculum_direction={self.config.curriculum_direction} | "
              f"alignment_gating={self.config.alignment_gating}")
        print("="*70)

        # Clear cache before starting
        torch.cuda.empty_cache()

        curriculum = CurriculumGenerator(
            n_teachers=self.config.n_teachers,
            n_buckets=self.config.n_buckets,
            window_step=self.config.window_step,
            teacher_epochs=self.config.teacher_epochs,
            head_lr_multiplier=self.config.head_lr_multiplier
        )

        compute_cost = {'per_language': {}}
        estimator = SelfPacedDifficultyEstimator(
            lambda_loss=self.config.lambda_loss,
            lambda_ent=self.config.lambda_ent,
            batch_size=self.config.eval_batch_size
        )

        if self.config.curriculum_mode == 'cross_review':
            annealed_tasks = self._generate_curriculum_cross_review(curriculum, compute_cost)
        elif self.config.curriculum_mode in ('dacl_static', 'dacl_dynamic'):
            annealed_tasks = self._generate_curriculum_dacl(curriculum, estimator, compute_cost)
        elif self.config.curriculum_mode == 'none':
            annealed_tasks = self._generate_no_curriculum(compute_cost)
        else:
            raise ValueError(f"Unknown curriculum_mode: {self.config.curriculum_mode}")

        self.training_history['compute_cost'] = compute_cost

        # Save curriculum
        curriculum_path = os.path.join(self.config.checkpoint_dir, 'curriculum_data.pkl')
        with open(curriculum_path, 'wb') as f:
            pickle.dump(annealed_tasks, f)
        print(f"\n✓ Curriculum saved to {curriculum_path}")

        # Competence-based window scheduling (Component 2). Meaningless under
        # curriculum_mode='none' (a single window spans the whole dataset, so
        # there is nothing to progress through) -- forced off regardless of
        # window_scheduling in that case.
        competence_scheduler = None
        if self.config.window_scheduling == 'competence' and self.config.curriculum_mode != 'none':
            competence_scheduler = CompetenceScheduler(
                n_episodes=self.config.meta_episodes,
                initial_competence=self.config.initial_competence,
                direction=self.config.curriculum_direction
            )
        window_rng = random.Random(self.config.seed)

        # Alignment-gated meta-updates (Component 3): fixed probe set from
        # the high-resource (English) validation split, sampled once with
        # the global seed. Never touches any target/auxiliary language data.
        alignment_gate = None
        if self.config.alignment_gating:
            en_val = self.dataset_manager.datasets['en_val']
            probe_rng = random.Random(self.config.seed)
            probe_size = min(self.config.probe_size, len(en_val))
            probe_indices = probe_rng.sample(range(len(en_val)), probe_size)
            probe_dataset = en_val.select(probe_indices)
            alignment_gate = AlignmentGate(
                self.model,
                probe_dataset=probe_dataset,
                processor=self.processor,
                device=self.device,
                probe_refresh_every=self.config.probe_refresh_every,
                gate_min=self.config.gate_min
            )

        # Meta-learning with MAML
        print("\n[Meta-Learning] Starting MAML training...")
        maml = MAML(
            self.model,
            inner_lr=self.config.inner_lr,
            outer_lr=self.config.outer_lr,
            inner_steps=self.config.inner_steps,
            head_lr_multiplier=self.config.head_lr_multiplier,
            inner_lr_head_multiplier=self.config.inner_lr_head_multiplier,
            clip_max_norm=self.config.warmup_clip_max_norm
        )

        for episode in tqdm(range(self.config.meta_episodes), desc="Meta-Episodes"):
            # Component 1 (dynamic): periodically re-estimate difficulty with
            # the current meta-model and rebuild the annealed curriculum.
            if (self.config.curriculum_mode == 'dacl_dynamic' and episode > 0
                    and episode % self.config.rescore_every_n_episodes == 0):
                for lang in self.config.aux_langs:
                    task_data = annealed_tasks[lang]
                    old_scores = task_data['difficulty_scores']
                    new_scores = estimator.score(self.model, task_data['dataset'], self.processor, self.device)
                    drift = spearman_drift(old_scores, new_scores)

                    annealed_batches, buckets = curriculum.annealing_arrangement(task_data['dataset'], new_scores)
                    task_data['difficulty_scores'] = new_scores
                    task_data['annealed_batches'] = annealed_batches
                    task_data['buckets'] = buckets

                    self.training_history['difficulty_drift'].append({
                        'episode': episode,
                        'language': lang,
                        'spearman_correlation': drift
                    })
                torch.cuda.empty_cache()

            # Sample tasks from curriculum
            task_batch = []

            for _ in range(self.config.tasks_per_episode):
                # Random auxiliary language
                lang = random.choice(self.config.aux_langs)
                task_data = annealed_tasks[lang]

                # Component 2: competence-based vs. random window selection
                if competence_scheduler is not None:
                    window_indices, window_idx, competence_value = competence_scheduler.select_window(
                        task_data['annealed_batches'],
                        n_buckets=self.config.n_buckets,
                        window_step=self.config.window_step,
                        episode=episode,
                        rng=window_rng
                    )
                else:
                    window_idx = random.randrange(len(task_data['annealed_batches']))
                    window_indices = task_data['annealed_batches'][window_idx]
                    competence_value = None

                self.training_history['curriculum_schedule'].append({
                    'episode': episode,
                    'language': lang,
                    'window_idx': window_idx,
                    'competence': competence_value
                })

                # Sample k_shot support + query_size query examples from window
                if self.config.stratified_sampling:
                    support_indices, query_indices = stratified_task_sample(
                        task_data['labels'], window_indices,
                        k_shot=self.config.k_shot, query_size=self.config.query_size,
                        rng=random, num_labels=self.config.num_labels
                    )
                else:
                    total_needed = self.config.k_shot + self.config.query_size
                    if len(window_indices) >= total_needed:
                        sampled_indices = random.sample(window_indices, total_needed)
                    else:
                        sampled_indices = [random.choice(window_indices) for _ in range(total_needed)]
                    support_indices = sampled_indices[:self.config.k_shot]
                    query_indices = sampled_indices[self.config.k_shot:]

                support_data = task_data['dataset'].select(support_indices)
                query_data = task_data['dataset'].select(query_indices)

                support_batch = self.processor.prepare_batch(support_data)
                query_batch = self.processor.prepare_batch(query_data)

                task_batch.append((support_batch, query_batch))

            # Component 3: refresh cached probe gradient, then meta-update with gating
            if alignment_gate is not None:
                alignment_gate.maybe_refresh_probe_grad(episode)

            loss, pre_loss, gate_value, cos_sim = maml.meta_update(task_batch, self.device, alignment_gate=alignment_gate)
            self.training_history['meta_losses'].append(loss)
            self.training_history['meta_query_losses'].append({
                'episode': episode,
                'pre_adapt': pre_loss,
                'post_adapt': loss
            })

            if alignment_gate is not None:
                self.training_history['alignment_stats'].append({
                    'episode': episode,
                    'cosine_similarity': cos_sim,
                    'gate': gate_value,
                    'meta_loss': loss
                })

            # Clear cache periodically
            if (episode + 1) % 10 == 0:
                torch.cuda.empty_cache()

            # Periodic evaluation and checkpointing
            if (episode + 1) % self.config.save_every_n_episodes == 0:
                print(f"\n[Episode {episode+1}] Meta-loss (post-adapt query): {loss:.4f} | "
                      f"pre-adapt query loss: {pre_loss:.4f}")
                if loss < 0.01:
                    print("  ⚠️  Post-adaptation query loss is ~0 -- meta-tasks are still "
                          "trivial for this model; the meta-gradient carries little signal.")

                # Evaluate on all test sets
                eval_results = self.evaluate_all_languages()
                self.training_history['episode_eval_results'][episode+1] = eval_results

                # Save checkpoint
                self.save_checkpoint(
                    f'meta_episode_{episode+1}.pt',
                    phase='meta',
                    episode=episode+1,
                    meta_loss=loss,
                    eval_results=eval_results
                )

                # Guard against meta-training degrading the model: track the
                # best avg zero-shot accuracy seen so far and keep a
                # separate checkpoint for it, independent of whatever the
                # last episode happens to look like.
                if self.config.select_best_meta_checkpoint:
                    avg_acc = eval_results.get('average', 0.0)
                    if avg_acc > self.best_meta_avg_acc:
                        self.best_meta_avg_acc = avg_acc
                        self.save_checkpoint(
                            'best_meta_model.pt',
                            phase='meta',
                            episode=episode+1,
                            meta_loss=loss,
                            eval_results=eval_results
                        )
                        print(f"  ★ New best avg zero-shot accuracy: {avg_acc:.4f} "
                              f"(episode {episode+1}) -- saved best_meta_model.pt")

                # Clear cache after evaluation
                torch.cuda.empty_cache()

        print(f"\n✓ Meta-learning completed!")
    
    def evaluate(self, dataset, batch_size=None, tag=""):
        """Evaluate on single dataset"""
        if batch_size is None:
            batch_size = self.config.eval_batch_size
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch_data = dataset.select(range(i, min(i+batch_size, len(dataset))))
                batch = self.processor.prepare_batch(batch_data)
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        # DIAGNOSTIC: catch representation collapse (model predicting the same
        # class for every input) immediately instead of just observing a
        # frozen accuracy number with no explanation.
        if self.config.debug_prediction_distribution:
            pred_counts = np.bincount(np.array(all_preds), minlength=self.config.num_labels)
            label_counts = np.bincount(np.array(all_labels), minlength=self.config.num_labels)
            print(f"  [pred-dist{' ' + tag if tag else ''}] "
                  f"predicted={pred_counts.tolist()} | true={label_counts.tolist()}")
            if pred_counts.max() == len(all_preds):
                print("  ⚠️  COLLAPSE DETECTED: model is predicting a single "
                      "class for every example.")

        self.model.train()
        return accuracy
    
    def evaluate_all_languages(self):
        """Evaluate on all 15 languages (test sets)"""
        print("\n" + "-"*70)
        print("EVALUATING ALL LANGUAGES")
        print("-"*70)
        
        results = {}
        for lang in self.config.all_languages:
            dataset = self.dataset_manager.datasets[f'{lang}_test']
            if dataset:
                acc = self.evaluate(dataset, tag=lang)
                results[lang] = acc
                print(f"  {lang}: {acc:.4f}")
        
        avg_acc = np.mean(list(results.values()))
        results['average'] = avg_acc
        print(f"\n  Average: {avg_acc:.4f}")
        print("-"*70)
        
        return results
    
    def save_checkpoint(self, filename, **kwargs):
        """Save training checkpoint"""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'timestamp': get_timestamp(),
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filepath):
        """Load training checkpoint from absolute or relative path"""
        # Handle both absolute and relative paths
        if not os.path.isabs(filepath):
            # If relative path, look in checkpoint directory
            filepath = os.path.join(self.config.checkpoint_dir, filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # Load with weights_only=False since checkpoint contains config and history
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Checkpoint loaded: {os.path.basename(filepath)}")
        return checkpoint

# ============================================================================
# VISUALIZATION & RESULTS
# ============================================================================

class ResultsVisualizer:
    """Generate plots and result tables matching paper format"""
    
    def __init__(self, config):
        self.config = config
        sns.set_style('whitegrid')
        plt.rcParams['figure.dpi'] = 100
    
    def plot_training_curves(self, history, save_name='training_curves.png'):
        """Plot warmup and meta-training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Warmup loss
        if history['warmup_losses']:
            axes[0, 0].plot(history['warmup_losses'], marker='o')
            axes[0, 0].set_title('Warmup Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Warmup validation accuracy
        if history['warmup_val_accs']:
            axes[0, 1].plot(history['warmup_val_accs'], marker='o', color='green')
            axes[0, 1].set_title('Warmup Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True)
        
        # Meta-training loss
        if history['meta_losses']:
            axes[1, 0].plot(history['meta_losses'], alpha=0.6)
            # Smooth curve
            if len(history['meta_losses']) > 10:
                window = 10
                smoothed = pd.Series(history['meta_losses']).rolling(window=window).mean()
                axes[1, 0].plot(smoothed, linewidth=2, color='red', label='Smoothed')
                axes[1, 0].legend()
            axes[1, 0].set_title('Meta-Training Loss')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Average accuracy over episodes
        if history['episode_eval_results']:
            episodes = sorted(history['episode_eval_results'].keys())
            avg_accs = [history['episode_eval_results'][ep]['average'] for ep in episodes]
            axes[1, 1].plot(episodes, avg_accs, marker='o', color='purple')
            axes[1, 1].set_title('Zero-Shot Average Accuracy')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.plots_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_language_comparison(self, results, baseline_results=None, save_name='language_comparison.png'):
        """Plot per-language accuracy comparison (Figure 4 style)"""
        languages = [lang for lang in self.config.all_languages if lang in results]
        accuracies = [results[lang] * 100 for lang in languages]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(languages))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='CML (Ours)', color='#2E86AB')
        
        if baseline_results:
            baseline_accs = [baseline_results.get(lang, 0) * 100 for lang in languages]
            bars2 = ax.bar(x + width/2, baseline_accs, width, label='mDeBERTa Baseline', color='#A23B72')
        
        ax.set_xlabel('Languages', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Zero-Shot Cross-Lingual Transfer Performance (All 15 Languages)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.plots_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_heatmap_auxiliary_impact(self, mono_aux_results, save_name='auxiliary_impact_heatmap.png'):
        """Plot auxiliary language impact heatmap (Figure 5 style)"""
        # Create matrix: rows=target languages, cols=auxiliary languages
        target_langs = self.config.all_languages
        aux_langs = self.config.all_languages
        
        # Create accuracy matrix
        matrix = np.zeros((len(target_langs), len(aux_langs)))
        
        for i, target in enumerate(target_langs):
            for j, aux in enumerate(aux_langs):
                if target in mono_aux_results and aux in mono_aux_results[target]:
                    matrix[i, j] = mono_aux_results[target][aux] * 100
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlBu', aspect='auto', vmin=50, vmax=80)
        
        # Set ticks
        ax.set_xticks(np.arange(len(aux_langs)))
        ax.set_yticks(np.arange(len(target_langs)))
        ax.set_xticklabels(aux_langs)
        ax.set_yticklabels(target_langs)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Accuracy (%)', rotation=-90, va="bottom")
        
        # Add text annotations
        for i in range(len(target_langs)):
            for j in range(len(aux_langs)):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=7)
        
        ax.set_title('Auxiliary Language Selection Impact on Target Languages', fontsize=14)
        ax.set_xlabel('Auxiliary Language', fontsize=12)
        ax.set_ylabel('Target Language', fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.config.plots_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def create_results_table(self, results, baseline_results=None, save_name='results_table.csv'):
        """Create results table (Table 1 style)"""
        # Prepare data
        data = {
            'Language': [],
            'CML (Ours)': [],
        }
        
        if baseline_results:
            data['mDeBERTa Baseline'] = []
            data['Improvement'] = []
        
        for lang in self.config.all_languages:
            if lang in results:
                data['Language'].append(lang)
                data['CML (Ours)'].append(f"{results[lang]*100:.2f}")
                
                if baseline_results and lang in baseline_results:
                    data['mDeBERTa Baseline'].append(f"{baseline_results[lang]*100:.2f}")
                    improvement = (results[lang] - baseline_results[lang]) * 100
                    data['Improvement'].append(f"{improvement:+.2f}")
        
        # Add average
        data['Language'].append('Average')
        data['CML (Ours)'].append(f"{results.get('average', 0)*100:.2f}")
        
        if baseline_results:
            avg_baseline = np.mean([baseline_results[lang] for lang in self.config.all_languages if lang in baseline_results])
            data['mDeBERTa Baseline'].append(f"{avg_baseline*100:.2f}")
            avg_improvement = (results.get('average', 0) - avg_baseline) * 100
            data['Improvement'].append(f"{avg_improvement:+.2f}")
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        save_path = os.path.join(self.config.results_dir, save_name)
        df.to_csv(save_path, index=False)
        print(f"✓ Saved: {save_path}")
        
        # Also display
        print("\n" + "="*70)
        print("RESULTS TABLE")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)

        return df

    def plot_difficulty_drift(self, history, save_name='difficulty_drift.png'):
        """DACL: Spearman rank correlation of difficulty scores across re-scoring events.

        Low/decreasing correlation shows difficulty is non-stationary during
        meta-training, motivating the dynamic re-scoring over a static
        one-shot cross-review estimate."""
        drift = history.get('difficulty_drift', [])
        if not drift:
            print("  (skip) No difficulty_drift data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        by_lang = {}
        for entry in drift:
            by_lang.setdefault(entry['language'], []).append((entry['episode'], entry['spearman_correlation']))

        for lang, points in by_lang.items():
            points = sorted(points)
            episodes = [p[0] for p in points]
            corrs = [p[1] for p in points]
            ax.plot(episodes, corrs, marker='o', label=lang)

        ax.set_xlabel('Episode (re-scoring event)')
        ax.set_ylabel('Spearman correlation (prev vs. new difficulty ranking)')
        ax.set_title('DACL: Difficulty Ranking Drift Over Meta-Training')
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.plots_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_alignment_stats(self, history, save_name='alignment_stats.png'):
        """DACL: gate value histogram + per-episode cosine similarity/gate timeline."""
        stats = history.get('alignment_stats', [])
        if not stats:
            print("  (skip) No alignment_stats data to plot")
            return

        episodes = [s['episode'] for s in stats]
        gates = [s['gate'] for s in stats]
        cos_sims = [s['cosine_similarity'] for s in stats]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(gates, bins=20, color='#2E86AB', edgecolor='black', alpha=0.8)
        axes[0].set_title('Distribution of Alignment Gate Values')
        axes[0].set_xlabel('Gate value')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, cos_sims, alpha=0.5, label='cosine similarity', color='#A23B72')
        axes[1].plot(episodes, gates, alpha=0.8, label='gate', color='#2E86AB')
        axes[1].set_title('Alignment Signal Over Meta-Training')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.plots_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_ablation_comparison(self, ablation_results, save_name='ablation_comparison.png'):
        """Grouped bar chart of per-language accuracy across the 4 DACL ablation configs.

        `ablation_results` maps config label (e.g. 'A', 'B', 'C', 'D') to a
        final_results dict as returned by CMLTrainer.evaluate_all_languages."""
        languages = [lang for lang in self.config.all_languages
                     if any(lang in res for res in ablation_results.values())]
        if not languages:
            print("  (skip) No ablation results to plot")
            return

        configs = list(ablation_results.keys())
        n_configs = len(configs)
        fig, ax = plt.subplots(figsize=(16, 6))

        x = np.arange(len(languages))
        width = 0.8 / n_configs
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#3B7A57']

        for i, cfg in enumerate(configs):
            accs = [ablation_results[cfg].get(lang, 0) * 100 for lang in languages]
            offset = (i - (n_configs - 1) / 2) * width
            ax.bar(x + offset, accs, width, label=cfg, color=colors[i % len(colors)])

        ax.set_xlabel('Languages', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('DACL Ablation Comparison Across 15 Languages', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend(title='Configuration')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.config.plots_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_complete_experiment(debug_mode=False):
    """
    Run complete experiment with all phases
    
    Phases:
    1. Load all 15 XNLI language datasets
    2. Warmup fine-tuning on English
    3. Curriculum generation for auxiliary languages (ru, el)
    4. Meta-learning with MAML
    5. Zero-shot evaluation on all 15 languages
    6. Generate plots and results tables
    """
    
    # Update config for debug mode
    if debug_mode:
        config.debug_mode = True
        config.meta_episodes = 10
        config.n_teachers = 3
        config.n_buckets = 5
        # Original default (50) never fires within 10 debug episodes, which
        # would hide the periodic query-loss / prediction-distribution
        # diagnostics that quick_test() is meant to surface.
        config.save_every_n_episodes = 5
        print("\n⚠️  DEBUG MODE ENABLED - Using reduced dataset and episodes\n")
    
    print("="*70)
    print("CURRICULUM META-LEARNING FOR ZERO-SHOT CROSS-LINGUAL TRANSFER")
    print("Paper Implementation with All 15 XNLI Languages")
    print("="*70)
    print(f"\nDevice: {config.device}")
    print(f"Model: {config.model_name}")
    print(f"Languages: {', '.join(config.all_languages)}")
    print(f"Auxiliary: {', '.join(config.aux_langs)}")
    print(f"Target: {', '.join(config.target_langs)}")
    
    # ========================================================================
    # STEP 1: Load Datasets
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: LOADING DATASETS")
    print("="*70)
    
    dataset_manager = DatasetManager(config)
    datasets = dataset_manager.load_all_datasets()
    
    # ========================================================================
    # STEP 2: Initialize Trainer
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: INITIALIZING TRAINER")
    print("="*70)
    
    trainer = CMLTrainer(config, dataset_manager)
    
    # ========================================================================
    # STEP 3: Warmup Fine-tuning
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: WARMUP FINE-TUNING")
    print("="*70)

    if config.skip_warmup:
        print("\n⚠️  skip_warmup=True: pipeline-level ablation -- warmup_finetune() "
              "NOT run. Meta-training starts from the pretrained backbone with a "
              "randomly-initialized classification head.")
    else:
        trainer.warmup_finetune()

    # Evaluate before meta-training starts (after warmup, if it ran; otherwise
    # this is the raw pretrained-backbone + random-head zero-shot accuracy --
    # still saved as baseline_results.json for this run's own results_dir, but
    # NOT the same quantity as the main results/baseline_results.json and not
    # meant to be compared to it as if warmup had happened).
    print("\n[Baseline Evaluation] Before meta-training..."
          if config.skip_warmup else "\n[Baseline Evaluation] After warmup on English...")
    baseline_results = trainer.evaluate_all_languages()

    # Save baseline results
    baseline_path = os.path.join(config.results_dir, 'baseline_results.json')
    save_json(baseline_results, baseline_path)
    
    # ========================================================================
    # STEP 4: Meta-Learning with Curriculum
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: META-LEARNING WITH CURRICULUM")
    print("="*70)
    
    trainer.meta_train_with_curriculum()

    # If a best_meta_model.pt was saved (highest avg zero-shot accuracy seen
    # during periodic meta-eval), load it before the final evaluation so a
    # late-episode wobble doesn't determine the reported number.
    if config.select_best_meta_checkpoint:
        best_meta_path = os.path.join(config.checkpoint_dir, 'best_meta_model.pt')
        if os.path.exists(best_meta_path):
            print(f"\n[Best checkpoint] Loading best_meta_model.pt "
                  f"(avg zero-shot acc {trainer.best_meta_avg_acc:.4f}) for final evaluation...")
            trainer.load_checkpoint('best_meta_model.pt')
        else:
            print("\n[Best checkpoint] No best_meta_model.pt found "
                  "(no periodic eval improved on baseline); using last-episode weights.")

    # ========================================================================
    # STEP 5: Final Evaluation
    # ========================================================================

    print("\n" + "="*70)
    print("STEP 5: FINAL ZERO-SHOT EVALUATION")
    print("="*70)

    final_results = trainer.evaluate_all_languages()
    
    # Save final results
    final_path = os.path.join(config.results_dir, 'final_results.json')
    save_json(final_results, final_path)
    
    # Save complete training history
    history_path = os.path.join(config.results_dir, 'training_history.json')
    save_json(trainer.training_history, history_path)
    
    # ========================================================================
    # STEP 6: Generate Visualizations
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualizer = ResultsVisualizer(config)
    
    # Training curves
    visualizer.plot_training_curves(trainer.training_history)
    
    # Language comparison
    visualizer.plot_language_comparison(final_results, baseline_results)
    
    # Results table
    visualizer.create_results_table(final_results, baseline_results)

    # DACL diagnostics (only produce output if the relevant history is populated)
    visualizer.plot_difficulty_drift(trainer.training_history)
    visualizer.plot_alignment_stats(trainer.training_history)

    # ========================================================================
    # STEP 7: Summary Report
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\nBaseline (after warmup):")
    print(f"  Average Accuracy: {baseline_results.get('average', 0)*100:.2f}%")
    
    print(f"\nFinal (after meta-learning"
          f"{', best_meta_model.pt' if config.select_best_meta_checkpoint else ', last episode'}):")
    print(f"  Average Accuracy: {final_results.get('average', 0)*100:.2f}%")

    improvement = (final_results.get('average', 0) - baseline_results.get('average', 0)) * 100
    print(f"\nImprovement: {improvement:+.2f}%")
    
    print(f"\n✓ All results saved to: {config.results_dir}")
    print(f"✓ All plots saved to: {config.plots_dir}")
    print(f"✓ All checkpoints saved to: {config.checkpoint_dir}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    
    return trainer, final_results, baseline_results

# ============================================================================
# RESUME FROM CHECKPOINT
# ============================================================================

def resume_from_checkpoint(checkpoint_path, continue_training=True):
    """Resume training from a saved checkpoint"""
    
    print("="*70)
    print("RESUMING FROM CHECKPOINT")
    print("="*70)
    print(f"Loading: {checkpoint_path}")
    
    # Load datasets
    dataset_manager = DatasetManager(config)
    datasets = dataset_manager.load_all_datasets()
    
    # Initialize trainer
    trainer = CMLTrainer(config, dataset_manager)
    
    # Load checkpoint - pass full path directly
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    
    print(f"\nCheckpoint Info:")
    print(f"  Phase: {checkpoint.get('phase', 'unknown')}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'episode' in checkpoint:
        print(f"  Episode: {checkpoint['episode']}")
    
    # Restore training history
    if 'training_history' in checkpoint:
        trainer.training_history = checkpoint['training_history']
    
    if continue_training:
        print("\nContinuing training...")
        # Continue meta-training if needed
        if checkpoint.get('phase') == 'warmup':
            trainer.meta_train_with_curriculum()
    
    # Evaluate
    print("\nEvaluating current model...")
    results = trainer.evaluate_all_languages()
    
    return trainer, results

# ============================================================================
# RUN MULTIPLE SEEDS (Table 1: averaged 5 runs)
# ============================================================================

def run_multiple_seeds():
    """
    Run experiment with multiple seeds and average results
    As per paper: "averaged 5 runs" with seeds [42, 26, 11, 2000, 10]
    """
    
    print("="*70)
    print("RUNNING MULTIPLE SEEDS (5 RUNS)")
    print("="*70)
    
    all_results = []
    
    for seed_idx, seed in enumerate(config.seeds):
        print(f"\n{'='*70}")
        print(f"RUN {seed_idx + 1}/5 - SEED {seed}")
        print(f"{'='*70}")
        
        # Set seed
        set_seed(seed)
        config.seed = seed
        
        # Create seed-specific directories
        config.checkpoint_dir = f'./checkpoints_seed_{seed}'
        config.results_dir = f'./results_seed_{seed}'
        config.plots_dir = f'./plots_seed_{seed}'
        
        for dir_path in [config.checkpoint_dir, config.results_dir, config.plots_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Run experiment
        trainer, final_results, baseline_results = run_complete_experiment()
        all_results.append(final_results)
    
    # ========================================================================
    # Aggregate Results
    # ========================================================================
    
    print("\n" + "="*70)
    print("AGGREGATING RESULTS FROM 5 RUNS")
    print("="*70)
    
    # Calculate mean and std for each language
    aggregated = {}
    for lang in config.all_languages + ['average']:
        values = [results[lang] for results in all_results if lang in results]
        aggregated[lang] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    # Create summary table
    summary_data = {
        'Language': [],
        'Mean Accuracy': [],
        'Std Dev': [],
    }
    
    for lang in config.all_languages + ['average']:
        if lang in aggregated:
            summary_data['Language'].append(lang)
            summary_data['Mean Accuracy'].append(f"{aggregated[lang]['mean']*100:.2f}")
            summary_data['Std Dev'].append(f"{aggregated[lang]['std']*100:.2f}")
    
    df = pd.DataFrame(summary_data)
    
    # Save
    summary_path = './results_aggregated_5runs.csv'
    df.to_csv(summary_path, index=False)
    
    print("\n" + df.to_string(index=False))
    print(f"\n✓ Aggregated results saved to: {summary_path}")
    
    # Save detailed results
    detailed_path = './results_all_5runs.json'
    save_json(aggregated, detailed_path)
    
    return aggregated

# ============================================================================
# DACL ABLATION SUITE
# ============================================================================

# Config label -> (curriculum_mode, window_scheduling, alignment_gating)
ABLATION_CONFIGS = {
    'A': dict(curriculum_mode='cross_review', window_scheduling='random', alignment_gating=False),
    'B': dict(curriculum_mode='dacl_static', window_scheduling='random', alignment_gating=False),
    'C': dict(curriculum_mode='dacl_dynamic', window_scheduling='competence', alignment_gating=False),
    'D': dict(curriculum_mode='dacl_dynamic', window_scheduling='competence', alignment_gating=True),
}

# Pipeline-level ablation (Sec. results-ablation-pipeline): coarser than
# ABLATION_CONFIGS above -- varies whether entire PIPELINE STAGES (warmup,
# curriculum) run at all, not which curriculum mechanism is used within
# meta-training. Two of the four rows needed for this ablation are NOT here
# because they already exist on disk and must be reused, not rerun:
#   warmup-only, no meta-learning at all   -> results/baseline_results.json
#   warmup + meta-learning + full DACL     -> ABLATION_CONFIGS['D'] (already run)
PIPELINE_ABLATION_CONFIGS = {
    'no_curriculum_maml': dict(
        skip_warmup=False, curriculum_mode='none', window_scheduling='random', alignment_gating=False),
    'no_warmup_dacl': dict(
        skip_warmup=True, curriculum_mode='dacl_dynamic', window_scheduling='competence', alignment_gating=True),
    'floor_no_warmup_no_curriculum': dict(
        skip_warmup=True, curriculum_mode='none', window_scheduling='random', alignment_gating=False),
}


def run_ablation_suite(debug_mode=True):
    """
    Run the 4 DACL ablation configurations and save a comparison CSV + JSON.

    A: cross_review + random windows + no gating   (Doan & Le baseline)
    B: dacl_static  + random windows + no gating   (difficulty estimator alone)
    C: dacl_dynamic + competence windows + no gating (dynamic curriculum)
    D: dacl_dynamic + competence windows + gating  (full DACL)
    """
    print("="*70)
    print("DACL ABLATION SUITE (configs A/B/C/D)")
    print("="*70)

    ablation_results = {}
    ablation_compute_cost = {}

    for label, overrides in ABLATION_CONFIGS.items():
        print(f"\n{'='*70}\nABLATION {label}: {overrides}\n{'='*70}")

        for key, value in overrides.items():
            setattr(config, key, value)

        set_seed(config.seed)
        trainer, final_results, baseline_results = run_complete_experiment(debug_mode=debug_mode)

        ablation_results[label] = final_results
        ablation_compute_cost[label] = trainer.training_history.get('compute_cost', {})

    # Comparison CSV: rows = languages + average, columns = ablation configs
    rows = []
    for lang in config.all_languages + ['average']:
        row = {'Language': lang}
        for label in ABLATION_CONFIGS:
            row[label] = ablation_results[label].get(lang, None)
        rows.append(row)
    df = pd.DataFrame(rows)

    csv_path = os.path.join(config.results_dir, 'ablation_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved: {csv_path}")

    json_path = os.path.join(config.results_dir, 'ablation_comparison.json')
    save_json({
        'configs': ABLATION_CONFIGS,
        'results': ablation_results,
        'compute_cost': ablation_compute_cost
    }, json_path)

    # Ablation comparison plot
    visualizer = ResultsVisualizer(config)
    visualizer.plot_ablation_comparison(ablation_results)

    print("\n" + df.to_string(index=False))
    return ablation_results, ablation_compute_cost

# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def quick_test():
    """Quick test with minimal data for debugging"""
    print("\n🔧 QUICK TEST MODE")
    print("This will run a minimal experiment to test the pipeline\n")
    
    return run_complete_experiment(debug_mode=True)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CHOOSE YOUR EXECUTION MODE
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXECUTION OPTIONS")
    print("="*70)
    print("\n1. Quick Test (debug mode with minimal data)")
    print("2. Full Single Run (complete experiment with 1 seed)")
    print("3. Full Multiple Runs (5 seeds as per paper)")
    print("4. Resume from checkpoint")
    
    # For Colab/Kaggle, uncomment the mode you want:
    
    # MODE 1: Quick Test (recommended first)
    # trainer, final_results, baseline_results = quick_test()
    
    # MODE 2: Full Single Run (recommended for initial full run)
    trainer, final_results, baseline_results = run_complete_experiment(debug_mode=False)
    
    # MODE 3: Full Multiple Runs (for final paper results)
    #aggregated_results = run_multiple_seeds()
    
    # MODE 4: Resume from checkpoint (FIXED - now works with absolute paths)
    #trainer, results = resume_from_checkpoint('/kaggle/input/mdebertanlpxnnlibestpth/best_warmup_model.pt')
    
    print("\n✅ EXECUTION COMPLETE!")
    print("\nCheck these directories for outputs:")
    print(f"  📁 Checkpoints: {config.checkpoint_dir}")
    print(f"  📊 Results: {config.results_dir}")
    print(f"  📈 Plots: {config.plots_dir}")