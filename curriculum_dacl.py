# ============================================================================
# DACL: Dynamic Alignment-gated Curriculum Learning
#
# Replaces Doan & Le's static multi-teacher cross-review curriculum with:
#   1. SelfPacedDifficultyEstimator - teacher-free difficulty scoring using the
#      current meta-model itself (re-estimated periodically -> "dynamic").
#   2. CompetenceScheduler - competence-based window scheduling (Platanios et
#      al. 2019 style) instead of uniform-random window sampling.
#   3. AlignmentGate - gates each meta-update by its cosine alignment with a
#      probe gradient computed on a small fixed high-resource (English) held
#      out set, to mitigate negative transfer from noisy auxiliary tasks.
#
# See NOVELTY.md for the full method description and equations.
# ============================================================================

import random
import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except ImportError:
    _scipy_spearmanr = None


def spearman_drift(old_scores, new_scores):
    """Spearman rank correlation between two difficulty score vectors.

    Uses scipy if available, otherwise falls back to a manual rank-correlation
    computation (Pearson correlation of ranks) so this module has no hard
    scipy dependency.
    """
    old_scores = np.asarray(old_scores, dtype=np.float64)
    new_scores = np.asarray(new_scores, dtype=np.float64)

    if _scipy_spearmanr is not None:
        corr, _ = _scipy_spearmanr(old_scores, new_scores)
        return float(corr)

    old_ranks = np.argsort(np.argsort(old_scores))
    new_ranks = np.argsort(np.argsort(new_scores))
    if np.std(old_ranks) == 0 or np.std(new_ranks) == 0:
        return 0.0
    corr = np.corrcoef(old_ranks, new_ranks)[0, 1]
    return float(corr)


def _zscore(values):
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


class SelfPacedDifficultyEstimator:
    """Component 1: teacher-free dynamic difficulty estimation.

    difficulty_j(t) = lambda_loss * CE(f_theta_t(x_j), y_j)
                     + lambda_ent  * H(softmax(f_theta_t(x_j)))

    Both terms are z-normalized across the dataset before combining so their
    scales are comparable. A single no_grad forward pass over the dataset is
    used (cheap: ~2490 samples per aux language at eval batch size).
    """

    def __init__(self, lambda_loss=1.0, lambda_ent=0.5, batch_size=16):
        self.lambda_loss = lambda_loss
        self.lambda_ent = lambda_ent
        self.batch_size = batch_size
        self.forward_pass_count = 0

    def score(self, model, dataset, processor, device):
        """Compute difficulty scores for every sample in `dataset`.

        Returns a list of floats aligned with dataset indices, matching the
        return shape of CurriculumGenerator.cross_review_evaluate so it can
        be passed directly into CurriculumGenerator.annealing_arrangement.
        """
        was_training = model.training
        model.eval()

        losses = np.zeros(len(dataset), dtype=np.float64)
        entropies = np.zeros(len(dataset), dtype=np.float64)

        with torch.no_grad():
            for start in range(0, len(dataset), self.batch_size):
                end = min(start + self.batch_size, len(dataset))
                batch_data = dataset.select(range(start, end))
                batch = processor.prepare_batch(batch_data)

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                self.forward_pass_count += 1

                batch_losses = F.cross_entropy(logits, labels, reduction='none')

                probs = F.softmax(logits, dim=-1)
                batch_entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)

                losses[start:end] = batch_losses.cpu().numpy()
                entropies[start:end] = batch_entropy.cpu().numpy()

                del logits, probs, batch_losses, batch_entropy
                del input_ids, attention_mask, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if was_training:
            model.train()

        loss_z = _zscore(losses)
        ent_z = _zscore(entropies)
        difficulty = self.lambda_loss * loss_z + self.lambda_ent * ent_z

        return difficulty.tolist()


class CompetenceScheduler:
    """Component 2: competence-based window scheduling.

    competence(e) = min(1.0, sqrt( e * (1 - c0^2) / n_episodes + c0^2 ))

    At episode e, only annealed windows whose max difficulty percentile is
    <= competence(e) are eligible; sample uniformly among eligible windows.

    `direction='hard_first'` inverts this gate: instead of unlocking the
    easiest windows first (which, for an already-strong warmup model, are
    near-zero-loss and carry almost no meta-gradient signal), early episodes
    are restricted to the hardest (highest-difficulty-percentile) windows --
    where the model is currently wrong and the gradient is informative --
    and easier windows are folded in as competence grows. `direction=
    'easy_first'` reproduces the original Platanios et al. ordering.
    """

    def __init__(self, n_episodes, initial_competence=0.1, direction='easy_first'):
        self.n_episodes = max(1, n_episodes)
        self.c0 = initial_competence
        assert direction in ('easy_first', 'hard_first')
        self.direction = direction

    def competence(self, episode):
        c0_sq = self.c0 ** 2
        value = (episode * (1 - c0_sq) / self.n_episodes) + c0_sq
        return float(min(1.0, np.sqrt(max(value, 0.0))))

    def select_window(self, annealed_batches, n_buckets, window_step, episode, rng):
        """Select a window index eligible under the current competence level.

        Windows are built (see CurriculumGenerator.annealing_arrangement) as
        contiguous bucket ranges [i, i+window_step) over buckets sorted by
        difficulty ascending. Window i's max difficulty percentile is thus
        (i + window_step) / n_buckets, and its min difficulty percentile is
        i / n_buckets.
        """
        n_windows = len(annealed_batches)
        current_competence = self.competence(episode)

        eligible = []
        if self.direction == 'hard_first':
            for i in range(n_windows):
                min_percentile = i / n_buckets
                if min_percentile >= (1.0 - current_competence):
                    eligible.append(i)
            default = n_windows - 1
        else:
            for i in range(n_windows):
                max_percentile = min(1.0, (i + window_step) / n_buckets)
                if max_percentile <= current_competence:
                    eligible.append(i)
            default = 0

        if not eligible:
            eligible = [default]

        window_idx = rng.choice(eligible)
        return annealed_batches[window_idx], window_idx, current_competence


class AlignmentGate:
    """Component 3: alignment-gated meta-updates.

    Gates each meta-update by the cosine similarity between the classifier
    head gradient of the current task batch and a cached probe gradient
    computed on a small fixed high-resource (English) held-out set. This
    never touches target-language data, preserving the zero-shot protocol.
    """

    def __init__(self, model, probe_dataset, processor, device,
                 probe_refresh_every=10, gate_min=0.1):
        self.model = model
        self.probe_dataset = probe_dataset
        self.processor = processor
        self.device = device
        self.probe_refresh_every = probe_refresh_every
        self.gate_min = gate_min

        self.head_param_names = [
            name for name, _ in model.named_parameters()
            if 'classifier' in name or 'pooler' in name
        ]
        print(f"[AlignmentGate] Classifier head parameters: {self.head_param_names}")

        self._cached_probe_grad = None
        self.forward_pass_count = 0

    def _head_params(self):
        name_set = set(self.head_param_names)
        return [p for name, p in self.model.named_parameters() if name in name_set]

    def maybe_refresh_probe_grad(self, episode):
        """Recompute and cache the probe gradient every probe_refresh_every episodes."""
        if episode % self.probe_refresh_every != 0 and self._cached_probe_grad is not None:
            return

        batch = self.processor.prepare_batch(self.probe_dataset)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        head_params = self._head_params()
        logits = self.model(input_ids, attention_mask)
        self.forward_pass_count += 1
        loss = F.cross_entropy(logits, labels)

        grads = torch.autograd.grad(
            loss, head_params, create_graph=False, retain_graph=False
        )
        flat_grad = torch.cat([g.detach().reshape(-1) for g in grads])
        self._cached_probe_grad = flat_grad

        del logits, loss, grads, input_ids, attention_mask, labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_gate(self, task_head_grad_flat):
        """Cosine similarity between task head-grad and cached probe head-grad."""
        if self._cached_probe_grad is None:
            return 1.0, 0.0

        probe_grad = self._cached_probe_grad.to(task_head_grad_flat.device)
        cos_sim = F.cosine_similarity(
            task_head_grad_flat.unsqueeze(0), probe_grad.unsqueeze(0)
        ).item()

        gate = float(np.clip((1.0 + cos_sim) / 2.0, self.gate_min, 1.0))
        return gate, cos_sim

    def get_task_head_grad_flat(self):
        """Flatten the current .grad of the classifier head parameters."""
        name_set = set(self.head_param_names)
        grads = []
        for name, p in self.model.named_parameters():
            if name in name_set and p.grad is not None:
                grads.append(p.grad.detach().reshape(-1))
        if not grads:
            return None
        return torch.cat(grads)
