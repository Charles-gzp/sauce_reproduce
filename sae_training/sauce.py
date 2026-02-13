from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch


@dataclass
class SauceFeatureSelectionResult:
    scores: torch.Tensor
    topk_indices: torch.Tensor
    mu_pos: torch.Tensor
    mu_neg: torch.Tensor


def get_token_index(class_token: bool, sae_target_token: str) -> int:
    # Keep backward compatibility for old configs that only set class_token=True.
    if class_token:
        return 0
    if sae_target_token == "class":
        return 0
    if sae_target_token == "last":
        return -1
    raise ValueError(f"Unsupported sae_target_token: {sae_target_token}")


@torch.no_grad()
def get_feature_activations_for_tokens(
    sparse_autoencoder,
    token_activations: torch.Tensor,
    batch_size: int = 4096,
) -> torch.Tensor:
    acts: List[torch.Tensor] = []
    for start in range(0, token_activations.shape[0], batch_size):
        # Run SAE encoder on token activations and cache feature activations z.
        batch = token_activations[start : start + batch_size]
        _, feature_acts, _, _, _, _ = sparse_autoencoder(batch)
        acts.append(feature_acts.detach())
    return torch.cat(acts, dim=0)


@torch.no_grad()
def select_features_by_pos_neg_correlation(
    pos_feature_acts: torch.Tensor,
    neg_feature_acts: torch.Tensor,
    top_k: int,
    delta: float = 1e-6,
) -> SauceFeatureSelectionResult:
    if pos_feature_acts.ndim != 2 or neg_feature_acts.ndim != 2:
        raise ValueError("Feature activations must be rank-2 tensors [batch, d_sae].")
    if pos_feature_acts.shape[1] != neg_feature_acts.shape[1]:
        raise ValueError("Positive and negative activations must share d_sae.")

    # Mean feature activation in positive/negative sets.
    mu_pos = pos_feature_acts.mean(dim=0)
    mu_neg = neg_feature_acts.mean(dim=0)

    # SAUCE scoring: normalized positive mass minus normalized negative mass.
    pos_norm = mu_pos.sum() + delta
    neg_norm = mu_neg.sum() + delta
    scores = (mu_pos / pos_norm) - (mu_neg / neg_norm)

    _, topk_indices = torch.topk(scores, k=top_k, dim=0)
    return SauceFeatureSelectionResult(
        scores=scores,
        topk_indices=topk_indices,
        mu_pos=mu_pos,
        mu_neg=mu_neg,
    )


def build_sauce_intervention_hook(
    sparse_autoencoder,
    selected_feature_indices: Sequence[int],
    gamma: float = -0.5,
    token_index: int = -1,
):
    selected_feature_indices = list(selected_feature_indices)
    if len(selected_feature_indices) == 0:
        raise ValueError("selected_feature_indices must not be empty.")

    feature_index_tensor = torch.tensor(
        selected_feature_indices,
        dtype=torch.long,
        device=sparse_autoencoder.device,
    )

    @torch.no_grad()
    def hook_fn(activations):
        # Encode token activation into SAE feature space.
        token_acts = activations[:, token_index, :]
        token_acts = token_acts.to(sparse_autoencoder.dtype)
        sae_in = token_acts - sparse_autoencoder.b_dec
        hidden_pre = sae_in @ sparse_autoencoder.W_enc + sparse_autoencoder.b_enc
        feature_acts = torch.relu(hidden_pre)
        # Apply concept suppression only on selected feature indices.
        feature_acts[:, feature_index_tensor] *= gamma
        # Decode edited features back to token activation space.
        edited = feature_acts @ sparse_autoencoder.W_dec + sparse_autoencoder.b_dec
        activations = activations.clone()
        activations[:, token_index, :] = edited.to(activations.dtype)
        return (activations,)

    return hook_fn


def parse_yes_no_answer(text: str) -> Optional[bool]:
    lower = text.strip().lower()
    if "yes" in lower:
        return True
    if "no" in lower:
        return False
    return None


def compute_uad_score(answers: Iterable[str], concept_present_labels: Iterable[bool]) -> float:
    valid = 0
    correct = 0
    for answer, concept_present in zip(answers, concept_present_labels):
        yn = parse_yes_no_answer(answer)
        if yn is None:
            continue
        valid += 1
        correct += int(yn == concept_present)
    if valid == 0:
        return 0.0
    return correct / valid


def compute_uag_score(gpt_ratings: Iterable[int]) -> float:
    ratings = list(gpt_ratings)
    if len(ratings) == 0:
        return 0.0
    # Paper setup: 2 = does not include target concept (desired after unlearning)
    return float(sum(int(r == 2) for r in ratings)) / float(len(ratings))


def compute_retain_accuracy(non_target_concept_correct: Iterable[bool]) -> float:
    values = list(non_target_concept_correct)
    if len(values) == 0:
        return 0.0
    return float(sum(int(v) for v in values)) / float(len(values))
