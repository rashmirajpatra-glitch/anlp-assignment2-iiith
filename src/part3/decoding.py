from dataclasses import dataclass
import copy
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BeamHypothesis:
    """Hypothesis container for manual beam search tracking."""
    token_ids: List[int]
    cumulative_log_prob: float
    is_finished: bool = False
    generated_length: int = 0
    past_key_values: Optional[Any] = None

    def score(self, length_penalty: float = 1.0) -> float:
        """
        Compute length-normalized score:
        score = cumulative_log_prob / (length ^ length_penalty)
        """
        length = self.generated_length
        if length == 0:
            return -float("inf")
        return self.cumulative_log_prob / (length ** length_penalty)


def _extract_logits_and_kv(out: Any) -> Tuple[torch.Tensor, Any]:
    """Helper to extract logits and past_key_values from dict or ModelOutput objects."""
    if isinstance(out, dict):
        logits = out["logits"]
        past_kv = out.get("past_key_values", None)
    else:
        logits = getattr(out, "logits", out)
        past_kv = getattr(out, "past_key_values", None)
    return logits, past_kv


def top_p_filter_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest descending-probability nucleus whose mass reaches p."""
    if not 0.0 < p <= 1.0:
        raise ValueError("top-p threshold must be in (0, 1]")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove_sorted = cumulative_probs > p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    remove_original = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
    return logits.masked_fill(remove_original, -float("inf"))


def top_k_filter_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Retain exactly the top-k logit indices, including when values tie."""
    if k < 1:
        raise ValueError("top-k value must be at least 1")
    actual_k = min(k, logits.size(-1))
    top_values, top_indices = torch.topk(logits, k=actual_k, dim=-1)
    filtered = torch.full_like(logits, -float("inf"))
    return filtered.scatter(-1, top_indices, top_values)


def greedy_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Greedy Autoregressive Decoding implemented strictly via model.forward()
    (NEVER calls model.generate()).

    Selects argmax at each step. Employs KV caching when available for efficiency.
    """
    model.eval()
    curr_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():
        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                model_inputs = curr_ids[:, -1:]
                out = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            else:
                out = model(curr_ids, use_cache=use_cache)

            logits, past_key_values = _extract_logits_and_kv(out)
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return curr_ids


def top_k_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    k: int = 50,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Top-k Sampling implemented strictly via model.forward().
    Filters logits to only the top-k highest-probability candidates,
    renormalizes via softmax with temperature, and samples via torch.multinomial.
    """
    model.eval()
    curr_ids = input_ids.clone()
    past_key_values = None

    temperature = max(1e-5, temperature)

    with torch.no_grad():
        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                model_inputs = curr_ids[:, -1:]
                out = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            else:
                out = model(curr_ids, use_cache=use_cache)

            logits, past_key_values = _extract_logits_and_kv(out)
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Top-k filtering: set logits below top-k threshold to -inf
            filtered_logits = top_k_filter_logits(next_token_logits, k)

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return curr_ids


def top_p_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    p: float = 0.9,
    temperature: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Top-p (Nucleus) Sampling implemented strictly via model.forward().
    Computes cumulative distribution over descending-sorted tokens, truncates
    the tail whose cumulative probability exceeds p, and samples from the nucleus.
    """
    model.eval()
    curr_ids = input_ids.clone()
    past_key_values = None

    temperature = max(1e-5, temperature)

    with torch.no_grad():
        for step in range(max_new_tokens):
            if use_cache and past_key_values is not None:
                model_inputs = curr_ids[:, -1:]
                out = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            else:
                out = model(curr_ids, use_cache=use_cache)

            logits, past_key_values = _extract_logits_and_kv(out)
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            filtered_logits = top_p_filter_logits(next_token_logits, p)

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return curr_ids


def beam_search_decode(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
    beam_width: int = 4,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    From-scratch Beam Search Decoding at arbitrary beam widths (1, 2, 4).
    Built exclusively on repeated model.forward() calls.

    Tracks active hypotheses, cumulative log probabilities, and length normalization:
      score = cumulative_log_prob / (length ^ length_penalty)
    Batches active candidate forward passes to maintain memory efficiency.
    """
    model.eval()
    device = input_ids.device
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    prompt_tokens = input_ids[0].tolist()

    # Initial hypothesis
    beams: List[BeamHypothesis] = [
        BeamHypothesis(token_ids=prompt_tokens, cumulative_log_prob=0.0, is_finished=False)
    ]

    with torch.no_grad():
        for step in range(max_new_tokens):
            if all(b.is_finished for b in beams):
                break

            candidates: List[BeamHypothesis] = []
            active_beams = [b for b in beams if not b.is_finished]
            finished_beams = [b for b in beams if b.is_finished]

            for fb in finished_beams:
                candidates.append(fb)

            if len(active_beams) > 0:
                first_cache = active_beams[0].past_key_values
                tuple_cache = isinstance(first_cache, (list, tuple))
                dynamic_cache = (
                    first_cache is not None
                    and hasattr(first_cache, "to_legacy_cache")
                    and hasattr(type(first_cache), "from_legacy_cache")
                    and hasattr(first_cache, "batch_select_indices")
                )
                cached_beams = first_cache is not None and (tuple_cache or dynamic_cache) and all(
                    type(beam.past_key_values) is type(first_cache)
                    for beam in active_beams
                )
                if cached_beams:
                    batch_tensor = torch.tensor(
                        [[beam.token_ids[-1]] for beam in active_beams],
                        dtype=torch.long,
                        device=device,
                    )
                    if tuple_cache:
                        layer_count = len(first_cache)
                        batch_past = [
                            tuple(
                                torch.cat([beam.past_key_values[layer][component] for beam in active_beams], dim=0)
                                for component in range(2)
                            )
                            for layer in range(layer_count)
                        ]
                    else:
                        legacy_caches = [beam.past_key_values.to_legacy_cache() for beam in active_beams]
                        legacy_batch = tuple(
                            tuple(
                                torch.cat([cache[layer][component] for cache in legacy_caches], dim=0)
                                for component in range(2)
                            )
                            for layer in range(len(legacy_caches[0]))
                        )
                        batch_past = type(first_cache).from_legacy_cache(legacy_batch)
                    out = model(batch_tensor, past_key_values=batch_past, use_cache=True)
                else:
                    batch_tensor = torch.tensor(
                        [beam.token_ids for beam in active_beams],
                        dtype=torch.long,
                        device=device,
                    )
                    out = model(batch_tensor, use_cache=True)
                logits, updated_past = _extract_logits_and_kv(out)
                last_logits = logits[:, -1, :]  # (n_active, vocab_size)
                log_probs = F.log_softmax(last_logits, dim=-1)
                if isinstance(updated_past, (list, tuple)):
                    child_caches = [
                        [
                            tuple(component[index:index + 1] for component in layer)
                            for layer in updated_past
                        ]
                        for index in range(len(active_beams))
                    ]
                elif (
                    updated_past is not None
                    and hasattr(updated_past, "batch_select_indices")
                ):
                    child_caches = []
                    for index in range(len(active_beams)):
                        cache = copy.deepcopy(updated_past)
                        cache.batch_select_indices(torch.tensor([index], device=device))
                        child_caches.append(cache)
                else:
                    child_caches = [None] * len(active_beams)

                vocab_size = last_logits.size(-1)
                num_to_expand = min(2 * beam_width, vocab_size)

                for i, b in enumerate(active_beams):
                    top_lp, top_idx = torch.topk(log_probs[i], k=num_to_expand, dim=-1)

                    for lp_val, token_id_val in zip(top_lp, top_idx):
                        tid = token_id_val.item()
                        is_eos = (eos_token_id is not None and tid == eos_token_id)
                        cand = BeamHypothesis(
                            token_ids=b.token_ids + [tid],
                            cumulative_log_prob=b.cumulative_log_prob + lp_val.item(),
                            is_finished=is_eos,
                            generated_length=b.generated_length + 1,
                            past_key_values=child_caches[i],
                        )
                        candidates.append(cand)

            candidates.sort(key=lambda x: x.score(length_penalty=length_penalty), reverse=True)
            beams = candidates[:beam_width]

    best_hypothesis = max(beams, key=lambda x: x.score(length_penalty=length_penalty))
    return torch.tensor([best_hypothesis.token_ids], dtype=torch.long, device=device)


class ManualDecoder:
    """Unified wrapper around all 4 manual decoding strategies."""
    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def decode(
        self,
        strategy: str,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> torch.Tensor:
        strat = strategy.lower()
        if strat == "greedy":
            return greedy_decode(
                self.model, input_ids, max_new_tokens,
                eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
                **kwargs
            )
        elif strat in ["top_k", "topk"]:
            k = kwargs.get("k", 50)
            temp = kwargs.get("temperature", 1.0)
            return top_k_decode(
                self.model, input_ids, max_new_tokens,
                k=k, temperature=temp,
                eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
            )
        elif strat in ["top_p", "topp", "nucleus"]:
            p = kwargs.get("p", 0.9)
            temp = kwargs.get("temperature", 1.0)
            return top_p_decode(
                self.model, input_ids, max_new_tokens,
                p=p, temperature=temp,
                eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
            )
        elif strat in ["beam", "beam_search"]:
            beam_width = kwargs.get("beam_width", 4)
            length_penalty = kwargs.get("length_penalty", 1.0)
            return beam_search_decode(
                self.model, input_ids, max_new_tokens,
                beam_width=beam_width, length_penalty=length_penalty,
                eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id,
            )
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Choose from: greedy, top_k, top_p, beam_search")
