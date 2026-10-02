from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for Decoder-Only Transformer with pluggable FFN / MoE."""
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048                 # Standard MLP intermediate dimension (4 * d_model)
    vocab_size: int = 32000
    max_seq_len: int = 512
    dropout: float = 0.1
    use_moe: bool = False
    # Variant options: "dense", "top1", "top2", "shared_plus_routed", "top2_matched_active"
    moe_variant: str = "dense"
    n_experts: int = 4
    top_k: int = 1
    aux_loss_coef: float = 0.01      # Load balancing loss coefficient
    router_z_loss_coef: float = 1e-3
    tie_embeddings: bool = True
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        aliases = {
            "moe_4_1": "top1",
            "moe_4_2": "top2",
            "moe_shared": "shared_plus_routed",
            "moe_4_2_active": "top2_matched_active",
        }
        self.moe_variant = aliases.get(self.moe_variant, self.moe_variant)
        if self.moe_variant != "dense":
            self.use_moe = True

    def get_expert_dim(self) -> int:
        """
        Compute the intermediate hidden dimension per expert based on the variant
        to satisfy the assignment parameter matching rules:
        - Variants 1-4 (dense, top1, top2, shared_plus_routed): matched TOTAL parameters.
        - Variant 5 (top2_matched_active): matched ACTIVE parameters to Variant 1.
        """
        if not self.use_moe or self.moe_variant == "dense":
            return self.d_ff

        d_m = self.d_model
        dense_total = 2 * d_m * self.d_ff + self.d_ff + d_m  # Variant 1 (dense) FFN params/layer
        per_expert_coef = 2 * d_m + 1  # coefficient of d_expert in a single expert's param count

        if self.moe_variant in ["top1", "top2"]:
            # total = n_experts * single_expert + router(d_m * n_experts) == dense_total
            router_params = d_m * self.n_experts
            numerator = dense_total - router_params - self.n_experts * d_m
            return max(1, round(numerator / (self.n_experts * per_expert_coef)))
        elif self.moe_variant == "shared_plus_routed":
            # total = n_experts * single_expert + router(d_m * (n_experts - 1)) == dense_total
            router_params = d_m * (self.n_experts - 1)
            numerator = dense_total - router_params - self.n_experts * d_m
            return max(1, round(numerator / (self.n_experts * per_expert_coef)))
        elif self.moe_variant == "top2_matched_active":
            # active = top_k * single_expert + router(d_m * n_experts) == dense_total (Variant 1 active)
            router_params = d_m * self.n_experts
            numerator = dense_total - router_params - self.top_k * d_m
            return max(1, round(numerator / (self.top_k * per_expert_coef)))
        else:
            return max(1, self.d_ff // self.n_experts)

    def count_parameters_detail(self) -> Dict[str, int]:
        """Theoretical calculation of FFN total and active parameters per block."""
        d_m = self.d_model
        d_expert = self.get_expert_dim()

        if not self.use_moe or self.moe_variant == "dense":
            total_ffn_params = 2 * d_m * self.d_ff + self.d_ff + d_m
            active_ffn_params = total_ffn_params
            active_expert_params = total_ffn_params
        elif self.moe_variant in ["top1", "top2"]:
            single_expert = 2 * d_m * d_expert + d_expert + d_m
            router_params = d_m * self.n_experts
            total_ffn_params = self.n_experts * single_expert + router_params
            active_ffn_params = self.top_k * single_expert + router_params
            active_expert_params = self.top_k * single_expert
        elif self.moe_variant == "shared_plus_routed":
            # 1 shared expert + (n_experts - 1) routed experts
            single_expert = 2 * d_m * d_expert + d_expert + d_m
            router_params = d_m * (self.n_experts - 1)
            total_ffn_params = self.n_experts * single_expert + router_params
            # 1 shared active + 1 routed active = 2 active
            active_ffn_params = 2 * single_expert + router_params
            active_expert_params = 2 * single_expert
        elif self.moe_variant == "top2_matched_active":
            single_expert = 2 * d_m * d_expert + d_expert + d_m
            router_params = d_m * self.n_experts
            total_ffn_params = self.n_experts * single_expert + router_params
            active_ffn_params = self.top_k * single_expert + router_params
            active_expert_params = self.top_k * single_expert
        else:
            total_ffn_params = 0
            active_ffn_params = 0
            active_expert_params = 0

        return {
            "d_expert": d_expert,
            "total_ffn_params_per_layer": total_ffn_params,
            "active_ffn_params_per_layer": active_ffn_params,
            "active_expert_params_per_layer": active_expert_params,
        }


class MLP(nn.Module):
    """
    Standard 2-layer Feed-Forward Network with GELU activation.
    Built purely with lower-level nn.Linear primitives.
    """
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) or (N, D)
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class MoE(nn.Module):
    """
    Mixture of Experts FFN module supporting:
    1. 'top1': 4 experts, top-1 active, matched total params to dense baseline.
    2. 'top2': 4 experts, top-2 active, matched total params to dense baseline.
    3. 'shared_plus_routed': 1 shared + 3 routed (top-1 active), matched total params to dense.
    4. 'top2_matched_active': 4 experts, top-2 active, matched active params to dense baseline.

    Computes Switch/GShard load-balancing auxiliary loss and retains routing statistics
    for language-conditional activation heatmaps (e.g. Vietnamese vs Japanese).
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.moe_variant = config.moe_variant
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        self.aux_loss_coef = config.aux_loss_coef
        self.d_expert = config.get_expert_dim()

        # Build experts according to variant
        if self.moe_variant == "shared_plus_routed":
            # 1 always-active shared expert + (n_experts - 1) routed experts
            self.shared_expert = MLP(self.d_model, self.d_expert, dropout=config.dropout)
            self.n_routed = self.n_experts - 1
            self.routed_experts = nn.ModuleList([
                MLP(self.d_model, self.d_expert, dropout=config.dropout)
                for _ in range(self.n_routed)
            ])
            self.router = nn.Linear(self.d_model, self.n_routed, bias=False)
            self.top_k_routed = 1
        else:
            self.n_routed = self.n_experts
            self.experts = nn.ModuleList([
                MLP(self.d_model, self.d_expert, dropout=config.dropout)
                for _ in range(self.n_experts)
            ])
            self.router = nn.Linear(self.d_model, self.n_experts, bias=False)
            self.top_k_routed = self.top_k

        # Routing statistics cache for evaluation & heatmaps
        self.last_routing_indices: Optional[torch.Tensor] = None   # (N, top_k)
        self.last_router_probs: Optional[torch.Tensor] = None      # (N, n_routed)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, T, D)
        Returns:
            out: Output tensor of shape (B, T, D)
            aux_loss: Scalar load-balancing auxiliary loss
            z_loss: Scalar router log-partition regularization loss
        """
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)  # (B*T, D)
        valid_mask = (
            attention_mask.reshape(-1).bool()
            if attention_mask is not None
            else torch.ones(x_flat.size(0), dtype=torch.bool, device=x.device)
        )
        valid_flat = x_flat[valid_mask]
        N = valid_flat.size(0)
        if N == 0:
            zero = torch.zeros((), device=x.device, dtype=x.dtype)
            self.last_routing_indices = None
            self.last_router_probs = None
            return torch.zeros_like(x), zero, zero

        if self.moe_variant == "shared_plus_routed":
            # 1. Compute shared expert output
            shared_out = self.shared_expert(valid_flat)

            # 2. Route across remaining routed experts
            router_logits = self.router(valid_flat)  # (N, n_routed)
            router_probs = F.softmax(router_logits, dim=-1)  # (N, n_routed)

            # Top-1 routed selection
            topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k_routed, dim=-1)
            # Normalize topk weights
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

            routed_out = torch.zeros_like(valid_flat)
            # Dispatch to routed experts
            for e_idx in range(self.n_routed):
                mask = (topk_indices[:, 0] == e_idx)
                if mask.any():
                    expert_input = valid_flat[mask]
                    expert_output = self.routed_experts[e_idx](expert_input)
                    routed_out[mask] += topk_weights[mask, 0:1] * expert_output

            out_flat = shared_out + routed_out

            # Load balancing auxiliary loss on routed experts (Switch/GShard)
            # f_i: fraction of tokens assigned to routed expert i
            f = torch.zeros(self.n_routed, device=x.device, dtype=torch.float32)
            assigned_counts = torch.bincount(topk_indices[:, 0], minlength=self.n_routed).float()
            f = assigned_counts / N
            # P_i: average probability of routing to expert i
            P = router_probs.mean(dim=0)
            aux_loss = self.aux_loss_coef * self.n_routed * torch.sum(f * P)
            z_loss = self.config.router_z_loss_coef * torch.mean(torch.logsumexp(router_logits, dim=-1).square())

            # Cache routing info for heatmaps (mark expert 0 as shared, 1..3 as routed)
            self.last_routing_indices = topk_indices.detach()
            self.last_router_probs = router_probs.detach()

        else:
            # Standard MoE: top-1 or top-2 across all n_experts
            router_logits = self.router(valid_flat)  # (N, n_experts)
            router_probs = F.softmax(router_logits, dim=-1)  # (N, n_experts)

            topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

            out_flat = torch.zeros_like(valid_flat)
            # Dispatch tokens to experts
            for k in range(self.top_k):
                k_indices = topk_indices[:, k]
                k_weights = topk_weights[:, k:k+1]
                for e_idx in range(self.n_experts):
                    mask = (k_indices == e_idx)
                    if mask.any():
                        expert_input = valid_flat[mask]
                        expert_output = self.experts[e_idx](expert_input)
                        out_flat[mask] += k_weights[mask] * expert_output

            # Load-balancing aux loss across all experts
            # f_i: fraction of all routing slots assigned to expert i
            all_assigned = topk_indices.view(-1)
            assigned_counts = torch.bincount(all_assigned, minlength=self.n_experts).float()
            f = assigned_counts / (N * self.top_k)
            P = router_probs.mean(dim=0)
            aux_loss = self.aux_loss_coef * self.n_experts * torch.sum(f * P)
            z_loss = self.config.router_z_loss_coef * torch.mean(torch.logsumexp(router_logits, dim=-1).square())

            self.last_routing_indices = topk_indices.detach()
            self.last_router_probs = router_probs.detach()

        padded_out = torch.zeros_like(x_flat)
        padded_out[valid_mask] = out_flat
        out = padded_out.view(orig_shape)
        return out, aux_loss, z_loss


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention constructed from lower-level nn.Linear primitives
    and torch.nn.functional.scaled_dot_product_attention (SDPA).
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout_p = config.dropout
        self.force_math_sdpa = False

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # x: (B, T, D)
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, d_k)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None

        # SDPA causal attention
        # If past_key_value is provided during incremental decoding, is_causal is False
        # because q attends to all cached past keys + current key
        is_causal = (past_key_value is None) and (T > 1)
        dropout_p = self.dropout_p if self.training else 0.0

        if self.force_math_sdpa:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )  # (B, H, T, d_k)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.resid_dropout(self.out_proj(attn_out))
        return out, new_kv


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer Decoder Block built from primitive layers:
    x -> LayerNorm -> CausalSelfAttention -> Residual
      -> LayerNorm -> (MoE or MLP)        -> Residual
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        if config.use_moe:
            self.ffn = MoE(config)
        else:
            self.ffn = MLP(config.d_model, config.d_ff, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Self-Attention
        normed_x = self.norm1(x)
        attn_out, new_kv = self.attn(normed_x, past_key_value=past_key_value, use_cache=use_cache)
        x = x + attn_out

        # Pre-LN FFN / MoE
        normed_x2 = self.norm2(x)
        if self.config.use_moe:
            ffn_out, aux_loss, router_z_loss = self.ffn(normed_x2, attention_mask=attention_mask)
        else:
            ffn_out = self.ffn(normed_x2)
            aux_loss = torch.tensor(0.0, device=x.device)
            router_z_loss = torch.tensor(0.0, device=x.device)

        x = x + ffn_out
        return x, aux_loss, router_z_loss, new_kv


class Transformer(nn.Module):
    """
    Complete Decoder-Only Transformer built strictly from lower-level primitives
    (no nn.TransformerDecoder or composite container classes).
    Supports standard dense baseline as well as all 4 MoE variants.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between token embeddings and LM projection
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Standard Gaussian initialization for Transformer weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for training and autoregressive generation.
        Args:
            input_ids: (B, T) LongTensor of token IDs
            labels: Optional (B, T) LongTensor for cross-entropy LM loss calculation
            past_key_values: Optional cached KV states per layer
            use_cache: Whether to return updated KV caches
        Returns:
            Dictionary with:
              'logits': (B, T, vocab_size)
              'loss': total loss (LM loss + aux_loss) if labels provided
              'lm_loss': pure cross-entropy loss if labels provided
              'aux_loss': total load-balancing auxiliary loss from MoE layers
              'past_key_values': updated list of KV tuples if use_cache=True
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Compute position IDs
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_len = past_key_values[0][0].size(2)
            pos_ids = torch.arange(past_len, past_len + T, dtype=torch.long, device=device).unsqueeze(0)
        else:
            pos_ids = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.tok_embeddings(input_ids)
        pos_emb = self.pos_embeddings(pos_ids)
        x = self.drop(tok_emb + pos_emb)

        total_aux_loss = torch.tensor(0.0, device=device)
        total_router_z_loss = torch.tensor(0.0, device=device)
        new_past_kvs = [] if use_cache else None

        for layer_idx, layer in enumerate(self.layers):
            layer_past = past_key_values[layer_idx] if past_key_values is not None else None
            x, aux_loss, router_z_loss, new_kv = layer(
                x, past_key_value=layer_past, use_cache=use_cache,
                attention_mask=attention_mask,
            )
            total_aux_loss = total_aux_loss + aux_loss
            total_router_z_loss = total_router_z_loss + router_z_loss
            if use_cache:
                new_past_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        output: Dict[str, Any] = {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "router_z_loss": total_router_z_loss,
            "past_key_values": new_past_kvs,
        }

        if labels is not None:
            # Shift tokens for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = lm_loss + total_aux_loss + total_router_z_loss
            output["loss"] = total_loss
            output["lm_loss"] = lm_loss

        return output

    def get_parameter_stats(self) -> Dict[str, Union[int, float]]:
        """Return total parameter count and active parameter count per forward token."""
        total_params = sum(p.numel() for p in self.parameters())
        ffn_detail = self.config.count_parameters_detail()

        # Non-FFN parameters (embeddings, attention, layer norms, head)
        d_m = self.config.d_model
        attn_params_per_layer = 4 * (d_m * d_m)  # q, k, v, out
        norms_per_layer = 4 * d_m                # norm1, norm2 (weights + biases)
        non_ffn_per_layer = attn_params_per_layer + norms_per_layer

        # Base embeddings + pos embeddings + final norm
        base_params = (
            self.config.vocab_size * d_m +
            self.config.max_seq_len * d_m +
            2 * d_m
        )
        if not self.config.tie_embeddings:
            base_params += self.config.vocab_size * d_m

        active_params = (
            base_params +
            self.config.n_layers * non_ffn_per_layer +
            self.config.n_layers * ffn_detail["active_ffn_params_per_layer"]
        )

        return {
            "total_params": total_params,
            "active_params": active_params,
            "d_expert": ffn_detail["d_expert"],
            "total_ffn_per_layer": ffn_detail["total_ffn_params_per_layer"],
            "active_ffn_per_layer": ffn_detail["active_ffn_params_per_layer"],
            "active_expert_ffn_per_layer": ffn_detail["active_expert_params_per_layer"],
        }
