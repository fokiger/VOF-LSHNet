

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_TO_STATE = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
CLASS_TO_FAULT = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
CLASS_TO_NORMAL_FAULT = torch.tensor(
    [0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long
)


@dataclass
class PredictorConfig:
    n_features: int = 5
    n_classes: int = 8
    seq_len: int = 64
    wave_feature_dim: int = 160
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 192
    dropout: float = 0.15
    semantic_dim: int = 4096
    use_semantic_prototypes: bool = True
    use_class_semantic_prototypes: bool = True
    use_compositional_prototypes: bool = True
    semantic_hierarchy_scale: float = 0.5
    initial_temperature: float = 0.07
    use_raw_branch: bool = True
    use_wave_branch: bool = True
    use_normal_fault_head: bool = False
    normal_fault_hierarchy_scale: float = 0.5
    use_state_condition: bool = False
    use_state_logit_mask: bool = False
    class_to_state: tuple[int, ...] = (0, 1, 0, 1, 0, 1, 0, 1)
    class_to_fault: tuple[int, ...] = (0, 0, 1, 1, 2, 2, 3, 3)
    class_to_normal_fault: tuple[int, ...] = (0, 0, 1, 1, 1, 1, 1, 1)
    state_logit_run_class_ids: tuple[int, ...] = (0, 2, 4, 6)
    state_logit_charge_class_ids: tuple[int, ...] = (1, 3, 5, 7)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        divisor = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class MultiScaleTemporalStem(nn.Module):
    def __init__(self, n_features: int, d_model: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Conv1d(n_features, d_model, kernel_size=1)
        branch_dim = d_model // 2
        self.short_branch = nn.Sequential(
            nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model
            ),
            nn.Conv1d(d_model, branch_dim, kernel_size=1),
            nn.GELU(),
        )
        self.long_branch = nn.Sequential(
            nn.Conv1d(
                d_model, d_model, kernel_size=7, padding=3, groups=d_model
            ),
            nn.Conv1d(d_model, branch_dim, kernel_size=1),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Conv1d(branch_dim * 2, d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x.transpose(1, 2))
        return self.output(
            torch.cat([self.short_branch(x), self.long_branch(x)], dim=1)
        ).transpose(1, 2)


class HierarchicalCrossGatedPredictor(nn.Module):


    def __init__(
        self,
        config: PredictorConfig,
        state_prototypes: torch.Tensor | None = None,
        fault_prototypes: torch.Tensor | None = None,
        class_prototypes: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self.class_to_state_values = tuple(int(x) for x in config.class_to_state)
        self.class_to_fault_values = tuple(int(x) for x in config.class_to_fault)
        self.class_to_normal_fault_values = tuple(
            int(x) for x in config.class_to_normal_fault
        )
        for name, values in (
            ("class_to_state", self.class_to_state_values),
            ("class_to_fault", self.class_to_fault_values),
            ("class_to_normal_fault", self.class_to_normal_fault_values),
        ):
            if len(values) != config.n_classes:
                raise ValueError(
                    f"{name} must contain {config.n_classes} entries, "
                    f"got {len(values)}"
                )
            if min(values) < 0:
                raise ValueError(f"{name} contains a negative class mapping")
        self.n_states = max(self.class_to_state_values) + 1
        self.n_faults = max(self.class_to_fault_values) + 1
        self.n_normal_fault = max(self.class_to_normal_fault_values) + 1
        if self.n_normal_fault != 2:
            raise ValueError("normal/fault hierarchy must contain 2 groups")
        for class_id in (
            *config.state_logit_run_class_ids,
            *config.state_logit_charge_class_ids,
        ):
            if class_id < 0 or class_id >= config.n_classes:
                raise ValueError(
                    f"state logit mask class id {class_id} is out of range "
                    f"for n_classes={config.n_classes}"
                )
        if not (config.use_raw_branch or config.use_wave_branch):
            raise ValueError("at least one signal branch must be enabled")
        if config.use_raw_branch:
            self.temporal_stem = MultiScaleTemporalStem(
                config.n_features, config.d_model, config.dropout
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            nn.init.normal_(self.cls_token, std=0.02)
            self.position = SinusoidalPositionEncoding(
                config.d_model, config.seq_len + 1
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.n_layers
            )
        if config.use_wave_branch:
            self.wave_encoder = nn.Sequential(
                nn.LayerNorm(config.wave_feature_dim),
                nn.Linear(config.wave_feature_dim, config.dim_feedforward),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_feedforward, config.d_model),
            )
        if config.use_raw_branch and config.use_wave_branch:
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.Sigmoid(),
            )
        self.fusion_norm = nn.LayerNorm(config.d_model)
        if config.use_state_condition:
            self.state_condition_embedding = nn.Embedding(
                self.n_states, config.d_model
            )
            self.state_condition_norm = nn.LayerNorm(config.d_model)
        self.register_buffer(
            "class_to_state",
            torch.tensor(self.class_to_state_values, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "class_to_fault",
            torch.tensor(self.class_to_fault_values, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "class_to_normal_fault",
            torch.tensor(
                self.class_to_normal_fault_values, dtype=torch.long
            ),
            persistent=False,
        )
        if config.use_normal_fault_head:
            self.normal_fault_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, 2),
            )
        if config.use_semantic_prototypes:
            if not (
                config.use_class_semantic_prototypes
                or config.use_compositional_prototypes
            ):
                raise ValueError(
                    "semantic prototypes require class, compositional, "
                    "or both semantic sources"
                )
            expected = (
                (state_prototypes, self.n_states, "state"),
                (fault_prototypes, self.n_faults, "fault"),
                (class_prototypes, config.n_classes, "class"),
            )
            for values, rows, name in expected:
                if values is None:
                    raise ValueError(f"{name} semantic prototypes are required")
                if values.shape != (rows, config.semantic_dim):
                    raise ValueError(
                        f"{name} prototypes must have shape "
                        f"({rows}, {config.semantic_dim}), got {tuple(values.shape)}"
                    )
            self.register_buffer(
                "state_semantic_prototypes",
                state_prototypes.detach().float().clone(),
            )
            self.register_buffer(
                "fault_semantic_prototypes",
                fault_prototypes.detach().float().clone(),
            )
            self.register_buffer(
                "class_semantic_prototypes",
                class_prototypes.detach().float().clone(),
            )
            self.signal_projection = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            self.semantic_projection = self._prototype_projection(config)
            if config.use_compositional_prototypes:
                self.composition_interaction = nn.Sequential(
                    nn.Linear(config.d_model * 2, config.d_model),
                    nn.GELU(),
                    nn.Linear(config.d_model, config.d_model),
                )
            initial_scale = torch.log(
                torch.tensor(1.0 / config.initial_temperature)
            )
            self.logit_scale = nn.Parameter(initial_scale.clone())
            self.state_logit_scale = nn.Parameter(initial_scale.clone())
            self.fault_logit_scale = nn.Parameter(initial_scale.clone())
        else:
            self.state_head = nn.Linear(config.d_model, self.n_states)
            self.fault_head = nn.Linear(config.d_model, self.n_faults)
            self.joint_residual_head = nn.Linear(
                config.d_model, config.n_classes
            )

    @staticmethod
    def _prototype_projection(config: PredictorConfig) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(config.semantic_dim),
            nn.Linear(config.semantic_dim, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.d_model),
        )

    @staticmethod
    def _scaled_similarity(
        signal: torch.Tensor,
        prototypes: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        similarity = F.normalize(signal, dim=1) @ F.normalize(
            prototypes, dim=1
        ).transpose(0, 1)
        scale = logit_scale.exp().clamp(max=100.0)
        return scale * similarity, similarity

    def _apply_state_logit_mask(
        self,
        logits: torch.Tensor,
        operating_state: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.config.use_state_logit_mask or operating_state is None:
            return logits
        state = operating_state.to(device=logits.device, dtype=torch.long)
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        run_ids = list(self.config.state_logit_run_class_ids)
        charge_ids = list(self.config.state_logit_charge_class_ids)
        if run_ids:
            allowed[:, run_ids] = (state == 0).unsqueeze(1)
        if charge_ids:
            allowed[:, charge_ids] = (state == 1).unsqueeze(1)
        return logits.masked_fill(~allowed, -1e4)

    def _semantic_forward(
        self,
        fused: torch.Tensor,
        normal_fault_logits: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        signal = self.signal_projection(fused)
        state = self.semantic_projection(self.state_semantic_prototypes)
        fault = self.semantic_projection(self.fault_semantic_prototypes)
        class_base = self.semantic_projection(self.class_semantic_prototypes)
        class_component = (
            class_base
            if self.config.use_class_semantic_prototypes
            else torch.zeros_like(class_base)
        )
        state_by_class = state[self.class_to_state]
        fault_by_class = fault[self.class_to_fault]
        if self.config.use_compositional_prototypes:
            interaction = self.composition_interaction(
                torch.cat([state_by_class, fault_by_class], dim=1)
            )
            composed = state_by_class + fault_by_class + interaction
            composed_component = composed
            joint_prototypes = F.normalize(
                class_component + composed_component, dim=1
            )
        else:
            composed = state_by_class + fault_by_class
            composed_component = torch.zeros_like(class_component)
            joint_prototypes = F.normalize(class_component, dim=1)

        prototype_logits, prototype_similarity = self._scaled_similarity(
            signal, joint_prototypes, self.logit_scale
        )
        state_logits, _ = self._scaled_similarity(
            signal, state, self.state_logit_scale
        )
        fault_logits, _ = self._scaled_similarity(
            signal, fault, self.fault_logit_scale
        )
        hierarchy = (
            state_logits[:, self.class_to_state]
            + fault_logits[:, self.class_to_fault]
        )
        semantic_hierarchy_logits = (
            self.config.semantic_hierarchy_scale * hierarchy
        )
        joint_logits = (
            prototype_logits
            + semantic_hierarchy_logits
        )
        if normal_fault_logits is not None:
            joint_logits = joint_logits + (
                self.config.normal_fault_hierarchy_scale
                * normal_fault_logits[:, self.class_to_normal_fault]
            )
        composition_similarity = F.cosine_similarity(
            F.normalize(class_base, dim=1),
            F.normalize(composed, dim=1),
            dim=1,
        )
        class_norm = class_component.norm(dim=1)
        composed_norm = composed_component.norm(dim=1)
        component_norm = class_norm + composed_norm + 1e-8
        return {
            "logits": joint_logits,
            "prototype_logits": prototype_logits,
            "prototype_similarity": prototype_similarity,
            "semantic_hierarchy_logits": semantic_hierarchy_logits,
            "state_logits": state_logits,
            "fault_logits": fault_logits,
            "composition_similarity": composition_similarity,
            "class_semantic_weight": class_norm / component_norm,
            "composed_semantic_weight": composed_norm / component_norm,
            "semantic_signal": F.normalize(signal, dim=1),
        }

    def forward(
        self,
        raw_sequence: torch.Tensor,
        wave_features: torch.Tensor,
        operating_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = raw_sequence.size(0)
        if self.config.use_raw_branch:
            temporal = self.temporal_stem(raw_sequence)
            cls = self.cls_token.expand(batch_size, -1, -1)
            temporal = self.temporal_encoder(
                self.position(torch.cat([cls, temporal], dim=1))
            )[:, 0]
        else:
            temporal = raw_sequence.new_zeros(
                batch_size, self.config.d_model
            )
        if self.config.use_wave_branch:
            wave = self.wave_encoder(wave_features)
        else:
            wave = raw_sequence.new_zeros(
                batch_size, self.config.d_model
            )
        if self.config.use_raw_branch and self.config.use_wave_branch:
            gate = self.fusion_gate(torch.cat([temporal, wave], dim=1))
        elif self.config.use_raw_branch:
            gate = raw_sequence.new_ones(batch_size, self.config.d_model)
        else:
            gate = raw_sequence.new_zeros(batch_size, self.config.d_model)
        fused = self.fusion_norm(gate * temporal + (1.0 - gate) * wave)
        if self.config.use_state_condition and operating_state is not None:
            state_condition = self.state_condition_embedding(
                operating_state.to(device=fused.device, dtype=torch.long)
            )
            fused = self.state_condition_norm(fused + state_condition)
        normal_fault_logits = (
            self.normal_fault_head(fused)
            if self.config.use_normal_fault_head
            else None
        )

        if self.config.use_semantic_prototypes:
            output = self._semantic_forward(fused, normal_fault_logits)
        else:
            state_logits = self.state_head(fused)
            fault_logits = self.fault_head(fused)
            hierarchy = (
                state_logits[:, self.class_to_state]
                + fault_logits[:, self.class_to_fault]
            )
            if normal_fault_logits is not None:
                hierarchy = hierarchy + (
                    self.config.normal_fault_hierarchy_scale
                    * normal_fault_logits[:, self.class_to_normal_fault]
                )
            output = {
                "logits": self.joint_residual_head(fused) + hierarchy,
                "state_logits": state_logits,
                "fault_logits": fault_logits,
            }
        state_logit_mask_applied = (
            self.config.use_state_logit_mask and operating_state is not None
        )
        output["logits"] = self._apply_state_logit_mask(
            output["logits"], operating_state
        )
        output["state_logit_mask_applied"] = state_logit_mask_applied
        if normal_fault_logits is not None:
            output["normal_fault_logits"] = normal_fault_logits
        output["fusion_gate"] = gate
        output["fused_features"] = fused
        return output


class HierarchicalConsistencyLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        normal_fault_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.03,
        state_weight: float = 0.20,
        fault_weight: float = 0.35,
        normal_fault_weight: float = 0.35,
        grouped_normal_fault_weight: float = 0.20,
        consistency_weight: float = 0.10,
        alignment_weight: float = 0.25,
        structured_margin_weight: float = 0.10,
        composition_weight: float = 0.05,
        structured_margin: float = 0.10,
        class_to_state: tuple[int, ...] | torch.Tensor = CLASS_TO_STATE,
        class_to_fault: tuple[int, ...] | torch.Tensor = CLASS_TO_FAULT,
        class_to_normal_fault: tuple[int, ...]
        | torch.Tensor = CLASS_TO_NORMAL_FAULT,
    ):
        super().__init__()
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else None,
        )
        self.register_buffer(
            "normal_fault_weights",
            normal_fault_weights if normal_fault_weights is not None else None,
        )
        self.label_smoothing = label_smoothing
        self.state_weight = state_weight
        self.fault_weight = fault_weight
        self.normal_fault_weight = normal_fault_weight
        self.grouped_normal_fault_weight = grouped_normal_fault_weight
        self.consistency_weight = consistency_weight
        self.alignment_weight = alignment_weight
        self.structured_margin_weight = structured_margin_weight
        self.composition_weight = composition_weight
        self.structured_margin = structured_margin
        self.register_buffer(
            "class_to_state",
            torch.as_tensor(class_to_state, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "class_to_fault",
            torch.as_tensor(class_to_fault, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "class_to_normal_fault",
            torch.as_tensor(class_to_normal_fault, dtype=torch.long),
            persistent=False,
        )
        self.n_states = int(self.class_to_state.max().item()) + 1
        self.n_faults = int(self.class_to_fault.max().item()) + 1

    def forward(
        self, output: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = labels.device
        class_to_state = self.class_to_state.to(device)
        class_to_fault = self.class_to_fault.to(device)
        class_to_normal_fault = self.class_to_normal_fault.to(device)
        state_labels = class_to_state[labels]
        fault_labels = class_to_fault[labels]
        normal_fault_labels = class_to_normal_fault[labels]
        joint_label_smoothing = (
            0.0
            if output.get("state_logit_mask_applied", False)
            else self.label_smoothing
        )
        joint_loss = F.cross_entropy(
            output["logits"],
            labels,
            weight=self.class_weights,
            label_smoothing=joint_label_smoothing,
        )
        state_loss = F.cross_entropy(output["state_logits"], state_labels)
        fault_loss = F.cross_entropy(output["fault_logits"], fault_labels)
        normal_fault_loss = joint_loss.new_tensor(0.0)
        if "normal_fault_logits" in output:
            normal_fault_loss = F.cross_entropy(
                output["normal_fault_logits"],
                normal_fault_labels,
                weight=self.normal_fault_weights,
            )

        joint_probability = torch.softmax(output["logits"], dim=1)
        grouped_normal_fault_probability = torch.stack(
            [
                joint_probability[
                    :, class_to_normal_fault == normal_fault
                ].sum(dim=1)
                for normal_fault in range(2)
            ],
            dim=1,
        )
        grouped_normal_fault_loss = F.nll_loss(
            grouped_normal_fault_probability.clamp_min(1e-8).log(),
            normal_fault_labels,
            weight=self.normal_fault_weights,
        )
        state_marginal = torch.stack(
            [
                joint_probability[:, class_to_state == state].sum(dim=1)
                for state in range(self.n_states)
            ],
            dim=1,
        )
        fault_marginal = torch.stack(
            [
                joint_probability[:, class_to_fault == fault].sum(dim=1)
                for fault in range(self.n_faults)
            ],
            dim=1,
        )
        consistency = F.mse_loss(
            state_marginal, torch.softmax(output["state_logits"], dim=1)
        ) + F.mse_loss(
            fault_marginal, torch.softmax(output["fault_logits"], dim=1)
        )
        if "normal_fault_logits" in output:
            consistency = consistency + F.mse_loss(
                grouped_normal_fault_probability,
                torch.softmax(output["normal_fault_logits"], dim=1),
            )
        alignment = joint_loss.new_tensor(0.0)
        structured_margin = joint_loss.new_tensor(0.0)
        composition = joint_loss.new_tensor(0.0)
        if "prototype_logits" in output:
            alignment = F.cross_entropy(
                output["prototype_logits"],
                labels,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            similarity = output["prototype_similarity"]
            positive = similarity.gather(1, labels[:, None]).squeeze(1)
            same_state = (
                class_to_state[None, :]
                == class_to_state[labels][:, None]
            )
            same_fault = (
                class_to_fault[None, :]
                == class_to_fault[labels][:, None]
            )
            structured_negative = torch.logical_xor(same_state, same_fault)
            structured_negative.scatter_(1, labels[:, None], False)
            negative = similarity.masked_fill(
                ~structured_negative, float("-inf")
            ).amax(dim=1)
            structured_margin = F.relu(
                self.structured_margin + negative - positive
            ).mean()
            composition = (
                1.0 - output["composition_similarity"]
            ).mean()
        total = (
            joint_loss
            + self.state_weight * state_loss
            + self.fault_weight * fault_loss
            + self.normal_fault_weight * normal_fault_loss
            + self.grouped_normal_fault_weight * grouped_normal_fault_loss
            + self.consistency_weight * consistency
            + self.alignment_weight * alignment
            + self.structured_margin_weight * structured_margin
            + self.composition_weight * composition
        )
        return total, {
            "joint": float(joint_loss.detach()),
            "state": float(state_loss.detach()),
            "fault": float(fault_loss.detach()),
            "normal_fault": float(normal_fault_loss.detach()),
            "grouped_normal_fault": float(
                grouped_normal_fault_loss.detach()
            ),
            "consistency": float(consistency.detach()),
            "alignment": float(alignment.detach()),
            "structured_margin": float(structured_margin.detach()),
            "composition": float(composition.detach()),
        }
