
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


STATE_NAMES = ("run", "charge")
FAULT_NAMES = ("normal", "insulation", "voltage", "temperature")
CLASS_NAMES = (
    "run_normal",
    "charge_normal",
    "run_insulation",
    "charge_insulation",
    "run_voltage",
    "charge_voltage",
    "run_temperature",
    "charge_temperature",
)

STATE_DESCRIPTIONS = {
    "run": (
        "行驶工况对应放电与动态负载背景，正常样本中单体电压均值可在64点窗口内轻微下降，"
        "电压差和绝缘电阻允许短时波动；故障判断应关注相对正常行驶更强的跨单体离散、"
        "绝缘通道非平稳或热通道波动。",
        "行驶状态下，短时电压扰动不应直接视为故障；更可靠的早期证据是电压差、温差或"
        "绝缘电阻在窗口内呈持续偏离、波动增强或相对其他通道不协调。",
    ),
    "charge": (
        "充电工况对应能量输入和单体电压均值上升，正常充电样本可出现平缓的电压均值增长"
        "和轻微温度变化；故障判断应区分正常充电趋势与异常的电压差扩大、绝缘通道非平稳"
        "或温差持续增长。",
        "充电状态的基线与行驶不同，应重点比较同为充电样本时的电压均值水平、上升幅度、"
        "温差演化和绝缘电阻稳定性，避免把正常充电升压误判为故障。",
    ),
}

FAULT_DESCRIPTIONS = {
    "normal": (
        "正常状态下，电压差、温差和绝缘电阻在64点窗口内总体稳定，局部扰动能够恢复；"
        "行驶正常可有负载导致的短时波动，充电正常可有电压均值缓慢上升。",
        "无故障样本的关键特征是多通道协调：电压、温度和绝缘参数没有相对同工况正常边界"
        "的持续放大、剧烈步进或跨尺度累积异常。",
    ),
    "insulation": (
        "绝缘故障的主证据是绝缘电阻通道非平稳，包括窗口内标准差、平均步进变化显著增大，"
        "或在充电工况下出现较低绝缘水平和不稳定波动；不应只依赖单点尖峰。",
        "绝缘风险常伴随与电压或温度通道不协调的变化。行驶绝缘更突出绝缘电阻强波动，"
        "充电绝缘还可能表现为电压均值偏低、正常充电升压幅度减弱。",
    ),
    "voltage": (
        "电压不一致故障的主证据是单体电压差相对同工况正常类升高，或电压均值演化与正常"
        "行驶或充电趋势不同；关键是单体间响应分离，而不是整体电压水平本身。",
        "行驶电压不一致更突出电压差增大和电压均值偏离；充电电压不一致除电压差升高外，"
        "当前数据中还常伴随温度均值上升和绝缘电阻非低值特征，这些应作为辅助边界而非"
        "绝缘退化证据。",
    ),
    "temperature": (
        "温度不一致故障的主证据是温度通道的相对偏离：温差水平、温差窗口内波动、温度均值"
        "趋势或平均步进变化相对同工况正常类增强。",
        "行驶温度类更应描述为热通道非平稳和局部热响应异常，而不只是假设温差绝对值升高；"
        "充电温度类更稳定表现为温差升高并在64点窗口内持续增长。",
    ),
}

CLASS_DESCRIPTIONS = {
    "run_normal": (
        "行驶正常：放电负载可造成短时电压差、绝缘电阻和温度通道扰动，但窗口内没有持续"
        "扩大的电压离散、绝缘强波动或温度通道非平稳。",
        "正常行驶样本的电压均值可轻微下降，多通道总体协调；异常扰动应能恢复，不形成稳定"
        "的绝缘、电压一致性或温度一致性退化模式。",
    ),
    "charge_normal": (
        "充电正常：单体电压均值随充电过程平缓上升，电压差和温差保持在同工况正常范围，"
        "绝缘电阻没有强烈步进或大幅波动。",
        "正常充电样本允许轻微升温和电压均值增长；关键边界是这些变化不应演化为持续扩大"
        "的电压差、温差或绝缘电阻非平稳。",
    ),
    "run_insulation": (
        "行驶绝缘故障：在动态放电背景下，绝缘电阻窗口内波动和平均步进变化明显强于"
        "行驶正常，可能伴随温度通道波动增强；该模式不能仅由负载扰动解释。",
        "行驶绝缘类的核心证据是绝缘通道非平稳，而不是电压差或温差绝对水平单独升高；"
        "应关注绝缘电阻相对其他通道的不协调变化。",
    ),
    "charge_insulation": (
        "充电绝缘故障：相对充电正常，绝缘电阻更低且窗口内波动或步进变化更强，表现为"
        "充电过程中的绝缘通道不稳定。",
        "充电绝缘类还常伴随单体电压均值偏低、64点窗口内升压幅度弱于正常充电；描述时"
        "应把这些作为辅助边界，主证据仍是绝缘电阻异常。",
    ),
    "run_voltage": (
        "行驶电压不一致：相对行驶正常，单体电压差显著升高，电压均值水平或窗口内演化"
        "也可偏离正常行驶放电趋势。",
        "该类应从正常负载波动中识别持续的单体离散；绝缘和温度变化可作为背景信息，"
        "但主要故障证据应落在电压差及其相对电压均值的异常。",
    ),
    "charge_voltage": (
        "充电电压不一致：相对充电正常，单体电压差更高，电压均值上升过程中的单体响应"
        "分离更明显，形成不同于正常充电趋势的一致性退化。",
        "当前数据中该类常伴随温度均值上升和高绝缘电阻水平；这些更适合作为数据边界或"
        "来源伴随特征，不应描述为绝缘电阻下降型故障。",
    ),
    "run_temperature": (
        "行驶温度不一致：相对行驶正常，温度均值更高，温度差的窗口内波动、步进变化或"
        "局部热响应更不稳定；不要只依赖温差绝对均值升高。",
        "该类的关键是热通道相对电压负载变化出现不匹配和非平稳累积，可能伴随电压差升高，"
        "但主证据应保持在温度通道。",
    ),
    "charge_temperature": (
        "充电温度不一致：相对充电正常，单体温度差更高，并在64点窗口内呈持续增长或"
        "更强线性斜率，说明局部单体升温偏离群体。",
        "充电温度类可伴随电压均值较高或温度均值趋势变化，但主要证据是温差水平、温差增长"
        "和热通道波动显著超过正常充电。",
    ),
}


@dataclass(frozen=True)
class SemanticPrototypeBundle:
    state_embeddings: torch.Tensor
    fault_embeddings: torch.Tensor
    class_embeddings: torch.Tensor
    metadata: dict[str, Any]

    @property
    def embedding_dim(self) -> int:
        return int(self.class_embeddings.shape[1])


def _prompt(entity_type: str, name: str, description: str) -> str:
    return (
        "你是新能源汽车动力电池故障机理专家。请将下面的对象理解为一个用于"
        "报警前64点早期诊断的语义概念，重点保留工况、故障机理、主要测量变量、"
        "时间演化和易混淆边界。不要给出维修建议。\n"
        f"对象类型：{entity_type}\n"
        f"对象名称：{name}\n"
        f"机理描述：{description}"
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("semantic embeddings contain NaN or Inf before normalization")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("semantic embeddings contain near-zero rows")
    normalized = values / norms
    if not np.isfinite(normalized).all():
        raise ValueError("semantic embeddings contain NaN or Inf after normalization")
    return normalized.astype(np.float32)


def _remove_common_semantic_center(
    state: np.ndarray,
    fault: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    counts = (len(state), len(fault), len(classes))
    combined = np.concatenate([state, fault, classes], axis=0)
    combined = _normalize_rows(
        combined - combined.mean(axis=0, keepdims=True)
    )
    state_end = counts[0]
    fault_end = state_end + counts[1]
    return (
        combined[:state_end],
        combined[state_end:fault_end],
        combined[fault_end:],
    )


def _pool_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    last_layers: int,
) -> torch.Tensor:
    selected = torch.stack(hidden_states[-last_layers:], dim=0).mean(dim=0)
    mask = attention_mask.unsqueeze(-1).to(selected.dtype)
    pooled = (selected * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled.float(), dim=1)


@torch.inference_mode()
def _encode_texts(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    last_layers: int,
) -> np.ndarray:
    outputs = []
    model.eval()
    target_device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        result = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        for layer_index, hidden_state in enumerate(result.hidden_states[-last_layers:]):
            if not torch.isfinite(hidden_state).all():
                raise RuntimeError(
                    "LLM produced non-finite hidden states while building "
                    f"semantic prototypes; layer offset {layer_index - last_layers}. "
                    "Try --no-4bit or another --torch-dtype."
                )
        pooled = _pool_hidden_states(
            result.hidden_states, encoded["attention_mask"], last_layers
        )
        if not torch.isfinite(pooled).all():
            raise RuntimeError(
                "LLM produced non-finite pooled embeddings while building "
                "semantic prototypes; try --no-4bit or another --torch-dtype."
            )
        outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _aggregate_descriptions(
    names: tuple[str, ...],
    descriptions: dict[str, tuple[str, ...]],
    entity_type: str,
    encode: Any,
) -> np.ndarray:
    prompts = []
    owners = []
    for index, name in enumerate(names):
        for description in descriptions[name]:
            prompts.append(_prompt(entity_type, name, description))
            owners.append(index)
    encoded = encode(prompts)
    prototypes = []
    for index in range(len(names)):
        prototypes.append(encoded[np.asarray(owners) == index].mean(axis=0))
    return _normalize_rows(np.stack(prototypes))


def build_semantic_prototypes(
    model_path: Path,
    output_path: Path,
    batch_size: int = 1,
    max_length: int = 384,
    last_layers: int = 4,
    load_in_4bit: bool = True,
    torch_dtype: torch.dtype = torch.float16,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("LLM prototype extraction requires CUDA.")

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    quantization = None
    if load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch_dtype,
    )

    def encode(texts: list[str]) -> np.ndarray:
        return _encode_texts(
            model,
            tokenizer,
            texts,
            batch_size=batch_size,
            max_length=max_length,
            last_layers=last_layers,
        )

    state_embeddings = _aggregate_descriptions(
        STATE_NAMES, STATE_DESCRIPTIONS, "operating_state", encode
    )
    fault_embeddings = _aggregate_descriptions(
        FAULT_NAMES, FAULT_DESCRIPTIONS, "fault_type", encode
    )
    class_embeddings = _aggregate_descriptions(
        CLASS_NAMES, CLASS_DESCRIPTIONS, "state_fault_combination", encode
    )
    state_embeddings, fault_embeddings, class_embeddings = (
        _remove_common_semantic_center(
            state_embeddings, fault_embeddings, class_embeddings
        )
    )
    for name, values in (
        ("state_embeddings", state_embeddings),
        ("fault_embeddings", fault_embeddings),
        ("class_embeddings", class_embeddings),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} contain NaN or Inf; semantic file not saved")
    metadata = {
        "version": "mechanism_semantic_prototype_v2",
        "encoder_model": str(model_path.resolve()),
        "pooling": f"mean_tokens_mean_last_{last_layers}_layers",
        "max_length": max_length,
        "state_names": list(STATE_NAMES),
        "fault_names": list(FAULT_NAMES),
        "class_names": list(CLASS_NAMES),
        "descriptions_per_concept": 2,
        "embedding_dim": int(class_embeddings.shape[1]),
        "postprocess": "shared_global_center_then_l2_normalize",
        "torch_dtype": str(torch_dtype).replace("torch.", ""),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        state_embeddings=state_embeddings,
        fault_embeddings=fault_embeddings,
        class_embeddings=class_embeddings,
        metadata_json=np.asarray(
            json.dumps(metadata, ensure_ascii=False), dtype=np.str_
        ),
    )
    return output_path


def load_semantic_prototypes(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> SemanticPrototypeBundle:
    path = Path(path)
    with np.load(path, allow_pickle=False) as values:
        state = np.asarray(values["state_embeddings"], dtype=np.float32)
        fault = np.asarray(values["fault_embeddings"], dtype=np.float32)
        classes = np.asarray(values["class_embeddings"], dtype=np.float32)
        metadata = json.loads(str(values["metadata_json"].item()))
    if state.shape[0] != 2 or fault.shape[0] != 4 or classes.shape[0] != 8:
        raise ValueError(f"invalid semantic prototype shapes in {path}")
    if not (state.shape[1] == fault.shape[1] == classes.shape[1]):
        raise ValueError(f"prototype embedding dimensions do not match in {path}")
    return SemanticPrototypeBundle(
        state_embeddings=torch.tensor(state, dtype=torch.float32, device=device),
        fault_embeddings=torch.tensor(fault, dtype=torch.float32, device=device),
        class_embeddings=torch.tensor(
            classes, dtype=torch.float32, device=device
        ),
        metadata=metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=r"DeepSeek-R1-8B-unsloth-4bit",
    )
    parser.add_argument(
        "--output",
        default="./semantic_prototypes_v4.npz",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--last-layers", type=int, default=4)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="Model and 4-bit compute dtype. Use float16 if bfloat16 CUDA ops fail.",
    )
    return parser.parse_args()


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


if __name__ == "__main__":
    args = parse_args()
    path = build_semantic_prototypes(
        Path(args.model_path),
        Path(args.output),
        batch_size=args.batch_size,
        max_length=args.max_length,
        last_layers=args.last_layers,
        load_in_4bit=not args.no_4bit,
        torch_dtype=_resolve_torch_dtype(args.torch_dtype),
    )
    print(f"Semantic prototypes saved: {path.resolve()}")
