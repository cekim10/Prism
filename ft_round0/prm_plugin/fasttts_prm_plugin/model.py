from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput


class ValueHead(nn.Module):
    """Mirrors trl-style ValueHead used by Skywork: dropout (identity at inference) + Linear(hidden, 1)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.summary = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.summary(hidden_states.to(self.summary.weight.dtype))


class Qwen2ForPrmModel(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.v_head = ValueHead(config.hidden_size)
        self._pooler = Pooler.from_config_with_defaults(
            vllm_config.model_config.pooler_config,
            pooling_type=PoolingType.ALL, normalize=False, softmax=False)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states
        return self.v_head(hidden_states)  # [num_tokens, 1]; FastTTS applies sigmoid to [i][0]

    def pooler(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."])
        return loader.load_weights(weights)
