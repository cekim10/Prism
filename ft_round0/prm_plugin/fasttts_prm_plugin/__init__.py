"""vLLM general plugin: registers Skywork-o1-Open-PRM's `Qwen2ForPrmModel` architecture.

vLLM 0.9.2 ships Qwen2ForRewardModel / Qwen2ForProcessRewardModel (2-layer `score` head) but not
Skywork's checkpoint layout (`v_head.summary` = Linear(hidden, 1), one scalar per token). Loaded
automatically in every vLLM process through the `vllm.general_plugins` entry point."""


def register():
    from vllm import ModelRegistry
    if "Qwen2ForPrmModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen2ForPrmModel", "fasttts_prm_plugin.model:Qwen2ForPrmModel")
