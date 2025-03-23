import os

from transformers import LlamaConfig, LlamaForCausalLM
from transformers_neuronx import NeuronAutoModelForCausalLM, NeuronConfig, QuantizationConfig

### from jlunt - https://amazon.enterprise.slack.com/files/W01BWLKGBNK/F06EHDD0LQM/untitled.py

def model_path(dictionary, name):
    directory = f'models/{name}'
    if os.path.exists(directory):
        return directory
    config = LlamaConfig.from_dict(dictionary)
    model = LlamaForCausalLM(config)
    model = model.bfloat16()
    model.save_pretrained(directory, safe_serialization=True)
    return directory


def path_100b():
    # Reference: https://code.amazon.com/packages/LLMInferenceEngineTools/blobs/68ddcf80958ceddf8ffbecde2cb835b07c298f74/--/benchmarking/vanilla_benchmark.py#L381-L398
    config = {
        'hidden_act': "silu",
        'hidden_size': 9216,
        'initializer_range': 0.02,
        'intermediate_size': 24576,
        'max_position_embeddings': 8192,
        'num_attention_heads': 96,
        'num_hidden_layers': 96,
        'pad_token_id': 0,
        'position_interpolation_factor': 4.0,
        'rms_norm_eps': 1e-06,
        # 'tie_word_embeddings': True,
        'torch_dtype': "bfloat16",
        'use_activation_offloading': False,
        'use_cache': False,
        'use_flash_attention': False,
        'use_flash_mlp': False,
        'vocab_size': 50000
    }
    return model_path(config, 'llama-100b')


def path_7b():
    # Reference: https://code.amazon.com/packages/LLMInferenceEngineTools/blobs/68ddcf80958ceddf8ffbecde2cb835b07c298f74/--/benchmarking/vanilla_benchmark.py#L419-L436
    config = {
        'hidden_act': "silu",
        'hidden_size': 4096,
        'initializer_range': 0.02,
        'intermediate_size': 11008,
        'max_position_embeddings': 8192,
        'num_attention_heads': 32,
        'num_hidden_layers': 32,
        'pad_token_id': 0,
        'position_interpolation_factor': 4.0,
        'rms_norm_eps': 1e-06,
        # 'tie_word_embeddings': True,
        'torch_dtype': "bfloat16",
        'use_activation_offloading': False,
        'use_cache': False,
        'use_flash_attention': False,
        'use_flash_mlp': False,
        'vocab_size': 50000
    }
    return model_path(config, 'llama-7b')



def model_100b():
    """
    The target model in speculation.

    Uses 28 buckets:
    - 7 token buckets
    - 7 context buckets
    - 14 speculation buckets (2 k values * 7 buckets for each)

    Reference: https://code.amazon.com/packages/LLMInferenceEngineTools/blobs/2be5975119a38be241bef6949b7997c2372d4556/--/benchmarking/speculative_decoding_benchmark.py#L627
    """
    checkpoint = path_100b()
    os.environ['NEURONX_DUMP_TO'] = 'cache/llama-100b'
    os.environ['NEURON_INTERNAL_TRANSFORM_WEIGHT_LAYOUT'] = '1'
    os.environ['NEURON_CC_FLAGS'] = "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' --internal-max-instruction-limit=10000000 "

    neuron_config = NeuronConfig(
        cast_logits_dtype="bfloat16",
        fuse_qkv=True,
        quant=QuantizationConfig(quant_dtype='s8', dequant_dtype='bf16'),
    )
    model = NeuronAutoModelForCausalLM.from_pretrained(
        checkpoint,
        n_positions=8196, # [128, 256, 512, 1024, 2048, 4096, 8192]
        context_length_estimate=[1024, 1280, 1536, 1792, 2048, 4096, 7692],
        context_unroll=12,
        tp_degree=32,
        neuron_config=neuron_config,
        amp='bf16',
    )
    model.enable_speculative_decoder([6, 20])
    model.to_neuron()
    return model


def model_7b():
    """
    The draft model in speculation.

    Uses 11 buckets:
    - 7 token buckets
    - 4 context buckets

    Reference: https://code.amazon.com/packages/LLMInferenceEngineTools/blobs/2be5975119a38be241bef6949b7997c2372d4556/--/benchmarking/speculative_decoding_benchmark.py#L627
    """
    checkpoint = path_7b()
    os.environ['NEURONX_DUMP_TO'] = 'cache/llama-7b'
    os.environ['NEURON_CC_FLAGS'] = "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "

    neuron_config = NeuronConfig(
        cast_logits_dtype="bfloat16",
        fuse_qkv=True,
    )
    model = NeuronAutoModelForCausalLM.from_pretrained(
        checkpoint,
        n_positions=8196, # [128, 256, 512, 1024, 2048, 4096, 8192]
        context_length_estimate=[1024, 2048, 4096, 7692],
        context_unroll=4,
        tp_degree=32,
        neuron_config=neuron_config,
        amp='bf16',
    )
    model.to_neuron()
    return model
