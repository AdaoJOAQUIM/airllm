from sys import platform

is_on_mac_os = False

if platform == "darwin":
    is_on_mac_os = True

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .auto_model import AutoModel
else:
    from .airllm import AirLLMLlama2
    from .airllm_chatglm import AirLLMChatGLM
    from .airllm_qwen import AirLLMQWen
    from .airllm_qwen2 import AirLLMQWen2
    from .airllm_baichuan import AirLLMBaichuan
    from .airllm_internlm import AirLLMInternLM
    from .airllm_mistral import AirLLMMistral
    from .airllm_mixtral import AirLLMMixtral
    from .airllm_base import AirLLMBaseModel
    from .auto_model import AutoModel
    from .utils import split_and_save_layers
    from .utils import NotEnoughSpaceException
    from .utils import compress_layer_state_dict, uncompress_layer_state_dict

from .lossless import (compress_tensor_lossless, decompress_tensor_lossless,
                       compress_state_dict_lossless, decompress_state_dict_lossless,
                       is_lossless_compressed, compression_report)
from .streaming import TensorStreamer, streamed_linear

