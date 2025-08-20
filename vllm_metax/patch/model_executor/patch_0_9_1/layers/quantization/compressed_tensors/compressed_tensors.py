import vllm
from vllm_metax.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.model_executor.layers.quantization.compressed_tensors import compressed_tensors
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

class MetaxCompressedTensorsConfig(CompressedTensorsConfig):
    def _check_scheme_supported(self,
                            min_capability: int,
                            error: bool = True,
                            match_exact: bool = False) -> bool:
    # ┌------------------------  Metax Modification -------------------------┐
        return False
    # └------------------------- Metax Modification -------------------------┘


compressed_tensors.CompressedTensorsConfig = MetaxCompressedTensorsConfig
register_patch("vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors", "CompressedTensorsConfig", MetaxCompressedTensorsConfig)