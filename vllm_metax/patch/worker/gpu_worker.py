from types import NoneType
from typing import TYPE_CHECKING, Any
import vllm
import torch
import triton
import triton.language as tl
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput
#--------------------------------------
#Add a tag kernel for decoding
#--------------------------------------
@torch.inference_mode()
def execute_model(
    self, scheduler_output: "SchedulerOutput"
) -> ModelRunnerOutput | None:
    intermediate_tensors = None
    forward_pass = scheduler_output.total_num_scheduled_tokens > 0
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    num_input_tokens = self.model_runner._get_num_input_tokens(num_scheduled_tokens)
    all_gather_tensors = {
        "residual": not is_residual_scattered_for_sp(
            self.vllm_config, num_input_tokens
        )
    }
    if forward_pass and not get_pp_group().is_first_rank:
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )
        )
    
    if(len(scheduler_output.scheduled_new_reqs) < 1 and  len(scheduler_output.scheduled_cached_reqs.req_ids) > 0) :
        @triton.jit
        def DECODE_TAG():
            pass
        DECODE_TAG[(1,)]()

    with self.annotate_profile(scheduler_output):
        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors
        )
        if isinstance(output, (ModelRunnerOutput, NoneType)):
            return output

    assert isinstance(output, IntermediateTensors)
    parallel_config = self.vllm_config.parallel_config
    assert (
        parallel_config.distributed_executor_backend != "external_launcher"
        and not get_pp_group().is_last_rank
    )

    get_pp_group().send_tensor_dict(
        output.tensors,
        all_gather_group=get_tp_group(),
        all_gather_tensors=all_gather_tensors,
    )

    return None
from vllm.v1.worker.gpu_worker import Worker
Worker.execute_model = execute_model