import torch
import time
import math
import sys
import os
import shutil
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from packaging import version
import importlib.metadata
from typing import Optional, Dict, List, Union, Any, Tuple, Mapping
from torch.utils.data import DataLoader,RandomSampler
from accelerate import skip_first_batches
from transformers import Trainer
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import (
    is_torch_xla_available,
    is_sagemaker_mp_enabled,
    is_peft_available,
    is_accelerate_available,
    logging,
)
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.integrations.deepspeed import deepspeed_init,deepspeed_load_checkpoint,is_deepspeed_available
from transformers.integrations import (
    hp_params,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    TrainOutput,
    HPSearchBackend,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import (
    EvalLoopContainer,
    IterableDatasetShard,
    find_batch_size,
    get_model_param_count,
)
logger = logging.get_logger(__name__)
from icecream import ic

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        is_mlu_available,
        is_mps_available,
        is_npu_available,
        is_torch_version,
        is_xpu_available,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration

TRAINER_STATE_NAME = "trainer_state.json"

class custom_trainer(Trainer):
    """
    Exisiting Trainer is enough for handling normal tasks.
    
    This Module is prepared for futuer expansion (e.g. Custom Sampler, Overrided Evaluation Process)

    All trainer should inherit from this class in case future expansion.
    """
    def _get_train_sampler(self):
        """
        Note: If val_sampler is needed, set val_sampler
        """
        return super()._get_train_sampler()
    
    def create_optimizer(self):
        return super().create_optimizer()