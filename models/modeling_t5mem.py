####///////////////////////   6 update memory and feed memory to the decoder   \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# coding=utf-8
# Copyright 2022 Mesh TensorFlow authors, T5MEM Author and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5MemModelodel."""


import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint  # TODO: modify

import einops

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
    ModelOutput,
)


from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

# from transformers.models.t5mem.configuration_t5mem import T5MemConfig  # TODO: modified
from .configuration_t5mem import T5MemConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5MemConfig"
_TOKENIZER_FOR_DOC = "T5MemTokenizer"
_CHECKPOINT_FOR_DOC = "t5-small"
T5Mem_START_DOCSTRING = r"""

    The T5Mem model for summarization task was proposed in [SAMDIT: Systematic Study of Adding Memory to Divided Input 
    in the Transformer to Process Long Documents](https://link.springer.com/chapter/10.1007/978-3-031-44865-2_10) by 
    Arij Al Adel. It's an encoder decoder transformer based on pre-trained in a
    text-to-text denoising generative setting T5 from huggingface.

    Parameters:
        config ([`T5MemConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. 
"""


@dataclass
class EncoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.

        last_memory_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    """

    last_hidden_state: torch.FloatTensor = None
    last_memory_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MemSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_last_memory_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_logits: torch.FloatTensor = None
    encoder_memory_logits: torch.FloatTensor = None


####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5mem-small",
    "t5mem-base",
    # "t5mem-base",
    # "t5mem-large",
    # "t5mem-3b",
    # "t5mem-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# ?????
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]
            elif scope_names[0] == "enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]
            elif scope_names[0] == "dense_relu_dense":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]
            elif scope_names[0] == "rms_norm":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")
            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue
            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif (
                scope_names[0] == "wi"
                and len(scope_names) > 1
                and scope_names[1].isdigit()
            ):
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info(f"Transposing numpy weight of shape {array.shape} for {name}")
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of nn.Module)
####################################################
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.
    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.
    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:
                - t5-small: 6
                - t5-base: 12
                - t5-large: 24
                - t5-3b: 24
                - t5-11b: 24
    Example:
    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.
    Example:
    ```python
    # On a 4 GPU machine with t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


#####################  Added Functions###########################


def _pad_to_multiple(
    x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0
) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = nn.functional.pad(x, pad=pad, mode="constant", value=pad_value)

    return x


def _make_mem_seq_relative_position_ids(
    num_slots: int, mem_size: int, block_size: int
) -> torch.Tensor:  # modified from original longt5
    """Create the relative position tensor for local -> global attention."""
    slot_positions_ids = get_mem(num_slots, mem_size)
    mem_realtive_positions = slot_positions_ids - slot_positions_ids[..., None]
    seq_side_ids = (
        torch.arange(block_size) + 1
    )  # position of segments tokens from the related memory
    seq_side_ids = seq_side_ids.repeat((mem_realtive_positions.shape[1], 1))
    mem_total_relative_position = torch.cat(
        [mem_realtive_positions, seq_side_ids], dim=1
    )  # global_positions - block_ids[..., None]
    return mem_total_relative_position.type(torch.int64)


def get_mem(num_block, num_mem):

    MEMS = torch.zeros(num_block * num_mem)
    for i in range(num_mem):
        MEMS[i::num_mem] = torch.arange(num_block)  # dim 2i

    return MEMS


# TODO
def _make_side_relative_position_ids(
    attention_mask: torch.Tensor, global_block_size: int, num_mem: int
) -> torch.Tensor:  # oroginal longt5
    """Create the relative position tensor for local -> global attention."""
    block_ids, global_segment_ids = _make_global_fixed_block_ids(
        attention_mask, global_block_size, num_mem
    )
    block_ids, global_segment_ids = block_ids.to(
        attention_mask.device
    ), global_segment_ids.to(attention_mask.device)
    global_seq_len = global_segment_ids.shape[-1]
    global_positions = torch.arange(global_seq_len, device=block_ids.device)
    side_relative_position = global_positions - block_ids[..., None]
    return side_relative_position.type(torch.int64)


# clean 3. function to segment an input to a number of segments using segment size _split_into_blocks
def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


# Clean function for my model 2. function to segment a long attention input  _make_global_fixed_block_ids
def _make_global_fixed_block_ids(
    attention_mask: torch.Tensor, block_size: int, num_mem: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the "fixed block" global id corresponding to each input token.
    This implementation is a simlified version of the original Flaxformr implementation adopted from:
    https://github.com/google/flaxformer/blob/main/flaxformer/architectures/longt5/long_attention.py.
    In our scenario, as we use this strategy only for a decoder, orphan tokens, i.e. those tokens which do not make for
    the whole fixed block, are assigned to the preceding block.
    Padding tokens from the original sequence are represented by -1.
    original code from longt5 https://github.com/huggingface/transformers/blob/a6b77598805f4e3c24a47767d503dc6ea20d1381/src/transformers/models/longt5/modeling_longt5.py#L894
    """
    batch_size, seq_len = attention_mask.shape[:2]
    fixed_block_mask = (
        torch.ones_like(attention_mask, device=attention_mask.device) / block_size
    )
    fixed_block_mask = torch.cumsum(fixed_block_mask, axis=1) - fixed_block_mask
    mask = torch.where(attention_mask != 0.0, 1.0, -1000.0).type(attention_mask.dtype)
    global_block_ids = torch.floor(mask + fixed_block_mask - 1.0).type(
        attention_mask.dtype
    )
    _global_block_ids_lower_bound = torch.tensor(
        -1.0, dtype=global_block_ids.dtype, device=global_block_ids.device
    )
    global_block_ids = torch.where(
        global_block_ids > _global_block_ids_lower_bound,
        global_block_ids,
        _global_block_ids_lower_bound,
    )

    # set padding tokens to -1
    global_block_ids = (global_block_ids * attention_mask) + (attention_mask - 1)
    # [batch_size, seq_len]
    # global_block_ids = handle_orphan_tokens(global_block_ids)
    if seq_len % block_size == 0:
        num_globals = seq_len // block_size  # represent the number of memory slots
    else:
        num_globals = seq_len // block_size + 1

    global_segment_ids = torch.ones(num_globals * num_mem).unsqueeze(dim=0)
    return global_block_ids.type(torch.int), global_segment_ids.type(torch.int)


#################################################################


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


try:
    from apex.normalization import FusedRMSNorm

    T5LayerNorm = FusedRMSNorm  # noqa

    logger.info(
        "Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm"
    )
except ImportError:
    # using the normal T5LayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass

ALL_LAYERNORM_LAYERS.append(T5LayerNorm)


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5MemConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5MemConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5MemConfig):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5MemConfig, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5MemAttention(nn.Module):
    def __init__(
        self, config: T5MemConfig, has_relative_attention_bias=False
    ):  # (self, config: T5MemConfig, num_blocks: int,  has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder

        # relative position parameters
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance

        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # memory parameters TODO: rename word block
        self.block_size = config.block_size
        self.num_mem = config.num_mem
        # self.num_blocks = num_blocks

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        ######################## TODO: Modified
        if self.has_relative_attention_bias:
            # Seq_Seq positional embedding
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )  # se_seq_embedding seq = rows= query, seq = columns = key used in  compute_bias function
            # seq_mem positional embedding
            self.seq_mem_relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )  # seq=rows= query, mem column=key  used in compute_side_bias function
            # mem_total positional embedding; total =[mem;seq]
            self.mem_total_relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )  # me =rows = query, total = [mem:seq] = columns = keys used in  compute_mem_total_bias  function

        self.pruned_heads = set()
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_postion_if_large
        )
        return relative_buckets

    def compute_bias(
        self, block_size, num_blocks, device=None
    ):  # used for seq_seq bias
        """Compute binned relative position bias"""
        context_position = torch.arange(block_size, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(block_size, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  ## shape (1,num_heads, query_length, key_length)  #  (1, 1, num_heads, block_length, 3 * block_length) longt5
        values = values.unsqueeze(0).expand(
            -1, num_blocks, -1, -1, -1
        )  # (1, num_block, num_heads, query_length, key_length)
        return values

    # not clean yet : imperical  seq_mem_attention_side_bias
    def compute_side_bias(
        self, mask: torch.Tensor, block_ids: int, global_segment_ids: int
    ) -> torch.Tensor:  # To compute the bias between memory slots and the seqments
        """
        compute masked   seq_mem bias
        """
        # (batch_size, seq_len, global_seq_len)
        side_relative_position = _make_side_relative_position_ids(
            mask, self.block_size, self.num_mem
        )

        side_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # (batch_size, seq_len, global_seq_len, num_heads)
        side_bias = self.seq_mem_relative_attention_bias(side_relative_position_bucket)

        # (batch_size, num_heads, seq_len, global_seq_len)
        side_bias = side_bias.permute([0, 3, 1, 2])

        # (batch_size, 1, seq_len, global_seq_len)
        side_attention_mask = torch.eq(mask[..., None], global_segment_ids[:, None, :])[
            :, None, ...
        ]
        attention_side_bias = torch.where(
            side_attention_mask > 0, 0.0, torch.finfo(torch.float).min
        )  # this is the mask

        # (batch_size, num_heads, seq_len, global_seq_len)
        attention_side_bias = attention_side_bias + side_bias

        return attention_side_bias

    ################### TODO: Compute masked mem_seq bias

    def compute_mem_total_bias(
        self, mask: torch.Tensor, block_size: int, mem_size: int, num_slots: int
    ) -> torch.Tensor:  # in my model
        """
        new verion
        Compute masked mem_total bias
        TODO: change global_segment_ids into slot_ids
        """
        # (batch_size, mem_len, global_seq_len)
        side_relative_position = _make_mem_seq_relative_position_ids(
            num_slots, mem_size, block_size
        ).unsqueeze(0)
        side_relative_position = einops.repeat(
            side_relative_position, "b m n -> b k m  n", k=num_slots
        ).to(mask.device)

        mem_seq_relative_position_bucket = self._relative_position_bucket(
            side_relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        # (batch_size, num_blocks, slot_len, slot_len*num_mem + global_seq_len, num_heads)

        mem_seq_bias = self.mem_total_relative_attention_bias(
            mem_seq_relative_position_bucket
        )

        # (batch_size, num_heads, slot_length, slot_length + seq_len)
        mem_seq_bias = mem_seq_bias.permute([0, 1, 4, 2, 3])
        mask = _split_into_blocks(mask, block_len=block_size, dim=1)
        mask = mask.repeat_interleave(mem_size, dim=-2)
        # from https://stackoverflow.com/questions/54856333/pytorch-set-block-diagonal-matrix-efficiently
        slots_mask = torch.ones(num_slots, mem_size, mem_size, dtype=int)
        slots_mask = torch.block_diag(*slots_mask).to(mask.device)
        # print("slots_mask = torch.block_diag(*slots_mask):  ", slots_mask.shape)
        # print(slots_mask)
        # print("slots_mask, mask shape", slots_mask.shape, mask.shape)
        # batch_size
        list_mem_seq_attention_mask = []
        for i in range(num_slots):
            x = torch.einsum("n,bnm->bnm", slots_mask[i * mem_size], mask)
            slot_mask = slots_mask.clone()
            # print("1: slot_mask:::::::::::::::::::::::::::::::::::::::::::::::::::::: ")
            # print(slot_mask)
            slot_mask[(i + 1) * mem_size :, :] = slot_mask[(i + 1) * mem_size :, :] * 0
            if i > 0:
                slot_mask[0 : i * mem_size, :] = slot_mask[0 : i * mem_size, :] * 0

            # print("2: slot_mask:::::::::::::::::::::::::::::::::::::::::::::::::::::: ")
            # print(slot_mask)
            x = torch.cat([slot_mask.unsqueeze(0), x], dim=-1)[:, None, ...]
            list_mem_seq_attention_mask.append(x)

        mem_seq_attention_mask = torch.cat(list_mem_seq_attention_mask, dim=1)  #
        mem_seq_attention_side_bias = torch.where(
            mem_seq_attention_mask > 0, 0.0, torch.finfo(torch.float).min
        )  # this is the mask

        # (batch_size, num_heads, mem_len, global_seq_len)
        # print("mem_seq_attention_side_bias min: ", mem_seq_attention_side_bias.min())
        # print("mem_seq_bias max: ", mem_seq_bias.max())
        mem_seq_attention_side_bias = (
            mem_seq_attention_side_bias[:, :, None, ...] + mem_seq_bias
        )
        # print("mem_seq_attention_side_bias min: ", mem_seq_attention_side_bias.min())

        # mem_seq_attention_side_bias = mem_seq_attention_side_bias.unsqueeze(2).expand(-1, -1, self.config.num_heads,-1,-1)

        return mem_seq_attention_side_bias.to(mask.device), mem_seq_attention_mask[
            :, :, None, ...
        ].to(mask.device)

    def forward(
        self,
        hidden_states,  # Chunked input
        mask=None,  # The mask still the same without changing from model input
        # key_value_states=None,
        position_bias=None,
        mem_total_attention_mask=None,
        # past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        hidden_states [memslots:seq]
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = mask.shape[:2]
        # get memory states
        if seq_length % self.block_size == 0:
            self.num_slots = seq_length // self.block_size
        else:
            self.num_slots = seq_length // self.block_size + 1

        def shape(states):
            """projection"""
            # (batch_size, num_blocks, block_size, n_heads, dim_per_head)
            return states.view(
                batch_size, self.num_slots, -1, self.n_heads, self.key_value_proj_dim
            )

        def unshape(states):
            """reshape"""
            # (batch_size, num_blocks, block_size, inner_dim)
            return states.contiguous().view(
                batch_size, self.num_slots, -1, self.inner_dim
            )

        def project(hidden_states, proj_layer):
            """projects hidden states correctly to key/query states"""

            hidden_states = shape(proj_layer(hidden_states))
            return hidden_states

        # get query states batch_size, seq_len, d_model

        # (batch_size, num_blocks, block_size+num_mem, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))

        # get key/value states
        # (batch_size, num_blocks, block_size+num_mem*num_slots, n_heads, dim_per_head)
        key_states = project(hidden_states, self.k)
        value_states = project(hidden_states, self.v)

        # compute scores  # (batch_size, num_block, n_heads, block_len+num_mem, block_len + num_mem*num_slots)  # batch_size, num_block, n_heads, num_mem*num_slots + block_len, num_mem*num_slots + block_len)
        scores = torch.einsum(
            "...qhd,...khd->...hqk", query_states, key_states
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                # print("***********************   if not self.has_relative_attention_bias:   *******************************")
                position_bias = torch.zeros(
                    (
                        1,
                        self.n_heads,
                        self.block_size + self.num_mem * self.num_slots,
                        self.block_size + self.num_mem * self.num_slots,
                    ),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.training and self.gradient_checkpointing:
                    position_bias.requires_grad = True
            else:
                # (1, num_blocks,num_heads, block_size, block_size)
                position_bias = self.compute_bias(self.block_size, self.num_slots)
                block_ids, global_segment_ids = _make_global_fixed_block_ids(
                    mask, self.block_size, self.num_mem
                )
                block_ids, global_segment_ids = block_ids.to(
                    hidden_states.device
                ), global_segment_ids.to(hidden_states.device)
                #  (batch_size, num_heads, seq_len, global_seq_len)
                side_position_bias = self.compute_side_bias(
                    mask, block_ids, global_segment_ids
                )  # masked
                side_position_bias = side_position_bias.type(scores.dtype).to(
                    scores.device
                )
                # batch_size,
                mem_total_position_bias, mem_total_attention_mask = (
                    self.compute_mem_total_bias(
                        mask, self.block_size, self.num_mem, self.num_slots
                    )
                )
                # mem_total_attention_mask = mem_total_attention_mask[:,:,None,:,:]
                # print("mem_total_attention_mask.shape",  mem_total_attention_mask.shape)

            if mask is not None:
                # batch_size, num_blocks, block_size, expanded dimension[1, 2, 5, 1])
                splitted_mask = _split_into_blocks(
                    mask.unsqueeze(2), self.block_size, -2
                )
                # batch_size, num_blocks, 1, 1, block_size
                splitted_mask = splitted_mask.permute(0, 1, 3, 2)[:, :, None, :, :]
                # batch_size, num_blocks, 1, 1, block_size
                splitted_mask = (1.0 - splitted_mask) * torch.finfo(torch.float).min
                position_bias = (
                    position_bias + splitted_mask
                )  # (batch_size, num_blocks,num_heads, block_size, block_size)
                # (batch_size,  num_heads, seq_len, global_seq_len) ---->  batch_size, num_heads, num_blocks, block_size, num_blocks*mem_size ----> batch_size, num_blocks, num_heads block_size, num_blocks*mem_size
                side_position_bias = _split_into_blocks(
                    side_position_bias, self.block_size, dim=-2
                ).transpose(1, 2)

                seq_total_position_bias = torch.cat(
                    [side_position_bias, position_bias], dim=-1
                )

                # position_bias = torch.cat([, seq_total_position_bias], dim=3)
                # batch_size, block_num, num_heads,
                # mem_total_position_bias =  # here we have two blocks we need to spread them on the heads # split_into_blocks(mem_total_position_bias, self.num_mem, dim=-2)
                position_bias = torch.cat(
                    [mem_total_position_bias, seq_total_position_bias], dim=-2
                )

        scores += position_bias

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        # print("================================  attn_weights shape  ===========================: ", attn_weights.shape)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_blocks, n_heads, seq_length, key_length)

        mem_weights = attn_weights[:, :, :, : self.num_mem * self.num_slots]
        # print("mem_total_attention_mask.shape before masking",  mem_weights.shape)

        # print("mem_weights.shape, mem_total_attention_mask.shape   attn_weights.shape",  mem_weights.shape,  mem_total_attention_mask.shape, attn_weights.shape)
        # print("mem_weights shape: ", mem_weights.shape, "mem_total_attention_mask shape: ",  mem_total_attention_mask.shape)
        mem_weights = mem_weights * mem_total_attention_mask
        attn_weights[:, :, :, : self.num_mem * self.num_slots] = mem_weights
        # print("mem_total_attention_mask.shape",  mem_total_attention_mask.shape)
        # for i in range(self.num_slots):
        #     print(f"============================   slot{i+1}   =================================")
        #     print(f"mem_total_attention_mask[:,{i},:,{i*self.num_mem}:{(i+1)*self.num_mem+2},{i*self.num_mem}:{(i+1)*self.num_mem+2}] shape: ", mem_total_attention_mask[:,i,:,i*self.num_mem:(i+1)*self.num_mem+2,i*self.num_mem:(i+1)*self.num_mem+2].shape)
        #     print(mem_total_attention_mask[:,i,:,i*self.num_mem:(i+1)*self.num_mem+2,i*self.num_mem:(i+1)*self.num_mem+2])

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        # print('attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights, value_states) shape: ', attn_output.shape)

        attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
        # print("attn_output = unshape(attn_output)  ", attn_output.shape)
        # for i in range(self.num_slots):
        #     print(f"============================   slot{i+1}   =================================")
        #     print(f"attn_output[:,{i},{i*self.num_mem}:{(i+1)*self.num_mem+2},:] shape: ", attn_output[:,i,i*self.num_mem:(i+1)*self.num_mem+2,:].shape)
        #     print( attn_output[:,i,i*self.num_mem:(i+1)*self.num_mem+2,:])
        #     print("------------------------------------------------------------------------------------")
        attn_output = self.o(attn_output)
        # print("***********************************   attn_output = self.o(attn_output)    ***********************************************************")
        # for i in range(self.num_slots):
        #     print(f"============================   slot{i+1}   =================================")
        #     print(f"attn_output[:,{i},{i*self.num_mem}:{(i+1)*self.num_mem+2},:] shape: ", attn_output[:,i,i*self.num_mem:(i+1)*self.num_mem+2,:].shape)
        #     print( attn_output[:,i,i*self.num_mem:(i+1)*self.num_mem+2,:])
        #     print("------------------------------------------------------------------------------------")

        # present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (
            attn_output,
            position_bias,
        )  #  (attn_output, position_bias,)

        if output_attentions:
            outputs = outputs + (
                attn_weights,
                mem_total_attention_mask,
            )  # (attn_output,position_bias, attn_weights, mem_seq_attention_mask)

        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


class T5MemLayerAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.Attention = T5MemAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config

    def forward(
        self,
        hidden_states,
        # key_value_states=None,
        attention_mask=None,
        position_bias=None,
        mem_total_attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):

        # for i in range(3):
        #     print(f"============================  ********* from T5MemLayerAttention slot{i+1}  hidden_states before doing anything  ********  =================================")
        #     print(f"hidden_states[:,{i},{i*self.config.num_mem}:{(i+1)*self.config.num_mem+2},:] shape: ", hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:].shape)
        #     print( hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:])
        #     print("------------------------------------------------------------------------------------")

        normed_hidden_states = self.layer_norm(hidden_states)
        # for i in range(3):
        #     print(f"============================   from T5MemLayerAttention slot{i+1}   normed_hidden_states = self.layer_norm(hidden_states) =================================")
        #     print(f"attn_output[:,{i},{i*self.config.num_mem}:{(i+1)*self.config.num_mem+2},:] shape: ", normed_hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:].shape)
        #     print( normed_hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:])
        #     print("------------------------------------------------------------------------------------")
        attention_output = self.Attention(
            normed_hidden_states,
            mask=attention_mask,
            # key_value_states=key_value_states,
            position_bias=position_bias,
            mem_total_attention_mask=mem_total_attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # for i in range(3):
        #     print(f"============================   from T5MemLayerAttention slot{i+1}  attention_output = self.Attention(  =================================")
        #     print(f"attention_output[0][:,{i},{i*self.config.num_mem}:{(i+1)*self.config.num_mem+2},:] shape: ", attention_output[0][:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:].shape)
        #     print( attention_output[0][:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:])
        #     print("------------------------------------------------------------------------------------")

        hidden_states = hidden_states + self.dropout(attention_output[0])
        # for i in range(3):
        #     print(f"============================   from T5MemLayerAttention slot{i+1}  hidden_states = hidden_states + self.dropout(attention_output[0]) =================================")
        #     print(f"hidden_states[:,{i},{i*self.config.num_mem}:{(i+1)*self.config.num_mem+2},:] shape: ", hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:].shape)
        #     print( hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem+2,:])
        #     print("------------------------------------------------------------------------------------")
        outputs = (hidden_states,) + attention_output[
            1:
        ]  # add attentions if we output them
        # print("T5MemLayerAttention", len(outputs))
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning(
                    "`past_key_values` is passed to the encoder. Please make sure this is intended."
                )
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:
        ]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if (
                hidden_states.dtype == torch.float16
                and torch.isinf(hidden_states).any()
            ):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = (
                    present_key_value_state + cross_attention_outputs[1]
                )

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5MemBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5MemLayerAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        mem_total_attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
        return_dict=True,
    ):

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            mem_total_attention_mask=mem_total_attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )  # (attn_output,) +(position_bias,), attn_weights
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[
            1:
        ]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        outputs = outputs + attention_outputs

        return outputs  # hidden-states, self-attention position bias, (self-attention weights), mask_mem_all


class T5MemPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5MemConfig
    # load_tf_weights = load_tf_weights_in_t5   ##  TODO:  understand why?
    base_model_prefix = "transformer"
    is_parallelizable = True  # ???
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block", "T5MemBlock"]  # ??

    @property
    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel.dummy_inputs
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = (
            self.config.initializer_factor
        )  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module, (T5MemModel, T5MemEncoderModel, T5MemForConditionalGeneration)
        ):  # TODO T5MemForConditionalGeneration is added
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            # if hasattr(module, "lm_head_encoder") and not self.config.tie_word_embeddings:
            #     module.lm_head_encoder.weight.data.normal_(mean=0.0, std=factor * 1.0)

        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5)
            )
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5)
            )
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model) ** -0.5)
                )

        elif isinstance(module, T5MemAttention):  # TODO is added
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5)
            )
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5)
            )
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model) ** -0.5)
                )
                module.seq_mem_relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model) ** -0.5)
                )
                module.mem_total_relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model) ** -0.5)
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module, (T5Attention, T5Stack, T5MemAttention, T5MemStack)
        ):  # TODO T5MemAttention, T5MemStack are added
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(
                input_ids.shape[:-1] + (1,), decoder_start_token_id
            )
            shifted_input_ids = torch.cat(
                [shifted_input_ids, input_ids[..., :-1]], dim=-1
            )
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class T5Stack(T5MemPreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device
                    )
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = (
                        encoder_extended_attention_mask.to(hidden_states.device)
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5MemStack(T5MemPreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                T5MemBlock(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        batch_size, seq_length = input_shape
        num_blocks = (
            seq_length // self.config.block_size
            if seq_length % self.config.block_size == 0
            else seq_length // self.config.block_size + 1
        )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)  # batch_size, se_len, d_model
            # initialize the memory tokens with pad tokens
            Q_mems = (
                (
                    torch.ones(
                        (batch_size, self.config.num_mem * num_blocks * num_blocks)
                    )
                    * self.config.mem_token_id
                ).to(int)
            ).to(input_ids.device)
            Q_mems_embeds = self.embed_tokens(Q_mems)
            inputs_embeds = _split_into_blocks(
                inputs_embeds, block_len=self.config.block_size, dim=-2
            )
            Q_mems_embeds = _split_into_blocks(
                Q_mems_embeds, block_len=self.config.num_mem * num_blocks, dim=-2
            )
            inputs_embeds = torch.cat(
                [Q_mems_embeds, inputs_embeds], dim=-2
            )  # batch_size, num_blocks, block_len+ num_blocks*num_mem, d_model
            # print("inputs_embeds shape: ", inputs_embeds.shape)

        # required mask seq length can be calculated via length of past
        mask_seq_length = seq_length

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )

        # initialize past_key_values with `None` if past does not exist
        # if past_key_values is None:
        past_key_values = [None] * len(self.block)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = None
        position_bias = None
        mem_total_attention_mask = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if mem_total_attention_mask is not None:
                    mem_total_attention_mask = mem_total_attention_mask.to(
                        hidden_states.device
                    )
                    print(
                        "mem_total_attention_mask", mem_total_attention_mask[:, 2].shape
                    )
                    print(mem_total_attention_mask[:, 2])
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):  # ???
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    position_bias,
                    mem_total_attention_mask,
                    layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )

            else:

                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,  # extended_attention_mask,
                    position_bias=position_bias,
                    mem_total_attention_mask=mem_total_attention_mask,
                    layer_head_mask=layer_head_mask,
                    output_attentions=output_attentions,
                )
                # print("********************  layer_outputs ", len(layer_outputs))

            hidden_states = layer_outputs[0]

            memory_hidden_states = hidden_states[
                :, :, : self.config.num_mem * num_blocks, :
            ] # batch, num_blocks, nummem*block, dmodel
            # TODO: update memory
            mem_upd = torch.zeros_like(memory_hidden_states)
            for i in range(num_blocks):
                mem_upd[
                    :, :, i * self.config.num_mem : (i + 1) * self.config.num_mem
                ] = memory_hidden_states[
                    :, i, i * self.config.num_mem : (i + 1) * self.config.num_mem
                ]
            hidden_states[:, :, : self.config.num_mem * num_blocks, :] = mem_upd

            position_bias = layer_outputs[1]
            if output_attentions:
                all_attentions = all_attentions + (
                    layer_outputs[2],
                )  # (attn_output,position_bias, attn_weights, mem_seq_attention_mask)
            mem_total_attention_mask = layer_outputs[3]
            # print('**************************  mem_total_attention_mask = layer_outputs[3]shape **************************',  mem_total_attention_mask.shape)
            # print("mem_total_attention_mask[:,0,:,:35,:35]:  ", mem_total_attention_mask[:,0,:,:32,:32].shape)
            # print(mem_total_attention_mask[:,0,:,:35,:35])
            # print("mem_total_attention_mask[:,1,:,30:66,30:66]:  ", mem_total_attention_mask[:,1,:,32:64,32:64].shape)
            # print(mem_total_attention_mask[:,1,:,30:66,30:66])
            # print("mem_total_attention_mask[:,2,:,62:98,62:98]")
            # print(mem_total_attention_mask[:,2,:,62:98,62:98])

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # print("Before norm================================================================================")
        # hidden_states = hidden_states[:,:,self.config.num_mem*num_blocks:,:]
        # print("memory_hidden_states = memory_hidden_states.reshape(batch_size,-1,self.config.d_model): ", memory_hidden_states.shape, " mem_total_attention_mask.shape: ",  mem_total_attention_mask.shape)
        # print("memory_hidden_states[:,0,:,:]")
        # print(memory_hidden_states[:,0,:,:])
        # print("memory_hidden_states[:,1,:,:]")
        # print(memory_hidden_states[:,1,:,:])
        # print("memory_hidden_states[:,2,:,:]")
        # print(memory_hidden_states[:,2,:,:])
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # ouput just the hidden states without memory representation
        # print("hidden_states shape", hidden_states.shape)
        hidden_states_ = hidden_states[
            :, :, self.config.num_mem * num_blocks :, :
        ]  # Discovered 23/2/2023 and corrected by adding _
        memory_hidden_states = hidden_states[
            :, :, : self.config.num_mem * num_blocks, :
        ]
        # print('memory_hidden_states = hidden_states[:,:,:self.config.num_mem*num_blocks,:] ', memory_hidden_states.shape)
        '''
        # TODO: update memory
        mem_upd = torch.zeros_like(memory_hidden_states)
        for i in range(num_blocks):
            mem_upd[:, :, i * self.config.num_mem : (i + 1) * self.config.num_mem] = (
                memory_hidden_states[
                    :, i, i * self.config.num_mem : (i + 1) * self.config.num_mem
                ]
            ) # 13/6/2024 fixed memory updated in the loop above

        
        # print("self.config.num_mem, num_blocks", self.config.num_mem, num_blocks, "self.config.num_mem*num_blocks", self.config.num_mem*num_blocks)
        # print("memory_hidden_states shape", memory_hidden_states.shape)
        # print("mem_upd  shape: ", mem_upd.shape, " mem_total_attention_mask.shape: ",  mem_total_attention_mask.shape, "self.config.num_mem: ", self.config.num_mem, "num_blocks: ", num_blocks)
        # print("mem_upd[:,0,:,:]")
        # print(mem_upd[:,0,:,:])
        # print("mem_upd[:,1,:,:]")
        # print(mem_upd[:,1,:,:])
        # print("mem_upd[:,2,:,:]")
        # print(mem_upd[:,2,:,:])
        # print("hidden_states[:,:,:self.config.num_mem*num_blocks,:] == hidden_states[:,:,:self.config.num_mem*num_blocks] ", hidden_states[:,:,:self.config.num_mem*num_blocks,:] == hidden_states[:,:,:self.config.num_mem*num_blocks] )
        memory_hidden_states = mem_upd.reshape(batch_size, -1, self.config.d_model)
        #         print("memory_hidden_states = memory_hidden_states.reshape(batch_size,-1,self.config.d_model): ", memory_hidden_states.shape)

        #         print("hidden_states = hidden_states.reshape(batch_size,-1,self.config.d_model)====>>>>batch_size: ", batch_size)

        #         print("=================================================================================================\n\n")
        '''
        hidden_states = hidden_states_.reshape(batch_size, -1, self.config.d_model)
        memory_hidden_states = memory_hidden_states.reshape(batch_size, -1, self.config.d_model)  # batch, num_mem * num_bloks * num_blocks, d_model
        # print("hidden_states===============================>>>>>>>>", hidden_states.shape, "return_dict: ", return_dict)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    memory_hidden_states,
                    all_hidden_states,
                    all_attentions,
                ]
                if v is not None
            )

        # print("hidden_states shape: ", hidden_states.shape)
        # print("memory_hidden_states shape: ", memory_hidden_states.shape)
        # print("all_hidden_states:  ", all_hidden_states)
        # print("all_attentions:   ", type(all_attentions))
        return EncoderOutput(
            last_hidden_state=hidden_states,
            last_memory_hidden_state=memory_hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         inputs_embeds=None,
#         head_mask=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         # Model parallel
#         if self.model_parallel:
#             torch.cuda.set_device(self.first_device)
#             self.embed_tokens = self.embed_tokens.to(self.first_device)
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             err_msg_prefix = "decoder_" if self.is_decoder else ""
#             raise ValueError(
#                 f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
#             )
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#             input_ids = input_ids.view(-1, input_shape[-1])
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             err_msg_prefix = "decoder_" if self.is_decoder else ""
#             raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

#         batch_size, seq_length = input_shape
#         num_blocks = seq_length//self.config.block_size if seq_length%self.config.block_size==0 else seq_length//self.config.block_size + 1

#         if inputs_embeds is None:
#             assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
#             inputs_embeds = self.embed_tokens(input_ids)  # batch_size, se_len, d_model
#             Q_mems = ((torch.ones((batch_size, self.config.num_mem*num_blocks*num_blocks))*self.config.mem_token_id).to(int)).to(input_ids.device)
#             Q_mems_embeds = self.embed_tokens(Q_mems)
#             inputs_embeds = _split_into_blocks(inputs_embeds, block_len=self.config.block_size, dim=-2)
#             Q_mems_embeds = _split_into_blocks(Q_mems_embeds, block_len=self.config.num_mem*num_blocks, dim=-2)
#             inputs_embeds = torch.cat([Q_mems_embeds, inputs_embeds], dim=-2)  # batch_size, num_blocks, block_len+ num_blocks*num_mem, d_model
#             # print("inputs_embeds shape: ", inputs_embeds.shape)


#         # required mask seq length can be calculated via length of past
#         mask_seq_length =  seq_length

#         if attention_mask is None:
#             attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

#         # initialize past_key_values with `None` if past does not exist
#         # if past_key_values is None:
#         past_key_values = [None] * len(self.block)

#         # Prepare head mask if needed
#         head_mask = self.get_head_mask(head_mask, self.config.num_layers)

#         all_hidden_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None
#         all_cross_attentions = None
#         position_bias = None
#         mem_total_attention_mask = None

#         hidden_states = self.dropout(inputs_embeds)


#         for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
#             layer_head_mask = head_mask[i]
#             # Model parallel
#             if self.model_parallel:
#                 torch.cuda.set_device(hidden_states.device)
#                 # Ensure that attention_mask is always on the same device as hidden_states
#                 if attention_mask is not None:
#                     attention_mask = attention_mask.to(hidden_states.device)
#                 if position_bias is not None:
#                     position_bias = position_bias.to(hidden_states.device)
#                 if mem_total_attention_mask is not None:
#                     mem_total_attention_mask = mem_total_attention_mask.to(hidden_states.device)
#                     # print("mem_total_attention_mask", mem_total_attention_mask[:,2].shape)
#                     print(mem_total_attention_mask[:,2])
#                 if layer_head_mask is not None:
#                     layer_head_mask = layer_head_mask.to(hidden_states.device)

#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 def create_custom_forward(module):  # ???
#                     def custom_forward(*inputs):
#                         return tuple(module(*inputs, output_attentions))

#                     return custom_forward

#                 layer_outputs = checkpoint(
#                     create_custom_forward(layer_module),
#                     hidden_states,
#                     attention_mask,
#                     position_bias,
#                     mem_total_attention_mask,
#                     layer_head_mask,
#                     None,  # past_key_value is always None with gradient checkpointing
#                 )

#             else:

#                 layer_outputs = layer_module(
#                     hidden_states=hidden_states,
#                     attention_mask=attention_mask,  #extended_attention_mask,
#                     position_bias=position_bias,
#                     mem_total_attention_mask=mem_total_attention_mask,
#                     layer_head_mask=layer_head_mask,
#                     output_attentions=output_attentions,
#                 )
#                 # print("********************  layer_outputs ", len(layer_outputs))


#             hidden_states = layer_outputs[0]


#             memory_hidden_states = hidden_states[:,:,:self.config.num_mem*num_blocks,:]
#             # TODO: update memory
#             mem_upd = torch.zeros_like(memory_hidden_states)
#             for i in range(num_blocks):
#                 mem_upd[:,:,i*self.config.num_mem:(i+1)*self.config.num_mem] = memory_hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem]
#             hidden_states[:,:,:self.config.num_mem*num_blocks,:] = mem_upd


#             position_bias = layer_outputs[1]
#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[2],) # (attn_output,position_bias, attn_weights, mem_seq_attention_mask)
#             mem_total_attention_mask = layer_outputs[3]
#             # print('**************************  mem_total_attention_mask = layer_outputs[3]shape **************************',  mem_total_attention_mask.shape)
#             # print("mem_total_attention_mask[:,0,:,:35,:35]:  ", mem_total_attention_mask[:,0,:,:32,:32].shape)
#             # print(mem_total_attention_mask[:,0,:,:35,:35])
#             # print("mem_total_attention_mask[:,1,:,30:66,30:66]:  ", mem_total_attention_mask[:,1,:,32:64,32:64].shape)
#             # print(mem_total_attention_mask[:,1,:,30:66,30:66])
#             # print("mem_total_attention_mask[:,2,:,62:98,62:98]")
#             # print(mem_total_attention_mask[:,2,:,62:98,62:98])


#             # Model Parallel: If it's the last layer for that device, put things on the next device
#             if self.model_parallel:
#                 for k, v in self.device_map.items():
#                     if i == v[-1] and "cuda:" + str(k) != self.last_device:
#                         hidden_states = hidden_states.to("cuda:" + str(k + 1))

#         # print("Before norm================================================================================")
#         # hidden_states = hidden_states[:,:,self.config.num_mem*num_blocks:,:]
#         # memory_hidden_states = hidden_states[:,:,:self.config.num_mem*num_blocks,:]
#         # print("memory_hidden_states = memory_hidden_states.reshape(batch_size,-1,self.config.d_model): ", memory_hidden_states.shape, " mem_total_attention_mask.shape: ",  mem_total_attention_mask.shape)
#         # print("memory_hidden_states[:,0,:,:]")
#         # print(memory_hidden_states[:,0,:,:])
#         # print("memory_hidden_states[:,1,:,:]")
#         # print(memory_hidden_states[:,1,:,:])
#         # print("memory_hidden_states[:,2,:,:]")
#         # print(memory_hidden_states[:,2,:,:])
#         hidden_states = self.final_layer_norm(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         # ouput just the hidden states without memory representation
#         hidden_states = hidden_states[:,:,self.config.num_mem*num_blocks:,:]
#         memory_hidden_states = hidden_states[:,:,:self.config.num_mem*num_blocks,:]

#         # TODO: update memory
#         mem_upd = torch.zeros_like(memory_hidden_states)
#         for i in range(num_blocks):
#             mem_upd[:,:,i*self.config.num_mem:(i+1)*self.config.num_mem] = memory_hidden_states[:,i,i*self.config.num_mem:(i+1)*self.config.num_mem]

#         # print("mem_upd  shape: ", mem_upd.shape, " mem_total_attention_mask.shape: ",  mem_total_attention_mask.shape)
#         # print("mem_upd shape: " , mem_upd.shape , "mem_upd.reshape(batch_size,-1,self.config.d_model).shape: ", mem_upd.reshape(batch_size,-1,self.config.d_model).shape)
#         # print("mem_upd[:,0,:,:]")
#         # print(mem_upd[:,0,:,:])
#         # print("mem_upd[:,1,:,:]")
#         # print(mem_upd[:,1,:,:])
#         # print("mem_upd[:,2,:,:]")
#         # print(mem_upd[:,2,:,:])
#         # print("hidden_states[:,:,:self.config.num_mem*num_blocks,:] == hidden_states[:,:,:self.config.num_mem*num_blocks] ", hidden_states[:,:,:self.config.num_mem*num_blocks,:] == hidden_states[:,:,:self.config.num_mem*num_blocks] )
#         memory_hidden_states = mem_upd.reshape(batch_size,-1,self.config.d_model)
# #         print("memory_hidden_states = memory_hidden_states.reshape(batch_size,-1,self.config.d_model): ", memory_hidden_states.shape)

# #         print("hidden_states = hidden_states.reshape(batch_size,-1,self.config.d_model)====>>>>batch_size: ", batch_size)


# #         print("=================================================================================================\n\n")
#         hidden_states = hidden_states.reshape(batch_size,-1,self.config.d_model)
#         # last_memory_hidden_state = mem_upd[mem_upd.nonzero(as_tuple=True)].view(n,n,-1)
#         # print("hidden_states===============================>>>>>>>>", hidden_states.shape, "return_dict: ", return_dict)

#         # Add last layer
#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(
#                 v
#                 for v in [
#                     hidden_states,
#                     memory_hidden_states,
#                     all_hidden_states,
#                     all_attentions,
#                 ]
#                 if v is not None
#             )

#         # print("hidden_states shape: ", hidden_states.shape)
#         # print("memory_hidden_states shape: ", memory_hidden_states.shape)
#         # print("all_hidden_states:  ", all_hidden_states)
#         # print("all_attentions:   ", type(all_attentions))
#         return EncoderOutput(
#             last_hidden_state=hidden_states,
#             last_memory_hidden_state= memory_hidden_states #mem_upd[:,0,:,:].reshape(batch_size,-1,self.config.d_model), fix 12/7/2024
#             hidden_states=all_hidden_states,
#             attentions=all_attentions,
#         )


@add_start_docstrings(
    "The bare T5Mem Model transformer outputting raw hidden-states without any specific head on top.",
    T5Mem_START_DOCSTRING,
)
class T5MemModel(T5MemPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5MemConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5MemStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:
        Example:
        ```python
        >>> from transformers import T5Tokenizer, T5Model
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)
        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        # print("\n\n\n########################################################## hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello ")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(
            encoder_outputs, BaseModelOutput
        ):  # maybe need to change
            # print("hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello ")
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        memory_states = encoder_outputs[1]
        """
                    last_hidden_state=hidden_states,
            last_memory_hidden_state=memory_hidden_states,
            # past_key_values=None,   #present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        """

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """T5Mem Model with a `language modeling` head on top.""", T5Mem_START_DOCSTRING
)
class T5MemForConditionalGeneration(T5MemPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
        # r"lm_head_encoder.weight",
    ]  #
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5MemConfig):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5MemStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.lm_head_encoder = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        # self.lm_head_encoder = self.lm_head_encoder.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        # self.lm_head_encoder = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        # self.lm_head_encoder = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    # @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=MemSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MemSeq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)  when we have num_beams > 1 the hidden state dimension with not be the same as meoery hidden states should we look into them too????
        if encoder_outputs is None: 
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>if encoder_outputs is None::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
            # print( "encoder_outputs.last_hidden_state.shape shape >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ", encoder_outputs.last_hidden_state.shape, encoder_outputs[0].shape)
            # print("encoder_outputs.last_memory_hidden_state.shape  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", encoder_outputs.last_memory_hidden_state.shape)

        elif return_dict:  # and not isinstance(encoder_outputs, EncoderOutput):
            encoder_outputs = EncoderOutput(
                last_hidden_state=encoder_outputs[0],
                last_memory_hidden_state=encoder_outputs[1],
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
            """
            
                last_hidden_state: torch.FloatTensor = None
                last_memory_hidden_state: torch.FloatTensor = None
                hidden_states: Optional[Tuple[torch.FloatTensor]] = None
                attentions: Optional[Tuple[torch.FloatTensor]] = None
            
            """
        else:
           
            hidden_states = encoder_outputs[0]
            memory = encoder_outputs[1] # batch, num_bloks*num_blocks*num_mem


        hidden_states = encoder_outputs[0]  # in case beam, beam*batch_size, input_len, d_model
        memory = encoder_outputs[1]   # # in case beam, beam*batch_size, num_blocks*num_mem*num_blocks, d_model
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        # batch_size, num_blocks, block_size
        attention_mask = _split_into_blocks(
            attention_mask, block_len=self.config.block_size, dim=1
        )
        batch_size, num_blocks, block_size = attention_mask.shape
        # batch_size, num_blocks*block_size
        attention_mask = attention_mask.reshape(
            (attention_mask.shape[0], hidden_states.shape[1])
        )
        memory_attention_mask = None

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            # encoder_hidden_states=memory,  #  TODO: 2: feed the memory to the decoder
            # encoder_attention_mask=memory_attention_mask,  # TODO: 3: modify the attention mask
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            # self.lm_head_encoder = self.lm_head_encoder.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
            encoder_outputs[0] = encoder_outputs[0].to(self.lm_head.weight.device)
            encoder_outputs[1] = encoder_outputs[1].to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:

            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
            # print("Before encoder_hidden_states = encoder_outputs[0] * (self.model_dim**-0.5)")
            # print("encoder_outputs[0] shape: ", encoder_outputs[0].shape, ">>>>>>>>>>>>>>..  self.model_dim**-0.5=", self.model_dim**-0.5)
            encoder_hidden_states = encoder_outputs[0] * (self.model_dim**-0.5)
            encoder_memory_hidden_states = encoder_outputs[1] * (
                self.model_dim**-0.5
            )  # Why we are doing that??

        lm_logits = self.lm_head(sequence_output)
        # print(" encoder_memory_hidden_states.shape,encoder_hidden_states.shape:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ", encoder_memory_hidden_states.shape,encoder_hidden_states.shape)
        # print("encoder_hidden_states[0] shape: ", encoder_hidden_states[0:1,:,:].shape)
        # print("encoder_memory_hidden_states shape:  ", encoder_memory_hidden_states.shape)
        encoder_logits = self.lm_head(
            torch.cat([encoder_memory_hidden_states, encoder_hidden_states], dim=1)
        )  # in case num_beam>1

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        # fix 8_12_2022
        if not return_dict:
            output = (
                (lm_logits,)
                + decoder_outputs[1:]
                + encoder_outputs
                + (
                    encoder_logits[:, memory.shape[1] :, :],
                    encoder_logits[:, : memory.shape[1], :],
                )
            )
            return ((loss,) + output) if loss is not None else output



        return MemSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_last_memory_hidden_state=encoder_outputs.last_memory_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            encoder_logits=encoder_logits[:, memory.shape[1] :, :],
            encoder_memory_logits=encoder_logits[:, :(num_blocks*self.config.num_mem), :],
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        # print("***********************     def _prepare_encoder_decoder_kwargs_for_generation     **************************")
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        # print('model_kwargs["encoder_outputs"] keys==============================',model_kwargs["encoder_outputs"].keys() )

        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # print("******************   def _expand_inputs_for_generation(  ********************")
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx
            )

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs["last_hidden_state"] = (
                encoder_outputs.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
                )
            )
            if "last_memory_hidden_state" in encoder_outputs:
                encoder_outputs["last_memory_hidden_state"] = (
                    encoder_outputs.last_memory_hidden_state.index_select(
                        0,
                        expanded_return_idx.to(
                            encoder_outputs.last_hidden_state.device
                        ),
                    )
                )

            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # print("***************   def prepare_inputs_for_generation  ****************")
        # print("encoder_outputs", encoder_outputs.keys())
        # print("encoder_outputs.last_hidden_state: ", encoder_outputs.last_hidden_state.shape )
        # print("encoder_outputs.last_memory_hidden_state:  ",  encoder_outputs.last_memory_hidden_state.shape )

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past


@add_start_docstrings(
    "The bare T5Mem Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    T5Mem_START_DOCSTRING,
)
class T5MemEncoderModel(T5MemPreTrainedModel):
    authorized_missing_keys = [
        r"encoder.embed_tokens.weight",
    ]

    def __init__(self, config: T5MemConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5MemStack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:
        Example:
        ```python
        >>> from transformers import T5Tokenizer, T5EncoderModel
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
