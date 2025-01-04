'''
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
'''

from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from functools import partial, reduce
from PIL import Image
import torch
import torch.utils.checkpoint
from torch import nn
import os
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (convert_to_rgb, normalize, rescale, resize, to_channel_dimension_format, )
from transformers.image_utils import (ChannelDimension, PILImageResampling, to_numpy_array, )
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from transformers.utils import ModelOutput

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               QKVParallelLinear,
                                               RowParallelLinear)

from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

class SigLipImageProcessor:
    def __init__(self,
                 image_mean=(0.5, 0.5, 0.5),
                 image_std=(0.5, 0.5, 0.5),
                 size=(384, 384),
                 crop_size: Dict[str, int] = None,
                 resample=PILImageResampling.BICUBIC,
                 rescale_factor=1 / 255,
                 data_format=ChannelDimension.FIRST):
        crop_size = crop_size if crop_size is not None else {"height": 384, "width": 384}
        crop_size = get_size_dict(crop_size, default_to_square=True, param_name="crop_size")

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors):
        if isinstance(images, Image.Image):
            images = [images]
        else:
            assert isinstance(images, list)

        transforms = [
            convert_to_rgb,
            to_numpy_array,
            partial(resize, size=self.size, resample=self.resample, data_format=self.data_format),
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(normalize, mean=self.image_mean, std=self.image_std, data_format=self.data_format),
            partial(to_channel_dimension_format, channel_dim=self.data_format, input_channel_dim=self.data_format),
        ]

        images = reduce(lambda x, f: [*map(f, x)], transforms, images)
        data = {"pixel_values": images}

        return BatchFeature(data=data, tensor_type=return_tensors)


class SigLipVisionConfig(PretrainedConfig):
    model_type = "siglip_vision_model"

    def __init__(
            self,
            hidden_size=1152,
            image_mean=(0.5, 0.5, 0.5),
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            num_channels=3,
            image_size=384,
            patch_size=14,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.image_mean = image_mean

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPVisionModelOutput with CLIP->SigLip
class SigLipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
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

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SigLipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, 
                 config,
                 linear_method: Optional[LinearMethodBase] = None):
        
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        
        # with tensor parallel
        self.total_num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.total_num_heads
        tensor_model_parallel_world_size = (get_tensor_model_parallel_world_size())
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads // tensor_model_parallel_world_size)
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
    
        # pylint: disable=C0103
        self.qkv_proj = QKVParallelLinear(
            self.embed_dim,
            self.head_dim,
            self.total_num_heads,
            bias = True,
            linear_method=linear_method,
        )
        
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            linear_method=linear_method,
        )
        
    def forward(
            self,
            hidden_states: torch.Tensor,
            input_metadata: InputMetadata
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()
        
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        
        query_states = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = k.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = v.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output[0]


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->SigLip
class SigLipMLP(nn.Module):
    def __init__(self, config, linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            linear_method=linear_method
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.hidden_act, quant_config, config.intermediate_size)
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            linear_method=linear_method
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)[0]
        return hidden_states


# Copied from transformers.models.clip.modeling_clip.CLIPEncoderLayer with CLIP->SigLip
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig, linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config, linear_method)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config, linear_method)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
            self,
            hidden_states: torch.Tensor,
            input_metadata: InputMetadata
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            input_metadata=input_metadata
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class SigLipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        """Load from a path"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v")
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                # pylint: disable=E1136

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

# Copied from transformers.models.clip.modeling_clip.CLIPEncoder with CLIP->SigLip
class SigLipEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SigLipEncoderLayer`].

    Args:
        config: SigLipVisionConfig
    """

    def __init__(self, 
                 config: SigLipVisionConfig, 
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SigLipEncoderLayer(config, linear_method) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    # Ignore copy
    def forward(
            self,
            inputs_embeds,
            input_metadata: InputMetadata,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            input_metadata:
                Input metadata for Paged attention
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = None

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    input_metadata
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    input_metadata
                )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class SigLipVisionTransformer(nn.Module):
    def __init__(self, 
                 config: SigLipVisionConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config, linear_method)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SigLipMultiheadAttentionPoolingHead(config)

    def forward(
            self,
            pixel_values,
            input_metadata: InputMetadata,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            input_metadata=input_metadata
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SigLipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionModel(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _no_split_modules = ["SigLipEncoderLayer"]

    def __init__(self, 
                 config: SigLipVisionConfig,
                 linear_method: Optional[LinearMethodBase] = None):
        
        super().__init__(config)

        self.vision_model = SigLipVisionTransformer(config, linear_method)
        del self.vision_model.encoder.layers[-1:]
        self.vision_model.head = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        """Load from a path"""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v")
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # # Skip loading extra bias for GPTQ models.
                # if name.endswith(".bias") and name not in params_dict:
                #     continue
                # pylint: disable=E1136

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                
    def forward(
            self,
            pixel_values,
            input_metadata: InputMetadata,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, SiglipVisionModel

        >>> model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled features
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            input_metadata=input_metadata,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class SigLipVisionTower(nn.Module):
    def __init__(self, 
                 vision_tower, 
                 vision_tower_cfg, 
                 linear_method: Optional[LinearMethodBase] = None,
                 delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.config = SigLipVisionConfig()

        self.vision_tower_name = vision_tower

        self.image_processor = SigLipImageProcessor()
        
        self.vision_tower = SiglipVisionModel(self.config, linear_method)
        
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = self.config

    def load_model(self):
        if self.is_loaded:
            return

        self.vision_tower.load_weights(self.vision_tower_name)

        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.is_loaded = True

    # @torch.no_grad()
    def forward(self, 
                images,
                input_metadata: InputMetadata):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      input_metadata=input_metadata, output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
                assert image_features.shape[-2] == 729
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                   input_metadata=input_metadata, output_hidden_states=True)
            image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
            assert image_features.shape[-2] == 729

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
