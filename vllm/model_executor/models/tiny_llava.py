"""Inference-only LLaVA model compatible with HuggingFace weights."""
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PhiConfig, AutoModel
# from transformers import SiglipVisionModel
from vllm.model_executor.vision_models.siglip_encoder import SigLipVisionTower
from vllm.model_executor.projectors.multimodal_projector import build_vision_projector
from transformers.activations import ACT2FN

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)

from vllm.model_executor.models.phi import PhiForCausalLM

from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear,
                                               LinearMethodBase)
from vllm.model_executor.layers.activation import get_act_fn

from vllm.sequence import SamplerOutput
from vllm.logger import init_logger
import numpy as np

logger = init_logger(__name__)
KVCache = Tuple[torch.Tensor, torch.Tensor]

class TinyLlavaPhiConfig(PhiConfig):
    model_type = "tiny_llava_phi"

class TinyLlavaMultiModalProjector(nn.Module):

    def __init__(self, config: TinyLlavaPhiConfig, linear_method: Optional[LinearMethodBase] = None):
        super().__init__()
        
        # mlp2xgelu
        # self.linear_1 = nn.Linear(config.hidden_size, 
        #                           config.intermediate_size)
        # self.act = ACT2FN[config.hidden_act]
        # self.linear_2 = nn.Linear(config.intermediate_size, 
        #                           config.hidden_size)
        self.linear_1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            linear_method=linear_method
        )
        quant_config = getattr(linear_method, "quant_config", None)
        self.act = get_act_fn(config.hidden_act, quant_config, config.intermediate_size)
        self.linear_2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            linear_method=linear_method
        )
 
    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
 
class TinyLlavaForConditionalGeneration(nn.Module):

    def __init__(
        self,
        config,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.linear_method = linear_method

        # image token is added into the model tokenizer 
        self.config.image_token_index = 50295
        # we should manually convert the input type to target model type
        # since in transformers, autocasting for Siglip has not yet been implemented
        self.vision_tower = SigLipVisionTower(config.mm_vision_tower, config)
        
        # delete last layer for 
        self.language_model = PhiForCausalLM(config, linear_method)
        self.multi_modal_projector = build_vision_projector(config)

        self.vocab_size = config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        config.image_size = 384
        config.patch_size = 14
        patches_per_image = int(config.image_size / config.patch_size) ** 2
        self.tokens_per_image = patches_per_image
        # if self.config.vision_feature_select_strategy == "default":
        #     self.tokens_per_image = patches_per_image
        # elif self.config.vision_feature_select_strategy == "full":
        #     self.tokens_per_image = patches_per_image + 1
        # else:
        #     raise ValueError(
        #         f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
        #     )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def prepare_promt(self,
                      input_ids: List[int],
                      pixel_values: torch.Tensor = None):
        """
        1.Check the validation of the imput.
        2.Expand each image token to the number of tokens per image.
          So the scheduler can allocate proper resources.

        We do not extract the image features here.
        This function deals with only one request/promt.
        """
        input_ids = np.asarray(input_ids)
        assert len(
            input_ids.shape
        ) == 1, f"input_ids should be 1D array, got {input_ids.shape}"

        # Create a mask to know where image tokens are
        image_token_mask = input_ids == self.config.image_token_index
        non_image_indices = np.where(
            input_ids != self.config.image_token_index)

        # check if the number of image tokens and images are matched
        num_image_tokens = image_token_mask.sum()
        num_images = 0 if pixel_values is None else pixel_values.shape[0]
        assert num_images == num_image_tokens, f" The input provided to the model are wrong. The number of image tokens ({num_image_tokens}) is not equal to the number of images ({num_images}) provided."

        # expand each image token to number of tokens per image
        if num_images > 0:
            # Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `image_token_mask` identifies image tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            new_token_positions = np.cumsum(
                (image_token_mask * (self.tokens_per_image - 1) + 1), -1) - 1
            text_to_overwrite = new_token_positions[non_image_indices]

            final_input_ids = np.ones(
                (num_images * (self.tokens_per_image - 1)) + len(input_ids),
                dtype=input_ids.dtype) * self.config.image_token_index
            final_input_ids[text_to_overwrite] = input_ids[non_image_indices]

            input_ids = final_input_ids
        else:
            final_input_ids = input_ids
        return final_input_ids

    def extract_visual_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[List[torch.Tensor]] = None,
        image_features: Optional[List[torch.Tensor]] = None
    ):
        """
        process batched inputs, extract visual features from pixel_values
        pixel_values: each element is a tensor of shape [num_images, 3, height, width]
        image_features: extracted visual features
        """
        if input_ids.shape[1] == 1:
            # in the case of generation with cache
            return None
        _pixel_values = [
            values for values in pixel_values if values is not None
        ]
        if len(_pixel_values) < 1:
            res_image_features = image_features
        else:
            _pixel_values = torch.cat(_pixel_values, dim=0).to(device=self.vision_tower.device, dtype=self.vision_tower.dtype)
            selected_image_feature = self.vision_tower(_pixel_values)
            #_pixel_values = torch.cat(_pixel_values, dim=0).to('cuda')
            # TODO change the vision_tower to parallel version
            # target_dtype = self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
            # _pixel_values = _pixel_values.to(dtype=target_dtype)
            # image_outputs = self.vision_tower(_pixel_values, output_hidden_states=True)

            # selected_image_feature = image_outputs.hidden_states[
            #     self.config.vision_feature_layer]
            # if self.config.vision_feature_select_strategy == "default":
            #     selected_image_feature = selected_image_feature[:, 1:]
            # elif self.config.vision_feature_select_strategy == "full":
            #     selected_image_feature = selected_image_feature
            # else:
            #     raise ValueError(
            #         f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
            #     )
            
            input_image_features = image_features if image_features is not None else [
                None
            ] * input_ids.shape[0]
            
            projected_image_features = self.multi_modal_projector(selected_image_feature)
            nb_images, image_hidden_dim, embed_dim = projected_image_features.shape
            res_image_features = []
            # flatten the image tokens for each prompt
            for i, value in enumerate(pixel_values):
                if value is None:
                    # if the prompt have no pixel_values, use the input image feature
                    res_image_features.append(
                        input_image_features[i].to('cuda'))
                else:
                    res_image_features.append(
                        projected_image_features[:value.shape[0]].contiguous(
                        ).reshape(-1, embed_dim))
                    projected_image_features = projected_image_features[
                        value.shape[0]:]
        return res_image_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        image_features: Optional[List[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        image_features is a list of tensor, len(image_features) == batch_size
          each tensor is a concatenation of image features, there shapes are not the same,
          and may be None if the prompt have no image tokens.
          shape: [image_num * image_hidden_dim, embed_dim], image_hidden_dim: feature tokens per image
        """

        if inputs_embeds is None:
            # Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # repace the embedding of image tokens with the image features.
            if input_ids.shape[1] != 1:
                # Extract the image features
                if pixel_values is not None:
                    # TODO Put the image process here seams won't impact the GUDA graph? But will comsume too
                    # more memory during the graph_runner trace.
                    # But put this out side may change the model_runner too much and not graceful.
                    image_features = self.extract_visual_features(
                        input_ids, 
                        pixel_values, 
                        image_features
                    )
                    # if image_features is None:
                    #     image_features = []
                    # print(input_ids.shape, [f if f is None else f.shape for f in image_features])
                    if image_features is not None:
                        image_token_mask = input_ids == self.config.image_token_index
                        for i, features in enumerate(image_features):
                            if features is not None:  # the prompt have a image
                                inputs_embeds[i][image_token_mask[i]] = features.to(inputs_embeds)
                    
                    # print("input_ids:", input_ids)
                    # print("input embeds:", inputs_embeds)
                    # print("image_pixels input:", pixel_values)
                    # print("image features after vision tower:", image_features)
            else:
                # we are in the case of generation with cache
                pass

        hidden_states = self.language_model(input_ids,
                                            positions,
                                            kv_caches,
                                            input_metadata,
                                            inputs_embeds=inputs_embeds)
        # if pixel_values is not None:
        #     print("Hidden states:", hidden_states)
        return hidden_states

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        return self.language_model.sample(hidden_states, sampling_metadata)

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v")
        ]
        params_dict = dict(self.named_parameters())
        unused_keys = []
        
        # for k in params_dict.keys():
        #     print(k)

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            if name.startswith("model."):
                name = name[6:]  # remove "model." prefix

            if name.startswith("vision_tower"):
                pass
            elif name.startswith("mm_projector"):
                name = name.replace("mm_projector", "multi_modal_projector")
            else:
                name = "language_model." + name
                
            if name.startswith("vision_tower") or name.startswith('multi_modal_projector'):  # load vision model weights
                # name = name[6:]  # remove "model." prefix
                if params_dict.get(name, None) is None:
                    unused_keys.append(name)
                else:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            else:  # load language model weights
                if "rotary_emb.inv_freq" in name:
                    continue
                if ("rotary_emb.cos_cached" in name
                        or "rotary_emb.sin_cached" in name):
                    # Models trained using ColossalAI may include these tensors in
                    # the checkpoint. Skip them.
                    continue
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name not in params_dict:
                        unused_keys.append(name)
                        continue
                    # pylint: disable=E1136

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

        if len(unused_keys) > 0:
            unused_keys.sort()
            logger.warning(
                f"These keys found in checkpoint but not used in model! {unused_keys}"
            )
