"""
Qwen3.5 全模态 megatron.bridge 插件。

注册 ``Qwen3_5ForConditionalGeneration``，使 ``AutoBridge.from_hf_pretrained`` 能够
识别 Qwen3.5（含纯文本和视觉两种权重）并提供 Megatron 兼容的 VL 模型 + 权重映射。

架构概览
--------
* HF 视觉编码器 ``Qwen3_5VisionModel``（冻结，仅挂在第一个 PP stage）
* Megatron Core GPTModel（M-RoPE + 混合 linear/full attention）
  - linear attention：``slime_plugins.models.qwen3_5.Attention``
  - full attention：标准 TE transformer layer
* MTP（Multi-Token Prediction）block：full attention（与语言模型 MTP 一致）

HF 权重前缀
-----------
* ``model.visual.**``                  → vision encoder
* ``model.language_model.layers.*``   → language decoder
* ``model.language_model.embed_tokens.weight``
* ``model.language_model.norm.weight``
* ``lm_head.weight``
* ``mtp.layers.0.**``                  → MTP block（full attention）
* ``mtp.fc.weight``
* ``mtp.norm.weight``
* ``mtp.pre_fc_norm_embedding.weight``
* ``mtp.pre_fc_norm_hidden.weight``
"""

from __future__ import annotations

import copy
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping, ReplicatedMapping
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# THD ↔ BSHD + CP 工具函数 —— 直接复用 GLM-4.6V 实现，零重复
# ---------------------------------------------------------------------------
from slime_plugins.megatron_bridge.glm4v_moe import (  # noqa: E402
    _bshd_to_thd,
    _gather_input_ids_from_cp,
    _select_local_image_embeds,
    _thd_to_bshd,
)


# ---------------------------------------------------------------------------
# Megatron VL 模型
# ---------------------------------------------------------------------------
class Qwen3_5VLModel(MegatronModule):
    """Qwen3.5 视觉语言模型（Megatron 训练）。

    第一个 PP stage 持有冻结的 HF 视觉编码器；所有 stage 共享
    标准 Megatron Core GPTModel（混合 linear/full attention + M-RoPE）。
    """

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        hf_vision_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.spatial_merge_size = language_transformer_config.spatial_merge_size

        self.share_embeddings_and_output_weights = False

        # 视觉编码器 —— 仅在第一个 PP stage 上实例化并冻结
        self.vision_model = None
        if self.pre_process:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

            self.vision_model = Qwen3_5VisionModel._from_config(hf_vision_config)
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")

        # 语言模型 —— 标准 Megatron Core GPT，使用 M-RoPE
        self.language_model = MCoreGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    # -- Megatron pipeline engine 需要的接口 -----------------------------------

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    # -- 视觉编码 ---------------------------------------------------------------

    def _get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """运行 HF 视觉编码器 + merger，返回 flat image embeddings [N_tokens, lm_hidden]。

        Qwen3_5VisionModel 结构：
          patch_embed → blocks (hidden=1152) → merger (1152→4096, 合并 spatial 2×2)
        forward 返回 BaseModelOutputWithPooling，last_hidden_state 是 merger **之前**
        的 ViT 输出（shape [N_patches, 1152]）；merger 需要在同一 no_grad 块内调用，
        确保 image_embeds 完全无 grad_fn，避免 in-place scatter 污染 autograd graph。
        """
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
        with torch.no_grad():
            out = self.vision_model(pixel_values, grid_thw=image_grid_thw)

            # 提取 ViT 输出（兼容裸 Tensor 和 ModelOutput）
            if isinstance(out, torch.Tensor):
                hidden_states = out
            elif hasattr(out, "last_hidden_state"):
                hidden_states = out.last_hidden_state
            else:
                hidden_states = out[0]

            # 若维度与语言模型不匹配，在 no_grad 内调用 merger
            # 确保最终的 image_embeds 是纯 detached tensor
            if hidden_states.shape[-1] != self.config.hidden_size:
                if hasattr(self.vision_model, "merger"):
                    hidden_states = self.vision_model.merger(hidden_states)
                else:
                    raise RuntimeError(
                        f"Vision encoder output dim {hidden_states.shape[-1]} != "
                        f"language model hidden_size {self.config.hidden_size}, "
                        f"but vision_model has no 'merger' attribute."
                    )

        return hidden_states

    # -- M-RoPE position IDs --------------------------------------------------

    @staticmethod
    def _get_vision_position_ids(
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int,
        spatial_merge_size: int,
        device,
    ) -> torch.Tensor:
        """计算单张图片/视频的 3D 位置 ID（移植自 HF）。"""
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        n_tokens = llm_grid_h * llm_grid_w * llm_grid_t

        pos_w = torch.arange(start_position, start_position + llm_grid_w, device=device)
        pos_w = pos_w.repeat(llm_grid_h * llm_grid_t)
        pos_h = torch.arange(start_position, start_position + llm_grid_h, device=device)
        pos_h = pos_h.repeat_interleave(llm_grid_w * llm_grid_t)
        pos_t = torch.full((n_tokens,), start_position, device=device, dtype=torch.long)
        return torch.stack([pos_t, pos_h, pos_w], dim=0)  # [3, n_tokens]

    def _compute_mrope_position_ids(
        self,
        input_ids_bshd: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """从 [bs, seq] 格式的 input_ids 计算 3D M-RoPE position IDs。

        通过连续 image_token_id run 定位图像区域，不依赖 mm_token_type_ids。
        """
        import itertools

        bs, seq_len = input_ids_bshd.shape
        device = input_ids_bshd.device
        spatial_merge_size = self.spatial_merge_size

        position_ids = torch.zeros(3, bs, seq_len, dtype=torch.long, device=device)

        if image_grid_thw is None or image_grid_thw.numel() == 0:
            # 纯文本：3 维重复 1D 位置
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            position_ids[0] = pos
            position_ids[1] = pos
            position_ids[2] = pos
            return position_ids

        grid_iter = iter(image_grid_thw)

        for b in range(bs):
            ids = input_ids_bshd[b]
            is_image = ids == self.image_token_id
            token_types = is_image.long()

            groups = []
            for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                g = list(group)
                groups.append((key, g[0][0], g[-1][0] + 1))

            current_pos = 0
            pos_list = []
            for modality, start, end in groups:
                if modality == 0:
                    # 文本 token
                    n = end - start
                    pos_list.append(
                        torch.arange(n, device=device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += n
                else:
                    # 图像 token
                    grid_thw = next(grid_iter)
                    temp_merge_size = grid_thw[0]
                    vis_pos = self._get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        temp_merge_size,
                        spatial_merge_size,
                        device,
                    )
                    pos_list.append(vis_pos)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            all_pos = torch.cat(pos_list, dim=1)  # [3, seq_for_this_sample]
            position_ids[:, b, : all_pos.shape[1]] = all_pos

        return position_ids

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params=None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        # 多模态输入（从 multimodal_train_inputs 解包）
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert pixel_values_videos is None, "Video input is not yet supported"
        assert inference_params is None, "Inference mode is not yet supported"

        # 提前拿到 cu_seqlens 和 CP 信息（视觉 scatter 和 M-RoPE 均需要）
        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        cp_size = parallel_state.get_context_parallel_world_size()
        full_input_ids = None  # 在视觉 scatter 和 M-RoPE 间复用

        combined_embeddings = None

        if self.pre_process:
            # 1. 文本 embedding（来自语言模型 embedding 层）
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()  # [seq, batch, hidden]

            # 2. 视觉编码 + masked scatter
            if pixel_values is not None and image_grid_thw is not None:
                image_embeds = self._get_image_features(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(
                    combined_embeddings.device, combined_embeddings.dtype
                )

                # CP > 1 时，input_ids 是本 rank 的局部 chunk，但 pixel_values
                # 包含所有图像；需要筛选落在本 rank 的 image tokens。
                if cp_size > 1 and cu_seqlens is not None:
                    full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    cp_rank = parallel_state.get_context_parallel_rank()
                    image_embeds = _select_local_image_embeds(
                        full_input_ids,
                        cu_seqlens,
                        self.image_token_id,
                        image_embeds,
                        cp_rank,
                        cp_size,
                    )

                image_mask = (input_ids == self.image_token_id).contiguous()
                # scatter：[seq, bs, hidden] → [bs, seq, hidden] → scatter → 转回
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if image_mask.any():
                    combined_embeddings[image_mask] = image_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            # sequence_parallel：scatter 到 SP 区域
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined_embeddings
                )
                combined_embeddings = combined_embeddings.contiguous()

        # 3. 计算 M-RoPE position IDs（所有 PP stage 均需要）
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if position_ids is None:
            if self.pre_process:
                # 第一个 PP stage：从 input_ids 计算
                if cu_seqlens is not None:
                    if cp_size > 1:
                        if full_input_ids is None:
                            full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    else:
                        full_input_ids = input_ids
                    input_ids_bshd = _thd_to_bshd(full_input_ids, cu_seqlens)
                    pos_bshd = self._compute_mrope_position_ids(input_ids_bshd, image_grid_thw)
                    pos_packed = _bshd_to_thd(pos_bshd.permute(1, 2, 0), cu_seqlens)
                    position_ids = pos_packed.permute(2, 0, 1).contiguous()  # [3, 1, T_global]
                else:
                    position_ids = self._compute_mrope_position_ids(input_ids, image_grid_thw)
            else:
                # 非第一个 PP stage：分配缓冲区等待 broadcast
                if cu_seqlens is not None:
                    T = cu_seqlens[-1].item()
                    position_ids = torch.zeros(
                        3, 1, T, dtype=torch.long, device=torch.cuda.current_device()
                    )
                else:
                    raise NotImplementedError(
                        "Non-THD position_ids broadcast not yet supported for non-first PP stages"
                    )

            # PP > 1：从第一个 stage broadcast 到所有 stage
            if pp_size > 1:
                src = parallel_state.get_pipeline_model_parallel_first_rank()
                torch.distributed.broadcast(
                    position_ids,
                    src=src,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        # 4. 语言模型 forward（跳过重新 embedding，直接传 decoder_input）
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        return output


# ---------------------------------------------------------------------------
# Model Provider
# ---------------------------------------------------------------------------
@dataclass
class Qwen3_5VLModelProvider(GPTModelProvider):
    """Provider，负责创建 Qwen3_5VLModel。

    在模块级别定义（而非函数内部），确保对象可被 pickle
    （megatron-bridge 通过 broadcast_object_list 跨 PP rank 传递 config）。
    """

    # VL 专属
    image_token_id: int = 248056
    video_token_id: int = 248057
    spatial_merge_size: int = 2

    # 视觉 config（HF config 对象）
    hf_vision_config: object = None
    # 文本 config（用于构建混合 layer spec）
    hf_text_config: object = None

    # M-RoPE
    position_embedding_type: str = "mrope"
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])
    scatter_embedding_sequence_parallel: bool = False

    # 语言模型最大序列长度
    language_max_sequence_length: int = 262144

    # linear attention backend（对应 args.qwen_gdn_backend）
    qwen_gdn_backend: str = "fla"

    # HF checkpoint 路径：linear attn 层的 HuggingfaceAttention.__init__
    # 会调用 _load_hf_config(args.hf_checkpoint)，必须是合法路径
    hf_checkpoint: str = ""

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """创建 Qwen3_5VLModel 实例。"""
        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage(
                ignore_virtual=False, vp_stage=vp_stage
            )
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage(
                ignore_virtual=False, vp_stage=vp_stage
            )

        transformer_layer_spec = self._build_hybrid_layer_spec(vp_stage)

        model = Qwen3_5VLModel(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            hf_vision_config=self.hf_vision_config,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
        return model

    def _build_hybrid_layer_spec(self, vp_stage=None):
        """构建混合 layer spec：linear_attention 层替换为 Qwen3_5GatedDeltaNet。

        复现 ``slime_plugins.models.qwen3_5.get_qwen3_5_spec`` 的逻辑，
        但不依赖 ``args``，直接使用 ``self.hf_text_config``。
        """
        from slime_plugins.models.qwen3_5 import Attention as LinearAttnLayer

        kwargs = {"use_transformer_engine": True}
        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage
        spec = get_gpt_decoder_block_spec(self, **kwargs)

        text_config = self.hf_text_config
        # text_config.layer_types 由 Qwen3.5 直接提供，无需推断
        if not hasattr(text_config, "layer_types"):
            interval = getattr(text_config, "full_attention_interval", 4)
            n = text_config.num_hidden_layers
            text_config.layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(n)
            ]

        num_layers_to_build = get_num_layers_to_build(self, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(self, vp_stage=vp_stage)

        # fake_args：LinearAttnLayer.__init__ → HuggingfaceAttention.__init__
        # 会调用 _load_hf_config(args.hf_checkpoint)，必须传真实路径
        fake_args = SimpleNamespace(
            qwen_gdn_backend=self.qwen_gdn_backend,
            hf_checkpoint=self.hf_checkpoint,
            sequence_parallel=getattr(self, "sequence_parallel", False),
        )

        for layer_id in range(num_layers_to_build):
            if text_config.layer_types[layer_id + offset] == "linear_attention":
                layer_specs = copy.deepcopy(spec.layer_specs[layer_id])
                layer_specs.submodules.self_attention = ModuleSpec(
                    module=LinearAttnLayer,
                    params={"args": fake_args},
                )
                spec.layer_specs[layer_id] = layer_specs

        return spec


# ---------------------------------------------------------------------------
# Bridge（向 AutoBridge 注册）
# ---------------------------------------------------------------------------
try:
    from transformers import Qwen3_5ForConditionalGeneration as _Qwen3_5VLHF
except ImportError:
    _Qwen3_5VLHF = "Qwen3_5ForConditionalGeneration"


@MegatronModelBridge.register_bridge(source=_Qwen3_5VLHF, target=Qwen3_5VLModel)
class Qwen3_5Bridge(MegatronModelBridge):
    """HuggingFace Qwen3.5 ↔ Megatron VL 模型的 Bridge。"""

    def provider_bridge(self, hf_pretrained):
        """从 HF config 构造 Qwen3_5VLModelProvider。"""
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = deepcopy(hf_config.vision_config)

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)
        vision_config.dtype = model_dtype

        # rope 参数（在 text_config.rope_parameters 里）
        rope_params = getattr(text_config, "rope_parameters", {}) or {}
        mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        rotary_base = rope_params.get("rope_theta", 10_000_000)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)

        # qk_layernorm：text_config 里没有显式字段，Qwen3.5 固定启用
        qk_layernorm = getattr(text_config, "qk_layernorm", True)
        # attention_bias
        attention_bias = getattr(text_config, "attention_bias", False)

        base_kwargs = dict(
            # 基础 transformer config
            num_layers=text_config.num_hidden_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", None),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=attention_bias,
            hidden_dropout=0.0,
            attention_dropout=getattr(text_config, "attention_dropout", 0.0),
            autocast_dtype=model_dtype,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rotary_base,
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # 训练优化
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen3.5 attention 特性
            qk_layernorm=qk_layernorm,
            attention_output_gate=getattr(text_config, "attn_output_gate", True),
            # M-RoPE
            mrope_section=mrope_section,
            position_embedding_type="mrope",
            scatter_embedding_sequence_parallel=False,
            # Vision
            hf_vision_config=vision_config,
            hf_text_config=text_config,
            image_token_id=getattr(hf_config, "image_token_id", 248056),
            video_token_id=getattr(hf_config, "video_token_id", 248057),
            spatial_merge_size=getattr(vision_config, "spatial_merge_size", 2),
            language_max_sequence_length=text_config.max_position_embeddings,
            # HF checkpoint 路径，供 linear attn 层 _load_hf_config() 使用
            hf_checkpoint=getattr(hf_config, "_name_or_path", ""),
        )

        # MTP 配置
        if getattr(text_config, "mtp_num_hidden_layers", None):
            base_kwargs["mtp_num_layers"] = text_config.mtp_num_hidden_layers

        # MoE 配置（9B 是 dense，此处为 MoE 变体预留）
        if getattr(text_config, "num_experts", None):
            base_kwargs.update(
                ffn_hidden_size=getattr(
                    text_config,
                    "shared_expert_intermediate_size",
                    text_config.intermediate_size,
                ),
                num_moe_experts=text_config.num_experts,
                moe_router_topk=text_config.num_experts_per_tok,
                moe_ffn_hidden_size=text_config.moe_intermediate_size,
                moe_shared_expert_intermediate_size=getattr(
                    text_config, "shared_expert_intermediate_size", None
                ),
                moe_grouped_gemm=True,
                moe_token_dispatcher_type="alltoall",
                moe_router_score_function="softmax",
                moe_router_pre_softmax=False,
                moe_shared_expert_gate=getattr(text_config, "shared_expert_gate", True),
                moe_aux_loss_coeff=getattr(text_config, "router_aux_loss_coef", 0.0),
                moe_router_load_balancing_type="none",
            )

        return Qwen3_5VLModelProvider(**base_kwargs)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """HF Qwen3.5 ↔ Megatron 的全量权重映射。

        注意事项
        --------
        * VLM wrapper 的语言层前缀是 ``model.language_model.layers.*``
          （区别于纯文本 mbridge/qwen3_5.py 中从 text-only checkpoint 打印的
          ``model.layers.*``——那是不带 visual wrapper 的模型）。
        * MTP transformer layer 是 full attention（weight_map 确认
          ``mtp.layers.0.self_attn.*``），无 linear attn。
        * Vision encoder 权重前缀是 ``model.visual.**``。
        * Qwen3_5GatedDeltaNet / Attention（linear attn 层）的权重在
          HuggingfaceAttention 层统一做 TP gather/scatter，对 bridge 而言
          是 replicated（每个 TP rank 持有完整权重），需要显式注册。
        """
        # Linear attention 模块权重全部是 replicated（TP 在外层 HF wrapper 处理）
        # 需要把所有出现在 linear attn 层里的自定义子模块都注册，
        # 否则 AutoMapping._detect_parallelism_type 遇到未知模块类型会报错。
        from slime_plugins.models.qwen3_5 import Attention as _LinearAttnWrapper
        from slime_plugins.models.qwen3_5 import Qwen3_5GatedDeltaNet as _GatedDeltaNet

        _replicated_names = [
            _GatedDeltaNet.__name__,    # Qwen3_5GatedDeltaNet
            _LinearAttnWrapper.__name__, # Attention（外层 wrapper）
        ]

        # fla 库的子模块（ShortConvolution / FusedRMSNormGated）
        try:
            from fla.modules import FusedRMSNormGated, ShortConvolution
            _replicated_names += [ShortConvolution.__name__, FusedRMSNormGated.__name__]
        except ImportError:
            pass

        # Qwen3NextRMSNorm（linear attn 层的 input_layernorm）
        try:
            from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRMSNorm
            _replicated_names.append(Qwen3NextRMSNorm.__name__)
        except ImportError:
            pass

        for _name in _replicated_names:
            AutoMapping.register_module_type(_name, "replicated")
        # ── 直接映射（embedding / norm / lm_head）─────────────────────────
        param_mappings = {
            "language_model.embedding.word_embeddings.weight":
                "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight":
                "lm_head.weight",
            "language_model.decoder.final_layernorm.weight":
                "model.language_model.norm.weight",

            # ── Full-attention 层 ──────────────────────────────────────────
            # input layernorm（TE fused with QKV）
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight":
                "model.language_model.layers.*.input_layernorm.weight",
            # attention output proj
            "language_model.decoder.layers.*.self_attention.linear_proj.weight":
                "model.language_model.layers.*.self_attn.o_proj.weight",
            # QK norm（Qwen3.5 full attn 有 q_norm / k_norm）
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight":
                "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight":
                "model.language_model.layers.*.self_attn.k_norm.weight",

            # ── Linear-attention 层 ───────────────────────────────────────
            # input layernorm（standalone RMSNorm，非 TE fused）
            "language_model.decoder.layers.*.self_attention.input_layernorm.weight":
                "model.language_model.layers.*.input_layernorm.weight",
            # NOTE: GatedDeltaNet 的 9 个子权重（in_proj_*, conv1d, A_log, dt_bias,
            # norm, out_proj）全部走下方的 ReplicatedMapping("...linear_attn.**")，
            # 不在这里用 AutoMapping，原因：AutoMapping 会向下找到叶子 nn.Linear，
            # 而 nn.Linear 没有注册并行类型，导致 _detect_parallelism_type 报错。

            # ── MLP（dense，适用于 9B；MoE 变体见下方 GatedMLPMapping）────
            # post-attention layernorm（TE fused with fc1）
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight":
                "model.language_model.layers.*.post_attention_layernorm.weight",
            # MoE 路径的 pre_mlp_layernorm（standalone）
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight":
                "model.language_model.layers.*.post_attention_layernorm.weight",
            # MLP down proj
            "language_model.decoder.layers.*.mlp.linear_fc2.weight":
                "model.language_model.layers.*.mlp.down_proj.weight",

            # ── MTP 直接映射（fc / norm）──────────────────────────────────
            # HF 侧这 4 个权重在顶层（无层编号），Megatron 侧在 mtp.layers.0.*
            # AutoMapping 要求两边通配符数相同，故写死为 0（mtp_num_hidden_layers=1）
            "mtp.layers.0.eh_proj.weight":        "mtp.fc.weight",
            "mtp.layers.0.enorm.weight":          "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.hnorm.weight":          "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
            # MTP transformer layer（full attention）
            "mtp.layers.0.transformer_layer.self_attention.linear_proj.weight":
                "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.transformer_layer.self_attention.linear_qkv.layer_norm_weight":
                "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.transformer_layer.self_attention.q_layernorm.weight":
                "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.transformer_layer.self_attention.k_layernorm.weight":
                "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.transformer_layer.mlp.linear_fc1.layer_norm_weight":
                "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.transformer_layer.mlp.linear_fc2.weight":
                "mtp.layers.0.mlp.down_proj.weight",
        }

        mapping_list = [
            AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()
        ]

        mapping_list.extend([
            # ── Vision encoder：整体 replicate（不参与 TP 切分）──────────
            ReplicatedMapping(
                megatron_param="vision_model.**",
                hf_param="model.visual.**",
            ),
            # ── Linear-attention 层：每个子权重单独一条 ReplicatedMapping ──
            # 不用 "linear_attn.**" 混搭 * 和 **——ReplicatedMapping 混用时
            # 会把层号和参数名的位置对调（已确认产生 WARNING 且权重未加载）。
            # 9 条显式映射，每条只有一个 * 通配符，行为确定。
            *[
                ReplicatedMapping(
                    megatron_param=f"language_model.decoder.layers.*.self_attention.{mkey}",
                    hf_param=f"model.language_model.layers.*.{hkey}",
                )
                for mkey, hkey in [
                    ("linear_attn.A_log",            "linear_attn.A_log"),
                    ("linear_attn.conv1d.weight",     "linear_attn.conv1d.weight"),
                    ("linear_attn.dt_bias",           "linear_attn.dt_bias"),
                    ("linear_attn.in_proj_a.weight",  "linear_attn.in_proj_a.weight"),
                    ("linear_attn.in_proj_b.weight",  "linear_attn.in_proj_b.weight"),
                    ("linear_attn.in_proj_qkv.weight","linear_attn.in_proj_qkv.weight"),
                    ("linear_attn.in_proj_z.weight",  "linear_attn.in_proj_z.weight"),
                    ("linear_attn.norm.weight",       "linear_attn.norm.weight"),
                    ("linear_attn.out_proj.weight",   "linear_attn.out_proj.weight"),
                ]
            ],

            # ── Full-attention QKV 合并 ────────────────────────────────
            QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            ),
            # attention_bias=False，无 QKV bias；若 MoE 变体有 bias，按需添加

            # ── Dense MLP gate+up 合并 ────────────────────────────────
            GatedMLPMapping(
                megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.language_model.layers.*.mlp.gate_proj.weight",
                up="model.language_model.layers.*.mlp.up_proj.weight",
            ),

            # ── MoE 变体（9B dense 不用，router/shared expert 为 MoE 变体预留）──
            # MoE router
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.mlp.router.weight",
                hf_param="model.language_model.layers.*.mlp.gate.weight",
            ),
            # MoE shared expert
            GatedMLPMapping(
                megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.language_model.layers.*.mlp.shared_expert.gate_proj.weight",
                up="model.language_model.layers.*.mlp.shared_expert.up_proj.weight",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                hf_param="model.language_model.layers.*.mlp.shared_expert.down_proj.weight",
            ),
            AutoMapping(
                megatron_param="language_model.decoder.layers.*.mlp.shared_experts.gate_weight",
                hf_param="model.language_model.layers.*.mlp.shared_expert_gate.weight",
            ),
            # NOTE: MoE per-expert mappings (experts.linear_fc1.weight* / linear_fc2.weight*)
            # use a "weight*" suffix as the expert-id wildcard, giving 2 wildcards on the
            # megatron side but only 1 on the HF fused-tensor side — AutoMapping rejects this.
            # These are omitted here; add a custom ConversionTask subclass when enabling MoE.

            # ── MTP transformer layer QKV + MLP gate+up ──────────────
            QKVMapping(
                megatron_param="mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight",
                q="mtp.layers.0.self_attn.q_proj.weight",
                k="mtp.layers.0.self_attn.k_proj.weight",
                v="mtp.layers.0.self_attn.v_proj.weight",
            ),
            GatedMLPMapping(
                megatron_param="mtp.layers.0.transformer_layer.mlp.linear_fc1.weight",
                gate="mtp.layers.0.mlp.gate_proj.weight",
                up="mtp.layers.0.mlp.up_proj.weight",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)
