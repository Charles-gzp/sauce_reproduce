import importlib
from contextlib import contextmanager
from functools import partial
from typing import Callable, List, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor


def _get_nested_attr(module, attr_path: str):
    """按点路径读取嵌套属性，支持类似 layers[3] 的索引写法。"""
    current = module
    for attr in attr_path.split("."):
        if "[" in attr and attr.endswith("]"):
            attr_name, index = attr[:-1].split("[")
            current = getattr(current, attr_name)[int(index)]
        else:
            current = getattr(current, attr)
    return current


def _resolve_vision_layers(model):
    """兼容 CLIP/LLaVA 的 vision encoder layer 路径自动探测。"""
    candidate_paths = [
        "vision_model.encoder.layers",
        "model.vision_tower.vision_model.encoder.layers",
        "vision_tower.vision_model.encoder.layers",
        "base_model.vision_tower.vision_model.encoder.layers",
        "model.vision_model.encoder.layers",
    ]
    for path in candidate_paths:
        try:
            layers = _get_nested_attr(model, path)
            if hasattr(layers, "__getitem__"):
                return layers
        except Exception:
            continue
    # 兜底：遍历模块名，优先匹配包含 vision 的 encoder.layers。
    for name, module in model.named_modules():
        if "vision" not in name:
            continue
        if hasattr(module, "encoder") and hasattr(module.encoder, "layers"):
            layers = module.encoder.layers
            if hasattr(layers, "__getitem__"):
                return layers
    raise ValueError("未找到可用的 vision encoder layers 路径，请检查模型结构。")


def _try_load_multimodal_model(model_name: str):
    """优先尝试多模态 Auto 类，失败再降级到通用 AutoModel。"""
    transformers_mod = importlib.import_module("transformers")
    candidate_loader_names = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    last_err = None
    for loader_name in candidate_loader_names:
        if not hasattr(transformers_mod, loader_name):
            continue
        loader = getattr(transformers_mod, loader_name)
        try:
            return loader.from_pretrained(model_name)
        except Exception as err:
            last_err = err
            continue
    if last_err is not None:
        return AutoModel.from_pretrained(model_name)
    return AutoModel.from_pretrained(model_name)


class Hook:
    def __init__(self, block_layer: int, module_name: str, hook_fn: Callable, return_module_output=True):
        if module_name != "resid":
            raise ValueError(f"暂只支持 module_name='resid'，收到: {module_name}")
        self.block_layer = block_layer
        self.module_name = module_name
        self.return_module_output = return_module_output
        self.function = self.get_full_hook_fn(hook_fn)

    def get_full_hook_fn(self, hook_fn: Callable):
        def full_hook_fn(module, module_input, module_output):
            # 一些模型层输出是 tuple，第一个元素通常是隐藏状态。
            hidden = module_output[0] if isinstance(module_output, (tuple, list)) else module_output
            hook_fn_output = hook_fn(hidden)
            if self.return_module_output:
                return module_output
            # 保持输出结构与原层一致，减少 hook 后向前兼容问题。
            if isinstance(module_output, (tuple, list)):
                return hook_fn_output
            if isinstance(hook_fn_output, (tuple, list)):
                return hook_fn_output[0]
            return hook_fn_output

        return full_hook_fn

    def get_module(self, model):
        layers = _resolve_vision_layers(model)
        return layers[self.block_layer]


class HookedVisionTransformer:
    def __init__(
        self,
        model_name: str,
        device="cuda",
        vlm_family: str = "clip",
        torch_dtype=None,
    ):
        self.vlm_family = vlm_family
        # 按配置加载 CLIP/LLaVA/LLama-Vision，并使用指定精度减少显存。
        model, processor = self.get_ViT(model_name, vlm_family=vlm_family, torch_dtype=torch_dtype)
        self.model = model.to(device)
        self.processor = processor

    def get_ViT(self, model_name, vlm_family: str = "clip", torch_dtype=None):
        transformers_mod = importlib.import_module("transformers")
        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["low_cpu_mem_usage"] = True
        if vlm_family == "clip":
            # 明确使用 CLIP 类，避免 AutoModel 丢失对比输出。
            if hasattr(transformers_mod, "CLIPModel") and hasattr(transformers_mod, "CLIPProcessor"):
                model = transformers_mod.CLIPModel.from_pretrained(model_name, **load_kwargs)
                processor = transformers_mod.CLIPProcessor.from_pretrained(model_name)
                return model, processor
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
            processor = AutoProcessor.from_pretrained(model_name)
            return model, processor
        if vlm_family == "llava":
            # 优先使用 LLaVA 专用类，兼容性更好。
            if hasattr(transformers_mod, "LlavaForConditionalGeneration") and hasattr(transformers_mod, "LlavaProcessor"):
                model = transformers_mod.LlavaForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
                processor = transformers_mod.LlavaProcessor.from_pretrained(model_name)
                return model, processor
            model = _try_load_multimodal_model(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            return model, processor
        if vlm_family == "llama_vision":
            # Llama-3.2-Vision 对应的 HF 类名可能为 Mllama*。
            if hasattr(transformers_mod, "MllamaForConditionalGeneration"):
                model = transformers_mod.MllamaForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
                processor = AutoProcessor.from_pretrained(model_name)
                return model, processor
            model = _try_load_multimodal_model(model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            return model, processor
        raise ValueError(f"不支持的 vlm_family: {vlm_family}")

    def _get_model_device(self):
        return next(self.model.parameters()).device

    def _normalize_text_inputs(self, text, n_images: int):
        if isinstance(text, str):
            return [text] * n_images
        if isinstance(text, list):
            if len(text) == n_images:
                return text
            if len(text) == 1:
                return text * n_images
        return [""] * n_images

    def _format_multimodal_prompt(self, text: str) -> str:
        # LLaVA 常见输入格式包含 <image> 占位符。
        if self.vlm_family in ["llava", "llama_vision"]:
            if "<image>" in text:
                return text
            return f"<image>\n{text}"
        return text

    def prepare_inputs(self, images, text="", device=None):
        # 统一封装不同 VLM 的 processor 调用方式。
        n_images = len(images) if isinstance(images, list) else 1
        text_inputs = self._normalize_text_inputs(text, n_images)
        text_inputs = [self._format_multimodal_prompt(t) for t in text_inputs]
        inputs = self.processor(
            images=images,
            text=text_inputs,
            return_tensors="pt",
            padding=True,
        )
        target_device = self._get_model_device() if device is None else device
        return inputs.to(target_device)

    def run_with_cache(self, list_of_hook_locations: List[Tuple[int, str]], *args, return_type="output", **kwargs):
        cache_dict, list_of_hooks = self.get_caching_hooks(list_of_hook_locations)
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                output = hooked_model(*args, **kwargs)

        if return_type == "output":
            return output, cache_dict
        if return_type == "loss":
            if self.vlm_family == "clip":
                return self.contrastive_loss(output.logits_per_image, output.logits_per_text), cache_dict
            if hasattr(output, "loss") and output.loss is not None:
                return output.loss, cache_dict
            raise ValueError("当前模型不支持 contrastive loss，请使用 return_type='output'。")
        raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")

    def get_caching_hooks(self, list_of_hook_locations: List[Tuple[int, str]]):
        """cache_dict 的 key 为 (block_layer, module_name)。"""
        cache_dict = {}
        list_of_hooks = []

        def save_activations(name, activations):
            cache_dict[name] = activations.detach()

        for (block_layer, module_name) in list_of_hook_locations:
            hook_fn = partial(save_activations, (block_layer, module_name))
            hook = Hook(block_layer, module_name, hook_fn)
            list_of_hooks.append(hook)
        return cache_dict, list_of_hooks

    @torch.no_grad
    def run_with_hooks(self, list_of_hooks: List[Hook], *args, return_type="output", **kwargs):
        with self.hooks(list_of_hooks) as hooked_model:
            with torch.no_grad():
                output = hooked_model(*args, **kwargs)
        if return_type == "output":
            return output
        if return_type == "loss":
            if self.vlm_family == "clip":
                return self.contrastive_loss(output.logits_per_image, output.logits_per_text)
            if hasattr(output, "loss") and output.loss is not None:
                return output.loss
            raise ValueError("当前模型不支持 contrastive loss，请使用 return_type='output'。")
        raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")

    def contrastive_loss(
        self,
        logits_per_image: Float[Tensor, "n_images n_prompts"],
        logits_per_text: Float[Tensor, "n_prompts n_images"],
    ):
        assert logits_per_image.size()[0] == logits_per_image.size()[1], "The number of prompts does not match the number of images."
        batch_size = logits_per_image.size()[0]
        labels = torch.arange(batch_size).long().to(logits_per_image.device)
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        total_loss = (image_loss + text_loss) / 2
        return total_loss

    @contextmanager
    def hooks(self, hooks: List[Hook]):
        """上下文管理器：注册 hooks，执行 forward，最后统一清理。"""
        hook_handles = []
        try:
            for hook in hooks:
                module = hook.get_module(self.model)
                handle = module.register_forward_hook(hook.function)
                hook_handles.append(handle)
            yield self.model
        finally:
            for handle in hook_handles:
                handle.remove()

    def to(self, device):
        self.model = self.model.to(device)

    def __call__(self, *args, return_type="output", **kwargs):
        return self.forward(*args, return_type=return_type, **kwargs)

    def forward(self, *args, return_type="output", **kwargs):
        if return_type == "output":
            return self.model(*args, **kwargs)
        if return_type == "loss":
            output = self.model(*args, **kwargs)
            if self.vlm_family == "clip":
                return self.contrastive_loss(output.logits_per_image, output.logits_per_text)
            if hasattr(output, "loss") and output.loss is not None:
                return output.loss
            raise ValueError("当前模型不支持 contrastive loss，请使用 return_type='output'。")
        raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()
