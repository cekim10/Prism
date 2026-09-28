"""Int8 weight-only quantization (per output channel, symmetric absmax) with on-the-fly
dequantisation to the activation dtype. Needed only because the bf16 model (16 GB) does not stay
resident on this 24 GB Mac; every forward then pays ~50 s of paging."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        w = linear.weight.data.float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        self.register_buffer("w_int8", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.bfloat16))
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone())
        self.in_features, self.out_features = linear.in_features, linear.out_features
        for attr in ("_is_residual",):
            if hasattr(linear, attr):
                setattr(self, attr, getattr(linear, attr))

    def forward(self, x):
        w = self.w_int8.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        return F.linear(x, w, self.bias)


@torch.no_grad()
def quantize_model_inplace(model, device):
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                q = QuantLinear(child).to(device)
                setattr(module, child_name, q)
                del child
                n += 1
    if device == "mps":
        torch.mps.empty_cache()
    return n
