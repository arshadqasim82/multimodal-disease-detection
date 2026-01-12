import torch
import torch.nn.functional as F


class GradCAM:
    """
    DenseNet-safe Grad-CAM:
    - Forward hook captures activations.
    - We attach a gradient hook directly on the activation tensor (no backward module hooks),
      avoiding inplace/view issues.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

        # Capture gradients w.r.t. activations
        def _store_grad(grad):
            self.gradients = grad

        out.register_hook(_store_grad)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        loss = logits[torch.arange(logits.size(0)), class_idx].sum()
        loss.backward()

        # Global average pool gradients -> weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize CAM per-batch tensor
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach(), logits.detach()
