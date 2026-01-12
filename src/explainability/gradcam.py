import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1)

        loss = logits[torch.arange(logits.size(0)), class_idx].sum()
        loss.backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize CAM
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach(), logits.detach()
