
import torch
from losses import get_loss_function

batch_size = 2
num_classes = 8
height = 640
width = 304

# Simulated U-Net outputs and segmentation masks.
logits = torch.randn(
    batch_size, num_classes, height, width,
    requires_grad=True,
)
targets = torch.randint(
    0, num_classes, (batch_size, height, width)
)

criterion = get_loss_function(num_classes=num_classes)
loss = criterion(logits, targets)

print("Logits shape:", tuple(logits.shape))
print("Targets shape:", tuple(targets.shape))
print("Loss:", loss.item())

loss.backward()

print("Gradient computed:", logits.grad is not None)
print("Loss function test passed.")

