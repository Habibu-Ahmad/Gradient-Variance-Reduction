
# Gradient-Variance-Reduction

GVR is a novel gradient-based optimizer that introduces a variance penalty between gradients computed on two augmented views of the same input. Inspired by consistency regularization and the geometry-aware motivation in SAM (Sharpness-Aware Minimization), GVR targets a new direction: minimizing *gradient variance* on the last layer to encourage stable and generalizable learning.

###  Core Idea

Given two augmented views of the same input, `img1` and `img2`, and a shared label `y`, GVR performs:

$
L_{\text{total}} = \mathcal{L}(f(\text{img1}), y) + \mathcal{L}(f(\text{img2}), y) + \alpha \cdot \| \nabla_{\theta} \mathcal{L}_1 - \nabla_{\theta} \mathcal{L}_2 \|^2
$

This penalizes inconsistent gradients across augmentations, making the model less sensitive to minor perturbations — leading to better generalization and potential robustness to label noise.

## Code Snippet: GVR Step
```python
loss1 = criterion(model(img1), y)
grads1 = torch.autograd.grad(loss1, model.parameters(), create_graph=True)

loss2 = criterion(model(img2), y)
grads2 = torch.autograd.grad(loss2, model.parameters(), create_graph=True)

penalty = sum(((g1 - g2)**2).sum() for g1, g2 in zip(grads1, grads2))
total_loss = loss1 + loss2 + alpha * penalty
```
##  Performance on CIFAR-100 (ResNet-18)
<table>
  <tr>
    <td style="vertical-align: top; padding-right: 30px;">
      <p> 
      </p>
    </td>
    <td>
      <img width="590" height="590" alt="gvr_sgd_accuracy1" src="https://github.com/user-attachments/assets/0099f9ef-304c-4218-9a2e-917d70560f33" />
    </td>
  </tr>
</table>
