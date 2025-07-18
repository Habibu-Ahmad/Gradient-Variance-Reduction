
# Gradient-Variance-Reduction

GVR is a novel gradient-based optimizer that introduces a variance penalty between gradients computed on two augmented views of the same input. Inspired by consistency regularization and the geometry-aware motivation in SAM (Sharpness-Aware Minimization), GVR targets a new direction: minimizing *gradient variance* on the last layer to encourage stable and generalizable learning.

##  Core Idea

Given two augmented views of the same input, `img1` and `img2`, and a shared label `y`, GVR minimizes:

<img width="1299" height="271" alt="gvr_eq" src="https://github.com/user-attachments/assets/329a058f-6ceb-4291-9801-0ddcef3b8860" />



This penalizes inconsistent gradients across augmentations, making the model less sensitive to minor perturbations, improving generalization and potential robustness to label noise.
Note that, for efficiency, gradients are computed only with respect to the last layer and the views are subsampled independently during each forward pass.

**Code Snippet: GVR Step**
```python
loss1 = criterion(model(img1), y)
grads1 = torch.autograd.grad(loss1, model.parameters(), create_graph=True)

loss2 = criterion(model(img2), y)
grads2 = torch.autograd.grad(loss2, model.parameters(), create_graph=True)

penalty = sum(((g1 - g2)**2).sum() for g1, g2 in zip(grads1, grads2))
total_loss = loss1 + loss2 + alpha * penalty
```
## Experimental Setups
<p>
  The proposed <strong>GVR</strong> was compared against SGD on CIFAR-100 using ResNet-18.  
  Both models were trained for 200 epochs with a batch size of 128 and standard augmentations (random crop, horizontal flip, Cutout).  
  SGD hyperparameters followed the SAM (Sharpness-Aware Minimization) paper, while GVR used a penalty coefficient α = 0.01 based on light tuning.
</p>

<table>
  <tr>
    <td>
      <img width="275" height="290" alt="gvr_sgd" src="https://github.com/user-attachments/assets/48253002-6385-48b2-a901-cb1fb13761ea" />
    </td>
    <td style="vertical-align: top; padding-left: 20px;">
      <p><strong>Test Accuracy:</strong></p>
      <ul>
        <li><strong>GVR:</strong> 79.09%</li>
        <li><strong>SGD:</strong> 78.00%</li>
      </ul>
    </td>
  </tr>
</table>


## Project Structure 
<pre><code>.
├── src/                            # Source code
│   ├── models.py                   # Model architecture (e.g., ResNet-18)
│   ├── utils.py                    # Utility functions (data loading, transforms, etc.)
│   └── run.py                      # Training script using GVR or SGD
│
├── Scripts/                        # Shell scripts to run experiments
│   ├── run-gvr-cifar100.sh         # Run experiment with GVR
│   └── run-sgd-cifar100.sh         # Run experiment with SGD
│
├── Results/                        # Experiment logs and visualizations
│   ├── GVR_CIFAR100_ResNet18.log   # Training log for GVR
│   ├── SGD_CIFAR100_ResNet18.log   # Training log for SGD
│   └── gvr_sgd_accuracy1.png       # Accuracy comparison plot
│
├── README.md                       # Project documentation (this file)
</code></pre>
