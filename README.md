
# Gradient-Variance-Regularization (GVR)

GVR is a a novel gradient alignment regularization method that penalizes disagreement between gradients of augmented views of the same input. Inspired by consistency regularization and the geometry-aware motivation in SAM (Sharpness-Aware Minimization), GVR targets a new direction: minimizing *gradient variance* on the last layer to encourage stable and generalizable learning.

##  Core Idea

Given two augmented views of the same input, `img1` and `img2`, and a shared label `y`, GVR minimizes:

<img width="1299" height="271" alt="gvr_eq" src="https://github.com/user-attachments/assets/329a058f-6ceb-4291-9801-0ddcef3b8860" />



This penalizes inconsistent gradients across augmentations, making the model less sensitive to minor perturbations, improving generalization and potential robustness to label noise.
>  For efficiency, gradients are  computed only with respect to the **last layer**, and the views are **subsampled independently** during each forward pass.
##  How It Works

The GVR optimizer performs the following during a training step:

1. Takes two augmented versions of the same input (`img1`, `img2`) and their label.
2. Computes model outputs and gradients for both views separately.
3. Calculates a penalty term based on the **squared difference between the gradients**.
4. Combines both losses with the penalty, performs backpropagation, and updates parameters using the base optimizer.

This process **reduces the variance between gradients from different augmentations**, improving **generalization**.


**Code Snippet: GVR Step**
```python
loss1 = criterion(model(img1), y)
grads1 = torch.autograd.grad(loss1, model.parameters(), create_graph=True)

loss2 = criterion(model(img2), y)
grads2 = torch.autograd.grad(loss2, model.parameters(), create_graph=True)

penalty = sum(((g1 - g2)**2).sum() for g1, g2 in zip(grads1, grads2))
total_loss = loss1 + loss2 + alpha * penalty
```
## Experimental Setup
<table>
  <tr>
    <td style="width: 50%; vertical-align: top; padding-right: 20px;">
      <p>
        The proposed <strong>GVR</strong> was compared against SGD on CIFAR-100 using ResNet-18.<br><br>
        Both models were trained for 200 epochs with a batch size of 128 and standard augmentations (random crop, horizontal flip, Cutout).<br><br>
        SGD hyperparameters followed the SAM (Sharpness-Aware Minimization) paper, while GVR used a penalty coefficient α = 0.01 based on light tuning.<br><br>
        <strong>GVR achieved 79.09% test accuracy</strong>, outperforming <strong>SGD at 78.00%</strong>.
      </p>
    </td>
    <td style="width: 50%; vertical-align: top; text-align: center;">
      <img width="100%" alt="gvr_vs_sgd" src="https://github.com/user-attachments/assets/48253002-6385-48b2-a901-cb1fb13761ea" />
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
