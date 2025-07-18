
# Gradient-Variance-Reduction

GVR is a novel gradient-based optimizer that introduces a variance penalty between gradients computed on two augmented views of the same input. Inspired by consistency regularization and the geometry-aware motivation in SAM (Sharpness-Aware Minimization), GVR targets a new direction: minimizing *gradient variance* on the last layer to encourage stable and generalizable learning.

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

