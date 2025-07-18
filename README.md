# Gradient-Variance-Reduction

<table>
  <tr>
    <td style="vertical-align: top; padding-right: 20px;">
      <p>
        The proposed <strong>GVR</strong> optimizer is compared against SGD on CIFAR-100 using ResNet-18.
        Both models were trained for 200 epochs with a batch size of 128 and standard augmentations (random crop, horizontal flip, Cutout).
        <br/><br/>
        SGD hyperparameters followed the SAM paper, while GVR used a penalty coefficient α = 0.01 based on light tuning.
        <br/><br/>
        GVR achieved <strong>79.09%</strong> test accuracy, outperforming SGD at <strong>78.00%</strong>.
      </p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/519d8c56-54ca-408d-a8a9-91f20cb13bd1" width="500" alt="gvr_sgd_accuracy"/>
    </td>
  </tr>
</table>

