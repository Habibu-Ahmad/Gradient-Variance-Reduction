# Gradient-Variance-Reduction

<table>
  <tr>
    <td>
<img width="390" height="390" alt="gvr_sgd_accuracy" src="https://github.com/user-attachments/assets/519d8c56-54ca-408d-a8a9-91f20cb13bd1" /></td>
    <td>
      <p><strong>GVR Optimizer</strong><br/>
      Achieves 79.09% accuracy on CIFAR-100 using ResNet-18. The proposed GVR optimizer is compared against standard SGD on the CIFAR-100 dataset using the ResNet-18 architecture. Both models were trained for 200 epochs with a batch size of 128, and standard data augmentations including random cropping, horizontal flipping, and Cutout. To ensure a fair and strong baseline, the SGD hyperparameters (learning rate, momentum, and weight decay) were adopted from the settings used in the SAM paper. For GVR, the penalty coefficient alpha was set to 0.01 based on light hyperparameter tuning.
      Outperforms SGD by over 1%. <br/>
      Novel approach based on gradient variance penalty.</p>
    </td>
  </tr>
</table>


<table>
  <tr>
    <td>
      <strong>Experimental Setup:</strong><br>
      The proposed GVR optimizer is compared against standard SGD on the CIFAR-100 dataset using the ResNet-18 architecture.
      Both models were trained for 200 epochs with a batch size of 128, and standard data augmentations including random cropping, horizontal flipping. 
    <td>
      <img width="390" height="390" alt="gvr_sgd_accuracy" src="https://github.com/user-attachments/assets/519d8c56-54ca-408d-a8a9-91f20cb13bd1" />
    </td>
  </tr>
</table>
