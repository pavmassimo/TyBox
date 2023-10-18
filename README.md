# Full-integer quantization in TyBox

Full-integer quantization has the capability to convert model input, model output, weights, and activation outputs into 8-bit integer data, compared to other quantization techniques which may leave some amount of data in floating-point. This allows to achieve up to a 4x reduction in memory usage and up to a 3x improvement in latency.

![alt text](https://github.com/pavmassimo/TyBox/blob/feature-extractor-quantization/experiment_notebooks/QFE_TyBox.png)

In standard TyBox, the input static model $Φ$ is partitioned in a feature extractor $Φ_f$ and a classifier $Φ_c$. The primary advantage of TyBox lies in its incrementally learnable classifier, which constitutes a negligible portion of the overall model size. In order to reduce the model dimension while maintaining its prediction abilities, quantization is exclusively applied to $Φ_f$, producing $\hat{Φ}_f$. This approach allows the quantized feature extractor to have an 8-bit resolution, while the incremental classifier retains a 32-bit resolution, ensuring a more precise classification process is maintained. Since $\hat{Φ}_f$ produces 8-bit outputs, while $Φ_c$ is designed to receive 32-bit inputs, $ψ_I$ - the feature vector extracted by $\hat{Φ}_f$ - undergoes a dequantization process before being passed to $Φ_c$. The same process is also employed during the incremental training phase, where the quantized data samples stored in the buffer undergo dequantization before before being utilized for training the classifier.

## Experiments

The experimental setting concerns the image classification on a multi-class problem. For this purpose, the CIFAR-10 and Imagenette datasets have been considered. To validate the performance of the incremental models, we considered the same application scenarios that have been used for the validation of the TyBox multi-class case study: concept drift, incremental learning, and transfer learning. 

Notebook for the experiments can be found in the /experiment_notebooks folder.
