# TyBox

In order to deploy smaller and faster incremental on-device learning models, quantization is a natural evolution of the TyBox toolbox.

Full-integer quantization has the capability to convert model input, model output, weights, and activation outputs into 8-bit integer data, compared to other quantization techniques which may leave some amount of data in floating-point. This allows to achieve up to a 4x reduction in memory usage and up to a 3x improvement in latency.

In standard TyBox, the input static model $Φ$ is partitioned in a feature extractor $Φ_f$ and a classifier $Φ_c$. The primary advantage of TyBox lies in its incrementally learnable classifier, which constitutes a negligible portion of the overall model size. In order to reduce the model dimension while maintaining its prediction abilities, quantization is exclusively applied to $Φ_f$, producing $\hat{Φ_f}$. This approach allows the quantized feature extractor to have an 8-bit resolution, while the incremental classifier retains a 32-bit resolution, ensuring a more precise classification process is maintained. Since $\hat{Φ_f}$ produces 8-bit outputs, while $Φ_c$ is designed to receive 32-bit inputs, $ψ_I$ - the feature vector extracted by $\hat{Φ_f}$ - undergoes a dequantization process before being passed to $Φ_c$. The same process is also employed during the incremental training phase, where the quantized data samples stored in the buffer undergo dequantization before before being utilized for training the classifier.

## Experiments

You can find the notebook for the experiments both in the /experiment_notebooks folder and as colab at the links below:

### Concept Drift Experiment
https://colab.research.google.com/drive/1DhLffUVHxrn1wIOIjrcRLodc4sDtvra6?usp=sharing

### Incremental Learning Experiment

https://colab.research.google.com/drive/1E_Nfo68ksIHJC8J-Lcf6vOppU6bKFVZa?usp=sharing

### Transfer Learning Experiment

https://colab.research.google.com/drive/1nCgR0-XfQTZG1vvI6qLuEhkjZqxMb7XS?usp=sharing

### How to build project
- run `python setup.py bdist_wheel` at root
- `.whl` file is now available in `/dist`
- you can now install the project by running `pip install filename.whl` 

###### Example: how to use tybox in colab:
- build project
- upload `.whl` file to colab
- run `pip install name.whl`
- `import TyBox` ✅

###### Alternative:
- run `!git clone https://github.com/pavmassimo/TyBox.git` in a colab cell
- `from TyBox import TyBox` ✅
- __note__: this will install the version from the `main` branch
