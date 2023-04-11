# TyBox

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
