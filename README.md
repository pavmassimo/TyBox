# TyBox

## Experiments

### Concept Drift Experiment
https://colab.research.google.com/drive/1H5GmSao-ZI1jKifYny1pvOMs0IBoyq90?usp=sharing

### Incremental Learning Experiment

https://colab.research.google.com/drive/1lEWNFpj37FLiIoN-y0Unt8ojmA28Qe4t?usp=sharing

### Transfer Learning Experiment

https://colab.research.google.com/drive/1xIXcGr25CbY2p5iuQB2ck8p07jkV1jhT#scrollTo=dGvcsPJLkzjx

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
