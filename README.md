# TyBox

## Experiments

It's possible to find the notebook for the experiments in the /experiment_notebooks folder.

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
