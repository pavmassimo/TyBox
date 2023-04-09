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


### Example on-device application
###### How to
1. Deploy the project under `on_device_application/tybox_transfer_learning.7z` on the Arduino Nano 33 BLE using the arduino IDE.
2. Run `serial_server/serial_server.py` with the following parameters
   `COM<X> serial_server/data/transfer_fashion.data 9600` where X is the serial port your arduino is connected to.
   
###### Details
- This project contains a model trained on mnist deployed with the TyBox method.
- The application listens for input data on the serial port, every training input item is added to the buffer and used to train the model.
- The serial server streams the input items which are samples of fashion mnist (with labels)
- After every 50 train set items we stream the test set of 100 items to evaluate the model. When in test mode the data are only used to perform forward pass and the output label is recorded and compared to the true label to compute accuracy.
- The serial_server.py script streams the data and also record the accuracy results of every test phase, these are draw in real time on a matplotlib graph.
