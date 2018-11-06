# Computational Modeling of Cellular Structures Using Conditional Deep Generative Networks

This is the tensorflow implementation of our recent work, "Computational Modeling of Cellular Structures Using Conditional Deep Generative Networks". The paper is currently under review and not released yet. We have released our experimental results, please check the results folder.

## How to run it

1. Clone or download this repository to your working directory.
2. Install the requirements with pip. This is done with:
  `pip install -r requirements.txt`
  Python 3.6 is recommended. 3.7 may get a "package not found" error.
3. Get the datasets ready. Please check data folder for detail.
4. Set related arguments in main.py. There are three proposed skip connections, please choose the corresponding ops-XXX file and update the name to 'ops.py'.
5. Call ``` python main.py ``` or  ``` python main.py --action=train ``` to train the model.
6. If you wish to use "parzen window" to evaluate the model, set a checkpoint in arguments for the model to reload and then call ``` python main.py --action=test```.






