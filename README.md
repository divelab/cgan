# Conditional GAN for Cellular Structure

This is the tensorflow implementation of our recent work, "Conditional Deep Generative Networks for Building an Integrated Cell Model". The paper is not released yet. We will post more detailed information once the paper is published.

## How to run it

1. Clone or download this repository to your working directory.
2. Get the datasets ready. Please check data folder for detail.
3. Set related arguments in main.py. There are three proposed skip connections, please choose the corresponding ops-XXX file and update the name to 'ops.py'.
4. Call ``` python main.py ``` or  ``` python main.py --action=train ``` to train the model.
5. If you wish to use "parzen window" to evaluate the model, set a checkpoint in arguments for the model to reload and then call ``` python main.py --action=test```.






