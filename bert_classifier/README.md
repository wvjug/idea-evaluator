<div align="center">

# BERT Classifier for Evaluating Solutions to Problems

</div>

## Description

Using BERT classifier model to predict the quality of solutions in three criteria - relevance to circular economy, market potentials, and feasibility. 
It contains code to run the predictions. 

The code in this repository can be run in 2 ways:

1. All models can be trained from scratch.
2. The predictions can be re-run (using our pretrained saved models). 

The code is based on the pytorch project template. 

## Scripts and description

Please see the following scripts for implementation: 

1. Main files: 
- `run.py` is the main script for training
    ```bash
    python3 run.py
    ```
    **Note**: if we want to require a gpu session or submit a slurm job to run on a gpu, please at least require 32G memory.
- `eval.py` is the main script for evaluating
    ```bash
    python3 eval.py
    ```
- `model.py` contains BERT classifier architecture 
- `trainer.py` contains training loop codes
- `dataset.py` contains custom dataset class

## Results 


## Notes

- All of our experiments were run on computers with a GPU.
  None of our code needs multiple GPUs.
- All of our experiments were run on computers running Linux. 

## Licensing 

All rights are reserved by GreenWave and UChicago Data Science Institute. 


