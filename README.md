# Circular Economy Idea Evaluator

Our solution is a web app that evaluates the quality of a solution given a problem in the context of the circular economy. The quality of the solution is evaluated using 3 evaluation metrics from a scale of 1-5 using a BERT model, and then suggest alternative solutions using a Generative Chatbot. It then recommends improvements to the solution based on this evaluation using a GPT model. 
##
## Index
####
[Dependencies](#dependencies)
####
[Instructions to run app](#instructions-to-run-app)
####
[Input Evaluation Metrics](#input-evaluation-metrics)
####
[Instructions to Train Models](#architecture-description)
####
[Contributions](#contributions)
##
## Dependencies
The app is built on the Python 3 library Streamlit and requires the following packages to be run:
1) Streamlit (v1.29.0)
2) PyTorch
3) Transformers by Hugging Face
##
## Instructions to Run App
1. Set up environment and install dependencies
- Install miniconda
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh
    ```
- Initialize conda environment 
    ```bash
    conda activate 
    ```
- Update conda to avoid failing to install dependencies
    ```bash
    conda update --all
    ```
- Create a conda environment `mpcs` by running `env.yml`, which contains required packages
    ```bash
    conda env create -f env.yml
    ```
- Activate the environment 
    ```bash 
    conda activate mpcs
    ```
2. Run Streamlit app using
```
#bash
streamlit run chat_ui.py
```
3. This will run the app on localhost and open it on the default browser
## 
## Input Evaluation Metrics
####
The metrics used to evaluate solutions used as input to our tool is as follows (on a scale of 1-5):
#### 1. Adherence to Circular Economy
####
Fit into the three principles:
1) Waste and Pollution Elimination
2) Product Circulation
3) Nature Regeneration
####
#### 2. Market Potential
####
- Market size and saturation
- Current demands
- Growth potential
####
#### 3. Feasibility
- Implementation Difficulties and Risks
##
## Instructions for Training Models

The code in this repository can be run in 2 ways:
1. All models can be trained from scratch.
2. The predictions can be re-run (using our pretrained saved models). 

### BERT Classifier 
Using BERT classifier model to predict the quality of solutions in three criteria - adherence to the circular economy principles, market potentials, and feasibility. The scripts are all stored in directory `bert_classifier`. Please also see the [reference for text classification with BERT](https://medium.com/@khang.pham.exxact/text-classification-with-bert-7afaacc5e49b). Please see the following scripts for implementation: 
1. Main scripts: 
    - `run.py` is the main script for training
        ```bash
        python3 run.py --data_dir ../data/hackathon_data_annotated.csv
        ```
        **Note**: Since we do not annotate all data, please use the `hackathon_data_annotated.csv` as the training data source. 
    - `eval.py` is the main script for evaluating
        ```bash
        python3 eval.py
        ```
2. Other scripts: 
    - `model.py` contains BERT classifier architecture 
    - `trainer.py` contains training loop codes
    - `dataset.py` contains custom dataset class

### Generative Language Chatbot
Using GPT-based model to give the example solutions for the problems. The scripts are all stored in directory `generative_chatbot`. 

The `tinyllama.py` script contains setup to load from HuggingFace the generative language model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. This is a newer generative language model than GPT-2 with better performance.

The `generate_response` function takes in a prompt, expected to be a problem related to sustainability, and a max_length which defaults to 300 (enough for most reasonable responses). It tokenizes the input and feeds it into the TinyLlama model, and decodes it back into English.

The usage section specifies an example usage case. The most notable thing here is that we are manually prompting the model to provide a solution to the inputted problem, so the problem is prepended with the phrase "Propose a solution to this problem: ". This functionality is carried over to the Streamlit application.
##
## Limitations

BERT model tends to predict the same output no matter what input is. The potential reason is the model does not learn the context well. To address the issue, annotated data and the model complexity can be increased through stacked layers or increased dimensions of parameters to match data complexity

GPT model often runs out of memory when training. Downsizing the tokenized sentences can be used to address the issue.

## Contributions 

| Members  | Contact | Github username | 
| -------- | ------- | --------------- |
| Ming-Chieh Liu  | eddie.m.c.liu@gmail.com   | ming-chieh-liu |
| Natcha Choptnoparatpat | nchotnoparatpat@uchicago.edu    |FahChotnoparatpat |
| Varun Mohan   | vamohan@uchicago.edu   | vmohan96 | 
| Wisly Juganda | wisly@uchicago.edu | wvjug | 
