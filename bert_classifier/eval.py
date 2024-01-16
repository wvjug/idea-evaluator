import torch
import warnings
from pathlib import Path
# from . import model
import model
from transformers import BertTokenizer
import sys

sys.path.append("..")

def eval(problem:str, solution:str, model_ckpt:str):    
    model_ckpt = Path(model_ckpt).resolve()
    device = "cpu"
    # warning if not using cuda 
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        warnings.warn(f"Cuda: {torch.cuda.is_available()}. Cuda is not available. May run model much slower than expected.")

    # model config and model 
    bert_model_name = 'bert-base-uncased'
    num_classes = 9
    mconf = model.BERTClassifierConfig(num_classes, bert_model_name)
    classifier = model.BERTClassifier(mconf)

    classifier.load_state_dict(torch.load(model_ckpt, map_location=torch.device(device)))
    classifier.to(device)

    inputs = "Problem: " + problem + "\nSolution: " + solution 

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    encoding = tokenizer(inputs, return_tensors='pt', padding=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # print(inputs)
    # print(len(input_ids[0]), input_ids)
    # print(attention_mask)
    
    # move to GPU if necessary
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    classifier.eval()
    with torch.set_grad_enabled(False):
        outputs = classifier(input_ids, attention_mask=attention_mask)
    if len(outputs.shape) == 3:
        b, m, n = outputs.shape # m = 3 n = 9 
        flattened_logits = outputs.view(-1, 9)
        flattened_labels = torch.argmax(flattened_logits, dim=1)
        predicted_labels = flattened_labels.view(b, m)
    elif len(outputs.shape) == 2:
        predicted_labels = torch.argmax(outputs, dim=1)
    else:
        raise ValueError(f"Shape of output is not correct where current shape is {outputs.shape}")
    predicted_labels = (predicted_labels.cpu().numpy() + 2) / 2

    print(f"output shape: {outputs.shape}, raw output: {outputs}")
    print(f"label shape: {predicted_labels}, labels: {predicted_labels}")

    # the shape of predicted labels are (1, 3) where 1 is the batch size/number of the {problem, solution} pairs 
    # and 3 is the score for "Circular Economy", "Market Potentials", "Feasibility"
    # e.g. [[5., 4., 4.,]]

    return predicted_labels 

if __name__ == "__main__":
    problem = input("Enter problem: ")
    solution = input("Enter solution: ")
    eval(problem=problem, solution=solution, model_ckpt="/home/mingchiehliu/ai_earthhack/bert_classifier/model.pth")



