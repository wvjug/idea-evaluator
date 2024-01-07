# import modules 
import torch
import warnings
from pathlib import Path
import argparse
import model
from transformers import BertTokenizer

if __name__ == "__main__":
    # default config.json location 
    cur_dir = Path(__file__).resolve().parent

    # Wisly TODO:  figure out how to input the problem and solutions from the users 

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt", help="path to model checkpoint", default = cur_dir/"model.pth"
    )
    args = parser.parse_args()
    
    # read command args 
    model_ckpt = Path(args.model_ckpt).resolve()
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

    # Wisly TODO: input data 
    problem = ""
    solution = ""
    inputs = "Problem: " + problem + "\nSolution: " + solution 

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # move to GPU if necessary
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    outputs = classifier(input_ids, attention_mask=attention_mask)
    b, m, n = outputs.shape
    flattened_logits = outputs.view(-1, 9)
    flattened_labels = torch.argmax(flattened_logits, dim=1)
    predicted_labels = flattened_labels.view(b, m) 
    predicted_labels = (predicted_labels.cpu().numpy() + 2) / 2
    print(predicted_labels)

    # Wisly TODO: return the predicted labels back 
    # the shape of predicted labels are (1, 3) where 1 is the batch size/number of the {problem, solution} pairs 
    # and 3 is the score for "Circular Economy", "Market Potentials", "Feasibility"
    # e.g. [[5., 4., 4.,]]




