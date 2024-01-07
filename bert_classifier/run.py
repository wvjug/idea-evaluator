# import modules 
import torch
import warnings
from pathlib import Path
import argparse
from torch.utils.data import random_split
import dataset, model, trainer

if __name__ == "__main__":
    # default config.json location 
    cur_dir = Path(__file__).resolve().parent

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ckpt", help="path to model checkpoint", default = cur_dir/"model.pth"
    )
    parser.add_argument(
        "--data_dir", help="path to data", default = cur_dir/"data.csv"
    )
    parser.add_argument(
        "--mode", help="train or eval", default = "eval"
    )
    args = parser.parse_args()
    
    # read command args 
    model_ckpt = Path(args.model_ckpt).resolve()
    data_dir = Path(args.data_dir).resolve()
    
    # warning if not using cuda 
    if not torch.cuda.is_available():
        warnings.warn(f"Cuda: {torch.cuda.is_available()}. Cuda is not available. May run model much slower than expected.")

    # model config and model 
    bert_model_name = 'bert-base-uncased'
    num_classes = 9
    mconf = model.BERTClassifierConfig(num_classes, bert_model_name)
    classifier = model.BERTClassifier(mconf)

    train_data = dataset.HackathonDataset(data_dir)

    train_size = int(len(train_data) * 0.8)
    valid_size = len(train_data) - train_size
    train_dataset, valid_dataset = random_split(train_data, [train_size, valid_size])

    # train: 25 epochs, batch size 4
    tconf = trainer.TrainerConfig(max_epochs=25, batch_size=4, ckpt_path=model_ckpt, tokenizer_name=bert_model_name)
    t = trainer.Trainer(classifier, train_dataset, valid_dataset, tconf)
    t.train()