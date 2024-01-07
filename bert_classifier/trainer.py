import torch 
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
import torch.optim as optim
from torch import nn 
from typing import Optional

class TrainerConfig:
    """Trainer config, params common to all trainer versions

    @var max_epochs (int)
    @var batch_size (int)
    @var learning_rate (float)
    @var ckpt_path (str): checkpoint for the model 
    @var num_workers (int): number of workers for dataloader 
    @var criterion (nn.Module): loss function 
    @var tokenizer (BertTokenizer): tokenizer that tokenize sentences 
    """
    # optimization parameters
    max_epochs = 10
    batch_size = 4
    learning_rate = 3e-4

    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    criterion = nn.CrossEntropyLoss()
    
    def __init__(self, tokenizer_name, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    """Train and evaluate the model 
    """
    def __init__(self, model:nn.Module, train_dataset:Optional[Dataset], test_dataset:Optional[Dataset], config:TrainerConfig) -> None:
        """
        @param model (nn.Module) : model that predict the scores of the solutions
        @param train_dataset (Dataset) : dataset for training  
        @param test_dataset (Dataset) : dataset for testing
        @param config (TrainerConfig) : configuration for the trainer 
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self) -> None:
        if self.config.ckpt_path is not None:
            ckpt_model = self.model.module if hasattr(self.model, "module") else self.model
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)
            print("Save the model!")

    def train(self) -> None:
        """train the model with train dataset and validate performance with test dataset"""
        assert self.train_dataset is not None

        model, config = self.model, self.config

        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

        min_test_loss = float("inf")
        for epoch in range(config.max_epochs):
            # train
            print(f"Epoch {epoch+1}")
            self._run_epoch('train', optimizer=optimizer)

            # validation 
            # if test loss strictly decreases, save the current model
            if self.test_dataset is None:
                self.save_checkpoint()
            else:
                test_loss = self._run_epoch('test')

                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    self.save_checkpoint()
            
    def _run_epoch(self, split: str, optimizer: Optional[optim.AdamW]=None):
        model, config = self.model, self.config

        # train mode: backprop
        # test mode: no backprop
        is_train = split == 'train'
        
        # mode of model 
        model.train(is_train)

        # obtain corresponding datasets 
        if is_train: 
            data = self.train_dataset 
        else:
            data = self.test_dataset 
        
        loader = DataLoader(data, batch_size=config.batch_size, num_workers=config.num_workers)

        losses = []
        correct = 0 
        total = 0
        pbar = tqdm(enumerate(loader), total=len(loader)) 
        
        for it, (sentences, labels) in pbar:
            # place data on the correct device
            encoding = config.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # move to GPU if necessary
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)
            attention_mask = attention_mask.to(self.device)

            # forward the model
            with torch.set_grad_enabled(is_train):
                outputs = model(input_ids, attention_mask=attention_mask)  # NOT USING INTERNAL CrossEntropyLoss
                flattened_logits = outputs.view(-1, 9)
                flattened_targets = labels.view(-1)
                # print(flattened_logits)
                # print(flattened_targets)
                loss = config.criterion(flattened_logits, flattened_targets)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                
            if is_train:
                assert optimizer is not None 
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                optimizer.step()

                # report progress
                pbar.set_description(f"iter {it}: train loss {loss.item():.5f}.")

            # test mode: collect labels and corresponding predictions for plot 
            
            predicted_labels = torch.argmax(flattened_logits, dim=1)
            correct += torch.sum(predicted_labels == flattened_targets)
            total += len(flattened_targets)

        # print(testing_grad)
        print(f"{split} loss: {np.mean(losses)}; min: {np.min(losses)}, median: {np.median(losses)}, max: {np.max(losses)}")
        print("{} accuracy: {:.2f}".format(split, correct/total))
        
        # return mean of losses across batches, outputs (labels and prediction)
        return np.mean(losses)
