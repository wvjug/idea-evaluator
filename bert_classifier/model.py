import torch 
from torch import nn, Tensor
from transformers import BertModel # AdamW, get_linear_schedule_with_warmup

class BERTClassifierConfig:
    """BERT config, params common to all BERT versions
    @var num_classes (int): categories of classification 
    @var bert_model_name (str): pretrained model name to be downloaded 
    """
    pdrop = 0.1
    
    def __init__(self, num_classes:int, bert_model_name:str, **kwargs):
        self.num_classes = num_classes
        self.bert_model_name = bert_model_name

        for k,v in kwargs.items():
            setattr(self, k, v)

class MLP(nn.Module):
    def __init__(self, input_dim, tgt_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim*4), 
            nn.ReLU(),
            nn.Linear(input_dim*4, input_dim), 
            nn.ReLU(),
            nn.Linear(input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, tgt_dim)
        )
            
    def forward(self, x):
        return self.model(x)

class BERTClassifier(nn.Module):
    """Classifier that uses BERT to extract meanings and then fully connect to categories
    
    @var config (BERTClassifierConfig): config of the model
    @var bert (BertModel): BERT model for feature extraction with transformer architecture
    @var dropout (float): dropout ratio to improve robustness of the model 
    @var circular_economy (nn.Linear): fully connected layer to predict the score of criterion Circular Economy
    @var market_potentials (nn.Linear): fully connected layer to predict the score of criterion Market Potentials
    @var feasibility (nn.Linear): fully connected layer to predict the score of criterion Feasibility
    """
    def __init__(self, config: BERTClassifierConfig):
        super(BERTClassifier, self).__init__()
        self.config = config 
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.dropout = nn.Dropout(config.pdrop)
        self.circular_economy = MLP(self.bert.config.hidden_size, config.num_classes)
        self.market_potentials = MLP(self.bert.config.hidden_size, config.num_classes)
        self.feasibility = MLP(self.bert.config.hidden_size, config.num_classes)
        
    def forward(self, input_ids:Tensor, attention_mask:Tensor):
            # use bert to get extracted meanings of tokens 
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output

            x = self.dropout(pooled_output)

            # fully connected layers 
            c_logits = self.circular_economy(x)
            m_logits = self.market_potentials(x)
            f_logits = self.feasibility(x)

            # return: concatenate logit scores for three criteria
            concatenated_logits = torch.cat((c_logits.unsqueeze(1), m_logits.unsqueeze(1), f_logits.unsqueeze(1)), dim=1)
            return concatenated_logits