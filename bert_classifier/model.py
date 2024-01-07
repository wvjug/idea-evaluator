import torch 
from torch import nn 
from transformers import BertModel # AdamW, get_linear_schedule_with_warmup

class BERTClassifierConfig:
    """BERT config, params common to all BERT versions """
    pdrop = 0.1
    
    def __init__(self, num_classes, bert_model_name, **kwargs):
        self.num_classes = num_classes
        self.bert_model_name = bert_model_name
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        for k,v in kwargs.items():
            setattr(self, k, v)

class BERTClassifier(nn.Module):
    def __init__(self, config: BERTClassifierConfig):
        super(BERTClassifier, self).__init__()
        self.config = config 
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.dropout = nn.Dropout(config.pdrop)
        self.circular_economy = nn.Linear(self.bert.config.hidden_size, config.num_classes)
        self.market_potentials = nn.Linear(self.bert.config.hidden_size, config.num_classes)
        self.feasibility = nn.Linear(self.bert.config.hidden_size, config.num_classes)
        
    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            c_logits = self.circular_economy(x)
            m_logits = self.market_potentials(x)
            f_logits = self.feasibility(x)
            concatenated_logits = torch.cat((c_logits.unsqueeze(1), m_logits.unsqueeze(1), f_logits.unsqueeze(1)), dim=1)
            return concatenated_logits