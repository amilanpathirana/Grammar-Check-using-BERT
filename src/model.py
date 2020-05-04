
import torch.nn as nn
import config
import transformers

class BERTModel(nn.Module):
    def __init__(self):
        super(BERTModel,self).__init__()
        self.bert=transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop=nn.Dropout(0.3)
        self.out=nn.Linear(768,1)


    def forward(self,ids,mask,ttid):
        _,o2=self.bert(ids,
            attention_mask=mask,
            token_type_ids=ttid
        )

        bo=self.bert_drop(o2)
        output=self.out(bo)

        return output



