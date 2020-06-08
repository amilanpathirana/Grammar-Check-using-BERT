import config
import torch

class DataSet():
    def __init__(self,sentences,labels):
        self.sentences=sentences
        self.labels=labels
        self.max_len=config.MAX_LEN
        self.tokenizer=config.TOKENIZER

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self,item):
        sentence=str(self.sentences[item])
        sentence=" ".join(sentence.split())

        inputs=self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids=inputs['input_ids']
        mask=inputs['attention_mask']
        ttids=inputs['token_type_ids']

        return {
            "ids" : torch.tensor(ids,dtype=torch.long),
            "mask" : torch.tensor(mask,dtype=torch.long),
            "ttid" : torch.tensor(ttids,dtype=torch.long),
            "label" : torch.tensor(self.labels[item],dtype=torch.float)
        }


