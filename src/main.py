print("mainfile")

import pandas as pd
import config

from sklearn import model_selection
import dataclass

import model

from model import BERTModel
import compute
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch.nn as nn


import numpy as np
from sklearn import metrics
import torch

def run():
    dfx=pd.read_csv(config.DATA_PATH,delimiter="\t",header=None,names=['sentence_source','label','note','sentence'])
    dfx.fillna("none")
    dfx.label=dfx.label.apply(lambda x: 1 if x==1 else 0)


    df_train,df_valid=model_selection.train_test_split(
        dfx,
        test_size=0.1,
        stratify=dfx.label.values,
        random_state=111
    )

    df_train=df_train.reset_index(drop=True)
    df_valid=df_valid.reset_index(drop=True)

    df_train=df_train.drop(['note','sentence_source'],axis=1)
    df_valid=df_valid.drop(['note','sentence_source'],axis=1)

    print(df_train.head())

    train_dataset=dataclass.DataSet(
        sentences=df_train.sentence.values,
        labels=df_train.label.values
    )

    valid_dataset=dataclass.DataSet(
        sentences=df_valid.sentence.values,
        labels=df_valid.label.values
    )

    train_data_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_data_loader=torch.utils.data.DataLoader(
        valid_dataset,
        num_workers=1,
        batch_size=config.VALID_BATCH_SIZE
    )

    device=torch.device('cpu')
    model=BERTModel()
    model.to(device)


    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]


    num_train_steps=int(len(df_train)/config.TRAIN_BATCH_SIZE*config.EPOCHS)
    #print(optimizer_parameters[0])

    optimizer=AdamW(optimizer_parameters,lr=3e-5)

    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    model=nn.DataParallel(model)


    for epochs in range(config.EPOCHS):
        print("epoch start")
        compute.train(train_data_loader,model,optimizer,device,scheduler)
        outputs,targets=compute.validate(valid_data_loader,model,device)


        outputs=np.array(outputs) >=0.5

        accuracy= metrics.accuracy_score(targets,outputs)
        best_accuracy=0
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

if __name__== "__main__":
    run()



