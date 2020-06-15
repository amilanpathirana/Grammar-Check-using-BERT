import config
import torch
import flask
import time
from flask import Flask
from flask import render_template
from flask import request
from model import BERTModel
import functools
import torch.nn as nn
import joblib


app = Flask(__name__)

MODEL = None
DEVICE = "cpu"
PREDICTION_DICT = dict()
memory = joblib.Memory("./input/", verbose=0)




@memory.cache
def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review, None, add_special_tokens=True, max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(ids=ids, mask=mask, ttid=token_type_ids)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]


@app.route('/', methods=['GET','POST'])

def my_form_post():
    if request.method=="POST":
        text = request.form['text']
        processed_text = text.lower()
        text_file = open("inputs.txt", "a")
        text_file.write("Input Text: %s" % processed_text)
        text_file.write("\n")
        text_file.close()
        pred=sentence_prediction(processed_text)
        return render_template('index.html',prediction=pred)
    return render_template('index.html',prediction=0 )




if __name__ == "__main__":
    MODEL = BERTModel()
    MODEL = nn.DataParallel(MODEL)
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True)
