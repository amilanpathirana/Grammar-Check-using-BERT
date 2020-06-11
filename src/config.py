import transformers

DATA_PATH="./input/cola.tsv"
BERT_PATH="./input/bert_base_uncased/"
MODEL_PATH="model.bin"

TRAIN_BATCH_SIZE=4
VALID_BATCH_SIZE=1
EPOCHS=2
MAX_LEN=512

TOKENIZER=transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)

