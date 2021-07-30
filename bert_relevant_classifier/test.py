from model import relevantClassifier
from config import get_args
from transformers import RobertaTokenizerFast

args = get_args()
model = relevantClassifier(args)

s = 'Hey fuck you.'
tokenizer = RobertaTokenizerFast.from_pretrained('simonlevine/bioclinical-roberta-long')
tokens = tokenizer(s, return_attention_mask=True, return_tensors='pt')

outputs = model(tokens.input_ids, tokens.attention_mask)
print(outputs)
