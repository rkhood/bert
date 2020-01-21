import numpy as np
import torch
from transformers import BertModel, BertTokenizer


def batch(data, batch_size=100):
    n_batches = int(np.ceil(data.shape[0] / batch_size))
    for i in range(n_batches):
        yield data[i * batch_size:(i + 1) * batch_size]


def pad(tokens):
    max_len = 0
    for i in tokens.values:
        if len(i) > max_len:
            max_len = len(i)
    return np.array([i + [0]*(max_len-len(i)) for i in tokens.values])


def get_model(output_dir):
    model_state_dict = torch.load(output_dir)
    bert_model = 'bert-base-uncased',

    model = BertModel.from_pretrained(
            bert_model,
            state_dict=model_state_dict,
            )

    tokeniser = BertTokenizer.from_pretrained(
            bert_model,
            state_dict=model_state_dict,
            )
    return model, tokeniser


def get_vectors(data, model, tokeniser):
    tokenised = data.apply(
            lambda x: tokeniser.encode(x, add_special_tokens=True))
    padded = pad(tokenised)
    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    model.cuda()
    with torch.no_grad():
        vec = model(
                input_ids,
                attention_mask=attention_mask,
                )[0]
    return vec[:,0,:].cpu().numpy()


def batch_vectors(data, model, tokeniser):
    vec = []
    for each_batch in batch(data):
        vec.append([bert_embeddings(
                each_batch,
                model,
                tokeniser,
                )])
    return vec
