import torch
from transformers import BertModel, BertTokenizer


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
    vectors = []
    for text in data:
        input_ids = torch.tensor(
                [tokeniser.encode(text, add_special_tokens=True)])

        # cls token
        with torch.no_grad():
            vectors.append(
                    model(input_ids)[0].squeeze().tolist()[0])

