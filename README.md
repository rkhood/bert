# BERT

Using Google's [BERT](https://arxiv.org/abs/1810.04805) transformer model to create contextualised sentence embeddings.  Three methods utilising BERT are included:

1. Using [Hugging Face's](https://github.com/huggingface/transformers) transformer package:
	
	`pip install transformers`
2. Batching with the above transformer package.
3. Using [BERT-as-service](https://github.com/hanxiao/bert-as-service), a sentence encoding service:

	`pip install bert-serving-server bert-serving-client`
