from transformers.modeling_bert import BertConfig

param_dict = {}

param_dict['config'] = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

#print('param_dict:', param_dict)
