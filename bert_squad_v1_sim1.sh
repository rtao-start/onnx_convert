python model_convert.py   --model_path  /home/zqiu/models/bert_squad_metax_v1.pb  --model_type tf-graph  --output ./pb.onnx --inputs input_ids:0[1,128],input_mask:0[1,128],segment_ids:0[1,128]  --outputs start_prob:0,end_prob:0  --fuse_layernorm 0 --fuse_gelu 0


python ./dump.py


