import onnx
from onnx import TensorProto, ValueInfoProto
from onnx import helper

model = onnx.load('/home/zqiu/models/densenet121.onnx')

custom_attribute_key = "custom_key"
custom_attribute_value = "custom_value"

custom_attribute = onnx.StringStringEntryProto(key=custom_attribute_key, value=custom_attribute_value)

if model.metadata_props is None:
    model.metadata_props = {}

model.metadata_props.append(custom_attribute)

onnx.save(model, "./test1.onnx")
