import os
from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
from caffe2onnx.src.caffe2onnx import Caffe2Onnx
from caffe2onnx.src.args_parser import parse_args
from caffe2onnx.src.utils import freeze


def main(args):
    prototxt_path = args.prototxt

    if args.caffemodel is None:
        caffemodel_path = f'{os.path.splitext(prototxt_path)[0]}.caffemodel'
    else:
        caffemodel_path = args.caffemodel

    if args.onnx is None:
        onnxmodel_path = f'{os.path.splitext(prototxt_path)[0]}.onnx'
    else:
        onnxmodel_path = args.onnx

    graph, params = loadcaffemodel(prototxt_path, caffemodel_path)
    c2o = Caffe2Onnx(graph, params, onnxmodel_path)
    onnxmodel = c2o.createOnnxModel(11) #qiuzy debug

    if args.frozen is True:
        freeze(onnxmodel)
    
    saveonnxmodel(onnxmodel, onnxmodel_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    