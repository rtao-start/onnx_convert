import onnx
import sys, os
import numpy as np
import copy
from onnx import TensorProto

sys.path.append(os.path.abspath('..'))
import values

def proc_gemm_tct(model, node_id, node, attr):
    pass