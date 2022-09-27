import onnx
import sys, os
import numpy as np
import copy
from onnx import TensorProto

from ttc import proc_gemm_ttc_ttt

sys.path.append(os.path.abspath('..'))
import values

def proc_gemm_ctt(model, node_id, node, attr):
    pass