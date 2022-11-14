import onnx

def is_shared_init(model, init, node_name):
    for node in model.graph.node:
        if node.name != node_name:
            if init in node.input:
                return True

    return False            

def is_shared_constant(model, constant):
    count = 0
    for node in model.graph.node:
        if constant in node.input:
            count = count + 1

    if count > 1:
        return True            

    return False

def got_input_shape(mode, tensor):
    for vi in mode.graph.value_info:
        if vi.name == tensor:
            dim_proto_input = vi.type.tensor_type.shape.dim[0]
            print('got input shape: ', dim_proto_input.dim_value)
            return dim_proto_input.dim_value, True

    return -1000, False         