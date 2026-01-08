import onnx
model = onnx.load("model.onnx")
print("Inputs:")
for i in model.graph.input:
    print(i.name, i.type.tensor_type.shape)
print("Outputs:")
for o in model.graph.output:
    print(o.name, o.type.tensor_type.shape)