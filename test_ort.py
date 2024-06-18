import matplotlib.pyplot as plot
import numpy as np
import onnxruntime as ort
import os

models = os.path.join('build', 'DeepFilterNet3', 'tmp', 'export')

enc = ort.InferenceSession(os.path.join(models, 'enc.onnx'))
erb_dec = ort.InferenceSession(os.path.join(models, 'erb_dec.onnx'))
df_dec = ort.InferenceSession(os.path.join(models, 'df_dec.onnx'))

for name, model in dict(enc=enc, erb_dec=erb_dec, df_dec=df_dec).items():

    print(name)

    print('INPUTS')
    for x in model.get_inputs():
        print(x.name, x.shape, x.type)

    print('OUTPUTS')
    for y in model.get_outputs():
        print(y.name, y.shape, y.type)

    print()

# feat_erb = np.zeros((1, 1, 1, 32), np.float32)
# feat_spec = np.zeros((1, 2, 1, 96), np.float32)

# enc_output = enc.run(
#     ['e0', 'e1', 'e2', 'e3', 'c0', 'emb'],
#     {'feat_erb': feat_erb, 'feat_spec': feat_spec})

# print(type(enc_output))
