"""

"""

import os
import numpy as np
import torch
from torch.backends import cudnn
import matplotlib.pyplot as plt
from our_model.CGDTrans import CascadedGroupTransformer
from visualization.tSNE import plt_raw_tsne, plt_target_tsne

gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cudnn.benchmark = False
cudnn.deterministic = True

nSub = 0

npy_data = np.load("/home/guyue/zhao/PythonProject/dataset/Self-data/kFold/kFold(61x750)/10fold-segment-data/test/test0.npy", allow_pickle=True).item()
data = np.array(npy_data['fea'])
label = np.array(npy_data['label'])
print(np.shape(data))
# print(type(data))

device = torch.device("cpu")
model = CascadedGroupTransformer(num_heads=3, dim=61, d_ff=512, n_layers=1, max_len=750).to(device)

model.load_state_dict(
    torch.load('/home/guyue/zhao/PythonProject/Transformer/CGDTrans/self-data/segment/result/model/pre_2/sub_%d.pth' % nSub,
               map_location=device))

# model_layer = getattr(model, '')
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
raw_data = torch.as_tensor(data, dtype=torch.float32)
raw_label = torch.as_tensor(label, dtype=torch.float32)
plt_raw_tsne(axs[0], raw_data, raw_label, per=30)

target_data = np.expand_dims(data, axis=1)
target_data = torch.as_tensor(target_data, dtype=torch.float32)
target_label = torch.as_tensor(label, dtype=torch.float32)
output_data = model(target_data)
plt_target_tsne(axs[1], output_data, target_label, per=30)

plt.show()
# plt.savefig('/home/guyue/zhao/PythonProject/Transformer/visualization/tSNE-results/tsne', dpi=1200)
# print('the end')
