"""
将类激活图与地形图相结合，实现脑电模型可视化
代码:类激活图(CAM)，然后CAT
"""

import os
import numpy as np
import torch
import pandas as pd
from einops import rearrange, reduce, repeat
from matplotlib import pyplot as plt
from torch.backends import cudnn
from utils import GradCAM, show_cam_on_image
from our_model.CGDTrans import CascadedGroupTransformer

gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cudnn.benchmark = False
cudnn.deterministic = True

npy_data = np.load("/home/guyue/zhao/PythonProject/dataset/Self-data/kFold/kFold(61x750)/10fold-segment-data/test/test0.npy", allow_pickle=True).item()
data = np.array(npy_data['fea'])
data = np.expand_dims(data, axis=1)
# print(np.shape(data))
# print(type(data))

nSub = 0
target_category = 1  # 设置类(类激活映射)


# ! 这是适应Transformer的关键一步
# reshape_transform  b 61 750 -> b 750 1 61
def reshape_transform(tensor):
    result = rearrange(tensor, 'b (h w) e -> b e (h) (w)', h=1)
    return result


device = torch.device("cpu")
model = CascadedGroupTransformer(num_heads=3, dim=61, d_ff=512, n_layers=1, max_len=750).to(device)

model.load_state_dict(
    torch.load('/home/guyue/zhao/PythonProject/Transformer/CGDTrans/self-data/segment/result/model/pre_2/sub_%d.pth' % nSub,
               map_location=device))
target_layers = model.transformer_blocks  # 设置目标层
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False, reshape_transform=reshape_transform)

# TODO: 类激活地形图 (在论文中提出)
import mne

biosemi_montage = mne.channels.read_custom_montage('/home/guyue/zhao/PythonProject/Transformer/visualization/self-data.locs')
index = [1, 60, 2, 50, 36, 37, 51, 11, 44, 3, 30, 17, 31, 4, 45, 12, 58, 52, 25, 38, 21, 22, 39, 26, 53, 59, 13, 46, 5,
         32, 18, 33, 6, 47, 14, 54, 27, 40, 23, 61, 24, 41, 28, 55, 15, 48, 7, 34, 19, 35, 8, 49, 16, 56, 42, 29, 43,
         57, 9, 20, 10]  # correspond channel
biosemi_montage.ch_names = [biosemi_montage.ch_names[i - 1] for i in index]
# print(biosemi_montage.ch_names)
biosemi_montage.dig = [biosemi_montage.dig[i - 1] for i in index]
# print(biosemi_montage.dig)
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=750., ch_types='eeg')  # sample rate

all_cam = []
# 该循环用于获取每个试验/样品的CAM
for i in range(data.shape[0]):
    test = torch.as_tensor(data[i:i + 1, :, :, :], dtype=torch.float32)
    # print(test.shape)
    test = torch.autograd.Variable(test, requires_grad=True)

    grayscale_cam = cam(input_tensor=test)
    grayscale_cam = grayscale_cam[0, :]
    all_cam.append(grayscale_cam)

# 所有数据的平均值
test_all_data = np.squeeze(np.mean(data, axis=0))
# test_all_data = (test_all_data - np.mean(test_all_data)) / np.std(test_all_data)
mean_all_test = np.mean(test_all_data, axis=1)

# 所有CAM的均值
test_all_cam = np.mean(all_cam, axis=0)
# test_all_cam = (test_all_cam - np.mean(test_all_cam)) / np.std(test_all_cam)
mean_all_cam = np.mean(test_all_cam, axis=1)

# 对输入数据应用CAM
hyb_all = test_all_data * test_all_cam
# hyb_all = (hyb_all - np.mean(hyb_all)) / np.std(hyb_all)
mean_hyb_all = np.mean(hyb_all, axis=1)

df = pd.DataFrame({'Column1': mean_all_test, 'Column2': mean_all_cam, 'Column3': mean_hyb_all})
df.to_csv('/home/guyue/zhao/PythonProject/Transformer/visualization/cat-results/sub_0.csv', index=False)

evoked = mne.EvokedArray(test_all_data, info)
evoked.set_montage(biosemi_montage)

# fig, [ax1, ax2] = plt.subplots(nrows=2, figsize=(6, 4))
fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(6, 4))

plt.subplot(121)
im1, cn1 = mne.viz.plot_topomap(mean_all_test, evoked.info, show=False, axes=ax1)
plt.colorbar(im1)

plt.subplot(122)
im2, cn2 = mne.viz.plot_topomap(mean_hyb_all, evoked.info, show=False, axes=ax2)
plt.colorbar(im2)

# plt.show()
plt.savefig('/home/guyue/zhao/PythonProject/Transformer/visualization/cat-results/my_cat_1.png', dpi=1200)
print('the end')
