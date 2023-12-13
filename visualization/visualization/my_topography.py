# plot EEG topograpy with mne
# https://mne.tools/stable/index.html
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
# get the data and label
# data - (samples, channels, trials)
# label -  (label, 1)
data = np.load(f"C:/Users/313/Desktop/visualization/sub0.npy", allow_pickle=True)
data= data.item()
data=np.array(data['fea'])

# data = np.transpose(data, (2, 1, 0))
# label = np.squeeze(np.transpose(label))
# idx = np.where(label == 1)
# data_draw = data[idx]
print(data.shape)

mean_trial = np.mean(data, axis=0)  # mean trial
# use standardization or normalization to adjust
mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)

mean_ch = np.mean(mean_trial, axis=1)  # mean samples with channel dimension left

# Draw topography
biosemi_montage = mne.channels.read_custom_montage('G:/Thinkpad文件/数据集/Standard-10-20-Cap19new/channels.locs')
index = [1,2,11,3,17,4,12,13,5,18,6,14,15,7,19,8,16,9,10]  # correspond channel
biosemi_montage.ch_names = [biosemi_montage.ch_names[i - 1] for i in index]
print(biosemi_montage.ch_names)
biosemi_montage.dig = [biosemi_montage.dig[i - 1] for i in index]
# print(biosemi_montage.dig)
info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=750., ch_types='eeg')  # sample rate

evoked1 = mne.EvokedArray(mean_trial, info)
evoked1.set_montage(biosemi_montage)

fig, ax = plt.subplots(figsize=(6, 4))
print(mean_ch)
im, cn = mne.viz.plot_topomap(mean_ch, evoked1.info, show=False, axes=ax)
plt.colorbar(im)
# plt.show()
plt.savefig('./test.png', dpi=1200)
print('the end')
