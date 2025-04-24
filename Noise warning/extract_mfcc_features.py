import librosa
import numpy as np

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# 假设你有一个包含所有音频文件路径及其对应类别的列表 data
features = []
for file_path, class_label in data:
    mfcc_feature = extract_mfcc(file_path)
    features.append([mfcc_feature, class_label])

np.save('dataset.npy', features)