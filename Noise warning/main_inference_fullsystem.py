import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import tensorflow as tf
import pickle
from tqdm import tqdm
import datetime

# ---------------- 参数 ----------------
model_path = "multitask_noise_model.h5"            # 已训练好的模型
label_encoder_path = "label_encoder.pkl"            # 保存的标签编码器
audio_folder = "mixed_audio/"                       # 批量音频文件夹
log_csv_path = "inference_log.csv"                  # 保存推理日志
sr = 22050                                          # 采样率
duration = 4.0                                      # 每条音频时长
n_mfcc = 40
target_length = 174
threshold_db = 85                                   # 报警Leq阈值 dB

# ---------------- 加载模型和编码器 ----------------
print("Loading model and label encoder...")
model = tf.keras.models.load_model(model_path)

with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- 工具函数 ----------------

def compute_sound_levels(y, sr):
    """计算声压参数"""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    Leq = 10 * np.log10(np.mean(rms**2) + 1e-9)
    Lmax = 10 * np.log10(np.max(rms**2) + 1e-9)
    Lmin = 10 * np.log10(np.min(rms**2) + 1e-9)
    Lpeak = 10 * np.log10(np.max(np.abs(y)**2) + 1e-9)
    return Leq, Lmax, Lmin, Lpeak

def extract_mfcc(y, sr, n_mfcc, target_length):
    """提取MFCC特征"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    if mfcc.shape[1] < target_length:
        mfcc = np.pad(mfcc, ((0,0), (0,target_length - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :target_length]
    return mfcc.T[..., np.newaxis]

def classify_type(predicted_class):
    """根据分类分解类型"""
    human_keywords = ["human", "speech", "talking", "people"]
    traffic_keywords = ["traffic", "car", "engine", "road"]
    industrial_keywords = ["industrial", "machinery", "factory"]
    animal_keywords = ["animal", "dog", "bird", "cat"]

    if any(kw in predicted_class.lower() for kw in human_keywords):
        return "Human", "Strict Alarm"
    elif any(kw in predicted_class.lower() for kw in traffic_keywords):
        return "Traffic", "Normal Alarm"
    elif any(kw in predicted_class.lower() for kw in industrial_keywords):
        return "Industrial", "Strict Alarm"
    elif any(kw in predicted_class.lower() for kw in animal_keywords):
        return "Animal", "Ignore"
    else:
        return "Other", "Normal Alarm"

def determine_severity(leq_value, threshold=85):
    """根据 Leq 判断严重等级"""
    if leq_value >= threshold + 10:
        return "Severe"
    elif leq_value >= threshold + 5:
        return "Moderate"
    else:
        return "Mild"

# ---------------- 推理处理 ----------------

records = []

print("Processing audios...")
for file_name in tqdm(os.listdir(audio_folder)):
    if file_name.endswith(".wav"):
        file_path = os.path.join(audio_folder, file_name)
        try:
            # 加载音频
            y, _ = librosa.load(file_path, sr=sr, duration=duration)
            mfcc = extract_mfcc(y, sr, n_mfcc, target_length)
            mfcc = np.expand_dims(mfcc, axis=0)  # (1, target_length, n_mfcc, 1)

            # 推理
            preds = model.predict(mfcc)
            class_pred = np.argmax(preds[0], axis=1)[0]
            levels_pred = preds[1][0]

            # 分类标签
            predicted_class = label_encoder.inverse_transform([class_pred])[0]

            # 声级参数
            Leq, Lmax, Lmin, Lpeak = compute_sound_levels(y, sr)

            # 分类分组
            group_type, alarm_policy = classify_type(predicted_class)

            # 严重等级
            severity = determine_severity(Leq, threshold_db)

            # 是否异常
            abnormal = Leq >= threshold_db if alarm_policy != "Ignore" else False

            # 定位模拟 (默认放赫瑞瓦特大学附近，后续可改)
            lat = 55.9110 + np.random.uniform(-0.001, 0.001)
            lon = -3.3245 + np.random.uniform(-0.001, 0.001)
            doa_angle = np.random.uniform(0, 360)  # 方向角模拟

            # 记录
            records.append({
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "file": file_name,
                "classification": predicted_class,
                "group_type": group_type,
                "alarm_policy": alarm_policy,
                "severity_level": severity,
                "Leq": Leq,
                "Lmax": Lmax,
                "Lmin": Lmin,
                "Lpeak": Lpeak,
                "doa_angle": doa_angle,
                "abnormal": abnormal,
                "lat": lat,
                "lon": lon
            })

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# ---------------- 保存结果 ----------------

df = pd.DataFrame(records)
df.to_csv(log_csv_path, index=False)
print("All finished! Log saved to", log_csv_path)
