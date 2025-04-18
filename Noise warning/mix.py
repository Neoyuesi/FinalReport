import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from random import uniform, randint

# ================= 配置路径 =================
base_audio_path = Path(r"E:\bishe\UrbanSound8K\audio")
metadata_path = Path(r"E:\bishe\UrbanSound8K\metadata\UrbanSound8K.csv")
output_path = Path(r"E:\bishe\mixed_audio")

# ================= 类别映射 =================
CLASS_MAPPING = {
    'construction': ['drilling', 'jackhammer'],
    'traffic': ['car_horn', 'engine_idling', 'siren'],
    'human': ['children_playing', 'street_music']
}

# ================= 元数据加载与验证 =================
def load_metadata():
    """加载并验证元数据"""
    if not metadata_path.exists():
        raise FileNotFoundError(f"元数据文件未找到：{metadata_path}")
    
    metadata = pd.read_csv(metadata_path)
    required_columns = ['slice_file_name', 'fold', 'class']
    if not all(col in metadata.columns for col in required_columns):
        raise ValueError("元数据缺少必要字段，请检查数据集版本")
    return metadata

# ================= 文件路径生成 =================
def get_class_files(metadata, class_names):
    """根据元数据获取指定类别的有效文件路径"""
    valid_files = []
    filtered = metadata[metadata['class'].isin(class_names)]
    
    for _, row in filtered.iterrows():
        fold_dir = base_audio_path / f"fold{row['fold']}"
        file_path = fold_dir / row['slice_file_name']
        
        if file_path.exists():
            valid_files.append(file_path)
        else:
            print(f"警告：文件不存在 {file_path}")
    
    if not valid_files:
        raise ValueError(f"未找到任何 {class_names} 类别的有效文件")
    return valid_files

# ================= 音频处理核心 =================
def load_and_preprocess(file_path, target_length=4*22050):
    """加载并预处理音频"""
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        y = librosa.util.fix_length(y, size=target_length)
        return y / (np.max(np.abs(y)) + 1e-8)  # 安全归一化
    except Exception as e:
        print(f"加载失败：{file_path.name}，错误：{str(e)}")
        return None

def generate_mixed_audio(class_files, num_mixes=10):
    """生成混合音频"""
    output_path.mkdir(parents=True, exist_ok=True)
    
    for mix_id in range(1, num_mixes+1):
        # 随机选择源文件
        sources = {
            cls: np.random.choice(files)
            for cls, files in class_files.items()
        }
        
        # 加载音频
        audios = {cls: load_and_preprocess(path) for cls, path in sources.items()}
        if any(v is None for v in audios.values()):
            print("存在加载失败的音频文件，跳过当前混合")
            continue
            
        # 动态混合参数
        weights = {
            'construction': uniform(0.3, 0.6),
            'traffic': uniform(0.4, 0.8),
            'human': uniform(0.1, 0.3)
        }
        sum_weights = sum(weights.values())
        normalized_weights = {k: v/sum_weights for k, v in weights.items()}
        
        # 时间偏移（最大1秒）
        offset = randint(0, 22050)
        shifted_audios = {cls: np.roll(audio, offset) for cls, audio in audios.items()}
        
        # 混合与动态压缩
        mixed = sum(normalized_weights[cls] * audio for cls, audio in shifted_audios.items())
        mixed = np.tanh(mixed * 1.5)  # 压缩动态范围
        
        # 保存结果
        output_file = output_path / f"mix_{mix_id:03d}.wav"
        sf.write(output_file, mixed, 22050, subtype='PCM_16')
        print(f"已生成：{output_file}")

# ================= 主流程 =================
if __name__ == "__main__":
    # 加载元数据
    metadata = load_metadata()
    
    # 按类别筛选文件
    class_files = {
        cls: get_class_files(metadata, class_names)
        for cls, class_names in CLASS_MAPPING.items()
    }
    
    # 打印文件统计
    print("\n文件统计：")
    for cls, files in class_files.items():
        print(f"{cls.upper():<12}: {len(files)} 文件")
    
    # 生成混合音频
    generate_mixed_audio(class_files, num_mixes=10)
    print("\n===== 混合音频生成完成 =====")