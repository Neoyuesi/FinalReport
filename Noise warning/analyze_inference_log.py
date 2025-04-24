import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------ 配置 ------------------
log_csv_path = "inference_log.csv"  # 推理保存的日志
save_dir = "./analysis_results"     # 图保存目录

# ------------------ 读取日志 ------------------
df = pd.read_csv(log_csv_path)

# ------------------ 创建保存文件夹 ------------------
os.makedirs(save_dir, exist_ok=True)

# ------------------ 分类结果统计 - 饼图 ------------------
def plot_class_distribution(df):
    class_counts = df['classification'].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Noise Type Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(save_dir, "class_distribution_pie.png"))
    plt.show()
    print("分类统计饼图已保存。")

# ------------------ 声级参数分布 - 直方图 ------------------
def plot_sound_level_histograms(df):
    params = ['Leq', 'Lmax', 'Lmin', 'Lpeak']
    for param in params:
        plt.figure(figsize=(8, 6))
        plt.hist(df[param], bins=20, edgecolor='black')
        plt.title(f'{param} Distribution')
        plt.xlabel(f'{param} (dB)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{param}_histogram.png"))
        plt.show()
        print(f"{param}直方图已保存。")

# ------------------ Leq vs Lmax 散点图 ------------------
def plot_leq_lmax_scatter(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['Leq'], df['Lmax'], alpha=0.7, edgecolors='w')
    plt.xlabel('Leq (dB)')
    plt.ylabel('Lmax (dB)')
    plt.title('Scatter Plot of Leq vs Lmax')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "scatter_leq_lmax.png"))
    plt.show()
    print("Leq vs Lmax 散点图已保存。")

# ------------------ 每类噪声的平均 Leq 柱状图 ------------------
def plot_classwise_leq_bar(df):
    grouped = df.groupby('classification')['Leq'].mean().sort_values()

    plt.figure(figsize=(10, 6))
    grouped.plot(kind='barh', color='skyblue', edgecolor='black')
    plt.xlabel('Average Leq (dB)')
    plt.title('Average Leq per Noise Type')
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bar_leq_by_class.png"))
    plt.show()
    print("按类别平均Leq柱状图已保存。")

# ------------------ 主运行 ------------------
if __name__ == "__main__":
    plot_class_distribution(df)
    plot_sound_level_histograms(df)
    plot_leq_lmax_scatter(df)
    plot_classwise_leq_bar(df)
    print("所有分析图表绘制完成！")
