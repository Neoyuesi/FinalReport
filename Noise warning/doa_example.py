import numpy as np
from scipy.optimize import least_squares

# 传感器位置 (x, y)
sensor_positions = np.array([
    [0, 0],   # 传感器 1 位置
    [1, 0],   # 传感器 2 位置
    [0, 1]    # 传感器 3 位置
])

# 传感器接收到声音的时间 (假设已知)
# 时间单位是秒，表示声波到达各传感器的时间
# 假设声速为 343 m/s（常温下的声速）
sound_speed = 343  # m/s

# 假设声源与各传感器的到达时间差 (单位：秒)
# TDOA（到达时间差） = 时间差 * 声速
# 假设声源位于 (x, y) = (0.5, 0.5) 处
true_source = np.array([0.5, 0.5])
distances = np.linalg.norm(sensor_positions - true_source, axis=1)
arrival_times = distances / sound_speed

# 模拟测量的 TDOA 数据
# 假设测量的误差比较小
TDOA_measurements = arrival_times - arrival_times[0]  # 相对时间差

# 定义目标函数，用于最小化TDOA与传感器位置的误差
def objective_function(source_position, sensor_positions, TDOA_measurements, sound_speed):
    distances = np.linalg.norm(sensor_positions - source_position, axis=1)
    arrival_times = distances / sound_speed
    TDOA_calculated = arrival_times - arrival_times[0]
    return TDOA_calculated - TDOA_measurements

# 使用最小二乘法来求解声源位置
initial_guess = np.array([0.5, 0.5])  # 初始猜测
result = least_squares(objective_function, initial_guess, args=(sensor_positions, TDOA_measurements, sound_speed))

# 输出计算得到的声源位置
estimated_source = result.x
print(f"Estimated source position: {estimated_source}")

