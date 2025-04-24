import time
import random
import datetime
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# 校区中心位置
center_lat = 55.9110
center_lon = -3.3245

# 样例分类类型
sample_classes = [
    ("Human", "Strict Alarm"),
    ("Traffic", "Normal Alarm"),
    ("Industrial", "Strict Alarm"),
    ("Animal", "Ignore")
]

@app.route('/')
def index():
    return render_template('index.html')

def simulate_data_loop():
    while True:
        group_type, alarm_policy = random.choice(sample_classes)
        
        if group_type == "Human":
            classification = "speech"
        elif group_type == "Traffic":
            classification = "car_horn"
        elif group_type == "Industrial":
            classification = "machinery"
        else:
            classification = "dog_bark"

        leq = random.uniform(60, 100)  # Leq在60-100之间
        abnormal = leq >= 85 if alarm_policy != "Ignore" else False

        # 判断严重等级
        if leq >= 95:
            severity = "Severe"
        elif leq >= 90:
            severity = "Moderate"
        else:
            severity = "Mild"

        # 模拟 DOA 声源方向角度（0-360度）
        doa_angle = random.uniform(0, 360)

        payload = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
            "classification": classification,
            "group_type": group_type,
            "alarm_policy": alarm_policy,
            "severity_level": severity,
            "Leq": leq,
            "abnormal": abnormal,
            "lat": center_lat + random.uniform(-0.001, 0.001),
            "lon": center_lon + random.uniform(-0.001, 0.001),
            "doa_angle": doa_angle
        }

        socketio.emit('new_data', payload)
        print("Sent:", payload)
        time.sleep(1)  # 每秒推送一条数据

if __name__ == '__main__':
    socketio.start_background_task(simulate_data_loop)
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
