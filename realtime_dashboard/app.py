from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pandas as pd
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app)

log_csv_path = "../inference_log.csv"  # 推理日志路径

def background_task():
    """后台线程，持续监控新的推理数据"""
    last_len = 0
    while True:
        try:
            df = pd.read_csv(log_csv_path)
            if len(df) > last_len:
                new_data = df.iloc[-1].to_dict()
                socketio.emit('new_data', new_data)
                last_len = len(df)
        except Exception as e:
            print("Error reading CSV:", e)
        time.sleep(2)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=background_task).start()
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
