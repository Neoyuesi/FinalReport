# Noise Monitoring and Sound Source Localization System
Real-time Environmental Noise Detection, Classification, Localization, and Anomaly Alerting System

## Overview

This project implements an intelligent environmental noise monitoring system based on multi-task deep learning and sound source localization. The system is designed for real-time classification of noise types (traffic, industrial, human, animal), regression of key sound level parameters (Leq, Lmax, Lmin, Lpeak), sound source localization using Time Difference of Arrival (TDOA) estimation, and anomaly detection with alerting capabilities.

The system architecture integrates acoustic sensors, machine learning models, cloud communication via MQTT, and a web-based real-time visualization dashboard.

## Features

**Multi-task Learning Model:** Simultaneous noise classification and sound level prediction using MFCC features.
- **Sound Source Localization:** TDOA-based estimation of acoustic event positions.
- **Anomaly Detection:** Automatic flagging of abnormal noise events based on dynamic thresholds.
- **Real-time Dashboard:** Live visualization of classified events, severity levels, and localized positions.
- **Cloud Integration:** MQTT communication for data upload and event notification.
- **OpenStreetMap-based Mapping:** Visualization of event locations within a specified area.

## Project Structure

```plaintext
project_root/
├── datasets/               # Preprocessed audio and metadata
├── models/                 # Trained multi-task models (.h5 files)
├── dashboard/              # Flask + Socket.IO real-time visualization app
├── localization/           # TDOA-based sound source localization modules
├── mqtt_client/            # MQTT publisher scripts for cloud transmission
├── simulation/             # Synthetic data generation and visualization
├── inference/              # Batch and real-time inference scripts
├── train_multitask_model.py       # Multi-task model training script
├── main_inference_fullsystem.py   # Real-time inference and dashboard runner
├── simulate_data.py                # Simulation and synthetic data generator
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation (this file)
└── LICENSE                 # License file (e.g., MIT License)
```

## Installation
1.Clone the repository:

git clone https://github.com/your-repo-link-here.git
cd your-repo-link-here

2.Install required Python libraries:

pip install -r requirements.txt

3.Configure MQTT broker address in mqtt_client/config.py if needed.

## Usage

1. Model Training (Optional)
If you want to retrain the model:

python train_multitask_model.py

2. Real-time Inference
Run the real-time inference and dashboard server:

python main_inference_fullsystem.py
Dashboard accessible at:
http://localhost:5000

3. Simulation Mode
Generate synthetic acoustic events for testing:

python simulate_data.py

## System Requirements

Python 3.8+
TensorFlow 2.8+
Flask + Flask-SocketIO
Paho-MQTT
OpenCV (optional for audio/video integration)
Recommended hardware:
CPU (Intel i7 or above) or GPU acceleration for faster model inference.

## Demo Screenshots

###  Real-time Noise Monitoring Dashboard

The dashboard below demonstrates real-time noise classification, sound source localization (via DOA angle), and anomaly detection with severity-level policy handling. Each acoustic event is visualized with its category, direction, and alarm status.

![Real-time Dashboard Overview](realtime_dashboard.png)

-  **Pie chart**: Distribution of noise types (Human, Traffic, Industrial, Animal)  
-  **Bar chart**: Count of detected events by category  
-  **Event Table**: Displays acoustic event details:
  - Type & Group
  - DOA (Direction of Arrival)
  - Severity Level
  - Leq (dB)
  - Alarm Status (flag + reason)

---

###  Acoustic Event Localization (OpenStreetMap-based)

The figure below shows the geo-spatial visualization of detected sound sources using TDOA-based localization overlaid on a map. Each point corresponds to an acoustic event (color-coded), with direction lines representing estimated arrival angles.

![Localization Map](localization_map.png)

>  Built using OpenStreetMap + Leaflet.js  
>  Dynamic plotting of localized acoustic events  
>  Useful for assessing spatial noise distribution in residential or industrial zones



## Citation

If you use this work or find it helpful for your research, please cite:

Noise Monitoring and Sound Source Localization System for Residential Environments, 2025.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
