# 🏁 Skidpad Path Planning & Control in ROS 2 – FS-UK Simulation

This ROS 2 package simulates **skidpad path planning and control** using a **kinematic bicycle model**, developed by **Mahmoud Yasser** during his work with the **Ain Shams University Racing Team** for the **Formula Student UK (FS-UK)** competition.

---

## 📦 Package Overview

- ✅ **Skidpad path generator**
- ✅ **Kinematic bicycle model**
- ✅ **Lateral control** via **Pure Pursuit**
- ✅ **Longitudinal control** via **PID**
- ✅ **Visualization** using **RViz2**

---

## 🧠 Control Algorithms

### 🔹 Pure Pursuit (Lateral Control)
A geometric controller that computes steering angle based on a lookahead point.

### 🔹 PID Controller (Longitudinal)
Maintains target speed by computing throttle/brake commands.

---

## 🚘 Vehicle Model

A simplified **Kinematic Bicycle Model** is used to simulate a Formula Student vehicle’s planar motion for low-speed dynamic behavior, such as the skidpad maneuver.

---

## 🛠️ Requirements

- ROS 2 (Humble or Foxy recommended)
- RViz2
- `colcon` build system

---

## 🔧 Installation & Build

```bash
# Clone the repository inside your ROS 2 workspace
cd ~/ros2_ws/src
git clone https://github.com/mahmoudyasser32/Skidpad-Path-Planning---Control-for-Formula-Student---Kinematic-Bicycle-Model--ASU-Racing-Team-.git

# Build the workspace
cd ~/ros2_ws
colcon build
source install/setup.bash
