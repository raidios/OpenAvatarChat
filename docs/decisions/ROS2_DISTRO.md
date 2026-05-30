# ROS2 发行版决策

**结论：使用 ROS2 Jazzy（不是 Humble）。**

## 决策依据

- 部署机：RPi5 + Ubuntu **24.04** (Noble) aarch64。
- Humble 的官方支持平台是 Ubuntu 22.04（Jammy），24.04 上没有 binary apt 源，要么源码编译要么用 22.04 容器。
- Jazzy 是 24.04 LTS 上的 ROS2 LTS（2024-05 发布，支持到 2029-05），有官方 aarch64 binary。
- 我们的目标只是 ROS2 作为集成层（topic / 节点 / 标准消息类型），上下游是自己写的 Python 节点 + 既有 OpenAvatarChat handler；标准消息类型在 Humble/Jazzy 之间几乎无差异。
- 未来与底盘 / 激光雷达 / 深度相机统一时，Jazzy 也是 24.04 + LTS 的最佳选择。

## 安装方式

```bash
# 走官方 apt 源（aarch64 有 binary）
sudo apt install software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-ros-base python3-colcon-common-extensions python3-vcstool
```

环境激活（建议加到 `~/.bashrc`）：
```bash
source /opt/ros/jazzy/setup.bash
```

## 项目内 import 注意

- 我们的 ROS2 工作区在 `ros2_ws/src/`，用 `colcon build --symlink-install` 构建。
- 节点是 Python ament_python 包；`from live_stream import LiveStream` 通过 `pyproject.toml` 把 `thirdparty/M2_SDK/live_stream/host` 加进 `sys.path`。
- OpenAvatarChat 主进程仍在 `.venv` (Python 3.11)；ROS2 客户端 handler `src/handlers/client/ros2_client/` 用 `rclpy` 订阅 ROS2 topic。
- **rclpy 与 .venv 的兼容**：Jazzy 默认 Python 3.12，`.venv` 是 3.11。两者不直接兼容；首选方案：让 ROS2 节点（`audio_frontend_node` / `camera_node`）跑在系统 Python 3.12 + Jazzy 环境下；OpenAvatarChat 主进程（含 ros2_client handler）也切到 Jazzy 同 Python 环境（即把 `.venv` 改用 Python 3.12 重建，保留 hailo wheel 兼容性需测试）。
  - 备选：ROS2 节点跑 Jazzy，handler 通过 ZeroMQ/socket bridge 与之通信，不用 rclpy。
  - **本期采取首选方案**：先按需修 `.venv` 重建为 3.12（hailort 5.3.0 wheel 当前是 cp311，需要确认是否有 cp312 版或本地编译）。如果 hailort 仅 cp311 → 退回备选 ZeroMQ bridge。
- 此决策第一次接 Jazzy 实际安装时再最终验证；先行架构按"**节点跑 Jazzy + Python 3.12，handler 也跑同环境**"假设。
