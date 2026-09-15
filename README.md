# Tiny9DTact — 9DTact Windows 高分辨率实时重建

> [9DTact](https://linchangyi1.github.io/9DTact/) 视触觉传感器的 Windows 高分辨率移植版，聚焦实时三维形状重建：原生桌面窗口内的 WebGL 实时渲染，约 87 万顶点高度图目标 30 FPS，拔线不崩溃、重插自动恢复。

## 特性

- **实时 3D 重建**：USB 相机 → 畸变矫正裁剪 → 像素-深度标定映射 → WebGL 网格，jet 鲜艳配色（深蓝基底）
- **桌面应用形态**：pywebview + Edge WebView2 打包为原生窗口，不再是浏览器标签页；GPU 强制开启（绕过集成显卡黑名单）
- **自动参考帧采集**：程序启动后自动等待画面稳定（跳过暖机帧、检测帧间差分），无需按键
- **断线韧性**：拔掉传感器不崩溃——3D/2D 视图黑屏、状态灯转红；重新插入后自动重开相机、重采参考帧、恢复推流
- **毛刺抑制与轮廓增强**：时间域 EMA + 噪声底吸附 + USM 锐化 + 色彩 gamma，浅压痕（如按压字符）轮廓清晰
- **轻量二进制协议**：WebSocket 推送 ~1.7 MB/帧的紧凑格式，带背压丢帧（慢客户端不累积延迟）
- **工程化**：pytest 覆盖协议编解码与读数计算；设计文档见 `docs/superpowers/specs/`

## 环境要求

| 项 | 要求 |
| --- | --- |
| 操作系统 | Windows 10 / 11 |
| WebView2 Runtime | Win11 自带；Win10 若缺失自动回退系统浏览器，建议[安装](https://developer.microsoft.com/microsoft-edge/webview2/) |
| Python | ≥ 3.10（开发验证于 3.12.10，全局解释器即可，无需 conda） |
| 相机 | USB 相机，默认 1280×720 @ 30 fps（`camera_channel: 0`） |

## 安装

```bash
git clone https://github.com/Purer-lyk/Tiny9DTact.git
cd Tiny9DTact
pip install -r requirements.txt
```

只要跑实时重建的话，最小依赖集如下（标定脚本额外用到 scipy；**不需要** torch / ROS / open3d / PyQt5 / vispy）：

```bash
pip install numpy opencv-python PyYAML websockets pywebview scipy
```

> 首次运行桌面窗口需联网下载 three.js（CDN）；之后 WebView2 会缓存，离线可复现。

## 快速开始

```bash
python _3_Shape_Reconstruction.py
```

启动后自动等待画面稳定并采集参考帧（控制台会打印稳定耗时），然后弹出 9DTact Shape Reconstruction 桌面窗口开始实时重建。关闭窗口即退出。

## 标定流程

新传感器 / 更换相机 / 重灌凝胶后需要重新标定。两步均为交互式，按控制台提示操作（`y` 键确认采集）。标定产物保存在 `shape_reconstruction/calibration/sensor_<id>/` 下，随仓库一起管理。

### 1. 相机标定

先手动调焦：旋转镜头，使约 15 mm 远的物体清晰。标定板（上游仓库 `9DTact_Design/fabrication/calibration_board.STL`）3D 打印备用。

```bash
python _1_Camera_Calibration.py
```

采集参考图与多组按压图后，生成 `camera_calibration/` 下的 `row_index.npy`、`col_index.npy`、`position_scale.npy`（畸变矫正索引与像素-毫米 scale）。

### 2. 深度标定

准备钢球，半径与 `shape_config.yaml` 的 `BallRad` 一致。

```bash
python _2_Sensor_Calibration.py
```

先采无接触参考，再用钢球以不同力度按压采样，生成 `depth_calibration/Pixel_to_Depth.npy`（灰度差-深度查找表）。

## 运行模式

| 命令 | 说明 |
| --- | --- |
| `python _3_Shape_Reconstruction.py` | 桌面窗口（默认，推荐） |
| `python _3_Shape_Reconstruction.py --browser` | 用系统浏览器打开（无 pywebview 时的回退） |
| `python _3_Shape_Reconstruction.py --replay <dir>` | 回放目录中的 `frame_*.png`，无传感器开发调试用 |
| `python _3_Shape_Reconstruction.py --fps 60` | 目标帧率（默认 30） |
| `set 9DTACT_DEBUG=1 && python _3_Shape_Reconstruction.py` | 打开 DevTools 调试窗口 |

## 查看器说明

**3D 视图**（左侧）：左键拖拽旋转、滚轮缩放、右键平移；右侧按钮切换预设视角（Default 45° / Top / Side / Reset）。坐标轴位于网格左下角，显示方向与物理传感器一致（相机镜像已在解码端翻转）。

**2D 深度图**（右上）：俯视深度热图 + 右侧色标（顶部为当前 `depth_max`）。

**读数面板**：Max depth（当前帧最大深度 mm）、Contact area（接触面积 mm²）、FPS（采集端帧率）、Render FPS（渲染帧率）、Latency（编码到上屏延迟 ms）、GPU（实际 WebGL 渲染器，若显示 SwiftShader 说明落到了软件渲染）、Sensor 状态灯（LIVE / UNPLUGGED / RECONNECTING）。

**断连行为**：拔掉相机后 2D/3D 变黑、状态灯红色 UNPLUGGED；重新插入后转橙色 RECONNECTING，自动重采参考帧后恢复 LIVE，全程无需重启程序。

## 视觉调参

渲染端常量在 `web_viewer/index.html` 顶部，按需调整后重启程序：

| 常量 | 默认 | 作用 |
| --- | --- | --- |
| `Z_SCALE` | 6.0 | 深度视觉放大倍数（物理仅 1–2 mm，不放大则不可见） |
| `TEMPORAL_ALPHA` | 0.45 | 时间域 EMA 权重，调低更平滑但更迟滞 |
| `NOISE_FLOOR_MM` | 0.05 | 低于此深度吸附为 0，保证未接触区干净 |
| `USM_RADIUS` / `USM_AMOUNT` | 4 / 1.2 | USM 锐化半径与强度，增强按压字符等浅轮廓 |
| `COLOR_GAMMA` | 0.75 | 色彩 gamma，调低则浅深度更亮更艳 |

## 配置说明

`shape_reconstruction/shape_config.yaml` 关键字段：

| 字段 | 说明 |
| --- | --- |
| `sensor_id` | 标定数据目录编号（`calibration/sensor_<id>/`） |
| `camera_setting.camera_channel` | 相机索引（0/1，多相机时按需修改） |
| `camera_setting.resolution` / `fps` | 采集分辨率与帧率 |
| `camera_calibration.crop_size` | 矫正后裁剪尺寸（当前 1240×690 高分辨率） |
| `depth_calibration.BallRad` | 深度标定钢球半径 |
| `sensor_reconstruction.*` | 高度图计算参数：光照阈值、平滑核、深度映射系数 |

## 架构

```
USB 相机 ──► Camera (矫正+裁剪) ──► Sensor (高度图) ──► compute_readings
                                                              │
                                    WebSocketVisualizer ──────┘
                                    ws://127.0.0.1:8765 二进制帧
                                    （背压丢帧，慢客户端不累积延迟）
                                                              │
                                    pywebview / Edge WebView2
                                    Three.js 网格渲染 ──────────┘
```

**为什么是 WebSocket + Three.js**：早期的 PyQt5 + Open3D/VisPy/PyQtGraph 方案在集成 GPU 上全部遇到 PyOpenGL 管线瓶颈；把 GL 留给浏览器引擎后彻底绕开，并叠加了以下优化——中心差分法线替代通用 `computeVertexNormals`（约 10× 提速）、`setPixelRatio(1)` 限制填充率、WebView2 GPU 参数强制启用。

**线协议**（小端）：50 字节头 + `H×W` 个 uint16 深度（mm×100）：

| 偏移 | 类型 | 字段 |
| --- | --- | --- |
| 0 | u32 | magic `0x395D534D` |
| 4 | u32 | frame_index |
| 8 | f64 | timestamp_ms |
| 16 / 18 | u16 / u16 | H / W |
| 20–44 | 7×f32 | pixel_per_mm, max_depth, area_mm², cx, cy, fps, depth_max |
| 48 | u8 | status（0 实时 / 1 已拔线 / 2 重连中） |
| 49 | u8 | pad（保证深度数组 2 字节对齐） |

详细设计取舍见 [docs/superpowers/specs/2026-07-29-web-viewer-design.md](docs/superpowers/specs/2026-07-29-web-viewer-design.md)。

## 测试

```bash
python -m pytest tests/ -q
```

覆盖线协议编解码往返（含断连状态帧）与读数计算（最大深度 / 接触面积 / 接触中心）。

## 目录结构

```
_1_Camera_Calibration.py    相机标定（交互式）
_2_Sensor_Calibration.py    深度标定（钢球，交互式）
_3_Shape_Reconstruction.py  实时重建入口（桌面窗口/浏览器/回放）
shape_reconstruction/       核心：camera.py / sensor.py / visualizer*.py + 标定数据
web_viewer/                 Three.js 前端（index.html）
tests/                      pytest 协议与读数测试
docs/superpowers/specs/     设计文档
force_estimation/           6D 力估计（继承自上游，需 ROS，本环境未适配）
data_collection/           力数据采集（同上）
model/ saved_models/        力估计网络定义与预训练权重（Densenet-169）
```

## 仓库约定

- 默认分支 `main`；功能分支 `feat/<topic>`、修复 `fix/<topic>`、文档 `docs/<topic>`
- 行尾统一 LF（`.gitattributes`），勿提交 CRLF
- 提交由 `ClaudePartner` 机器人账号代为完成，仓库所有者负责审核与推送（GitHub Desktop）
- 标定数据（`calibration/`）与预训练权重（`saved_models/`）随仓库管理

## 上游与引用

本仓库 fork 自 [9DTact](https://linchangyi1.github.io/9DTact/)（RAL 2023），力估计与 ROS 部分的完整说明请参考[上游仓库](https://github.com/linchangyi1/9DTact)。

**DTact 系列论文**：

- [DTact: A Vision-Based Tactile Sensor that Measures High-Resolution 3D Geometry Directly from Darkness](https://arxiv.org/abs/2209.13916), Lin et al., ICRA 2023
- [9DTact: A Compact Vision-Based Tactile Sensor for Accurate 3D Shape Reconstruction and Generalizable 6D Force Estimation](https://arxiv.org/abs/2308.14277), Lin et al., RAL 2023
- [Design and Evaluation of a Rapid Monolithic Manufacturing Technique for a Novel Vision-Based Tactile Sensor: C-Sight](https://www.mdpi.com/1424-8220/24/14/4603), Fan et al., MDPI Sensors 2024
- [DTactive: A Vision-Based Tactile Sensor with Active Surface](https://arxiv.org/abs/2410.08337), Xu et al., arxiv 2024
- [VET: A Visual-Electronic Tactile System for Immersive Human-Machine Interaction](https://arxiv.org/pdf/2503.23440), Zhang et al., arxiv 2025
- [PP-Tac: Paper Picking Using Tactile Feedback in Dexterous Robotic Hands](https://arxiv.org/abs/2504.16649), Lin et al., RSS 2025
- [AllTact Fin Ray: A Compliant Robot Gripper with Omni-Directional Tactile Sensing](https://arxiv.org/pdf/2504.18064), Liang et al., arxiv 2025

```bibtex
@inproceedings{lin2023dtact,
  title={Dtact: A vision-based tactile sensor that measures high-resolution 3d geometry directly from darkness},
  author={Lin, Changyi and Lin, Ziqi and Wang, Shaoxiong and Xu, Huazhe},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={10359--10366},
  year={2023},
  organization={IEEE}
}
```

```bibtex
@article{lin20239dtact,
  title={9dtact: A compact vision-based tactile sensor for accurate 3d shape reconstruction and generalizable 6d force estimation},
  author={Lin, Changyi and Zhang, Han and Xu, Jikai and Wu, Lei and Xu, Huazhe},
  journal={IEEE Robotics and Automation Letters},
  volume={9},
  number={2},
  pages={923--930},
  year={2023},
  publisher={IEEE}
}
```
