# release/ — Tiny9DTact 发布目录

这里存放 **打包后的成品**：`Tiny9DTact.exe` + 它的依赖。exe 及其产物不进 git（见根目录 `.gitignore`），由构建脚本在本地生成。

## 怎么构建

在 Windows 开发机上，**双击仓库根目录的 `build_release.bat`**（或命令行运行）。脚本会：

1. 找到 PATH 里的 Python（就是平时跑 `_3_Shape_Reconstruction.py` 的那个解释器，无版本漂移）
2. 缺 PyInstaller 就自动装
3. 按 `Tiny9DTact.spec` 做 onedir 打包 → `dist/Tiny9DTact/`
4. 把数据文件拷到 exe 旁边，组装成 `release/Tiny9DTact/`
5. 顺手压一个 `release/Tiny9DTact_win64.zip` 方便分发

## 产物结构

```
release/
├── Tiny9DTact/                      ← 可整体拷给别人的文件夹
│   ├── Tiny9DTact.exe               双击运行
│   ├── _internal/                   Python 运行时 + numpy/opencv/pywebview 等（勿动）
│   ├── shape_reconstruction/
│   │   ├── shape_config.yaml        可直接改（camera_channel / fps / 分辨率）
│   │   └── calibration/             标定数据（sensor_118/...），重标定后覆盖即可
│   ├── web_viewer/index.html        前端页面（视觉调参常量在文件顶部）
│   └── 使用说明.txt                 给拿到这个文件夹的人看
├── Tiny9DTact_win64.zip             同上内容打包
└── README.md                        本文件
```

配置 / 标定 / 前端常量都放在 exe **外面**，改完重启 exe 即生效，不需要重新打包；只有改了 Python 代码才需要重跑 `build_release.bat`。

## 目标机器要求

- Windows 10 / 11，无需安装 Python
- WebView2 Runtime：Win11 自带；Win10 缺失时程序会回退系统浏览器（建议[安装](https://developer.microsoft.com/microsoft-edge/webview2/)）
- **首次运行需联网**（前端从 CDN 加载 three.js，之后 WebView2 会缓存，可离线复现）

## 已知注意事项

- **杀毒软件误报**：PyInstaller 打包的 exe 偶尔被 Defender/其他杀软标记，属常见误报；加白名单即可。介意的话可对 exe 做代码签名
- 窗口后面会跟一个**控制台黑框**（保留重连/采集日志，排障用）。想要纯窗口版：把 `Tiny9DTact.spec` 里 `console=True` 改成 `False`，重跑 bat
- onedir（exe + `_internal` 文件夹）而不是单文件 exe：cv2 + numpy 的单文件版每次启动要先解压几百 MB 到临时目录，慢且易被杀软盯上；onedir 启动快、行为稳定
- 帧率、Z_SCALE 等视觉参数的调整方式与源码运行完全一致，见根目录 README 的「视觉调参」
