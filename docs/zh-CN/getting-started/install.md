---
title: 安装 Koharu
description: 在本地构建并安装 Koharu。
---

# 安装 Koharu

此分支仅从源码在本地构建。先安装[开发环境](/zh-CN/development/setup/)中列出的依赖，然后运行：

```powershell
bun install
cargo tauri build --bundles nsis
```

Windows 安装程序会输出到 `target/release/bundle/nsis`。

## 首次启动

原生运行时准备完毕后，Koharu 会显示项目浏览器。首次启动可能需要下载原生运行时包，因此耗时更长。具体模型文件会在第一次使用该模型时解析。

下载需要访问 GitHub Release 资源；模型权重通常还需要访问 Hugging Face。进度显示在活动中心。软件包发布到本地缓存期间不要关闭应用。

下一步请[翻译第一个项目](/zh-CN/getting-started/first-project/)。硬件选择与缓存行为见[运行时、模型与硬件](/zh-CN/getting-started/runtime-models-and-hardware/)。
