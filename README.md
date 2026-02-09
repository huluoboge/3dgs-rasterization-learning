# 3D Gaussian Splatting 学习资源

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个完整的 3D Gaussian Splatting (3DGS) 技术学习资源仓库

</div>

## 📚 项目简介

本项目旨在为学习 3D Gaussian Splatting 技术提供一个全面、系统的中文学习资源。通过理论文档、代码实现和完整示例，帮助你从零开始理解和掌握这项革命性的 3D 场景重建与渲染技术。

### 什么是 3D Gaussian Splatting？

3D Gaussian Splatting 是一种新型的实时辐射场渲染技术，由 INRIA 和 Max Planck Institute 于 2023 年提出。它使用 3D 高斯函数来表示场景，相比 NeRF 等方法具有以下优势：

- ⚡ **实时渲染**：支持高分辨率实时渲染（>30 FPS）
- 🎯 **高质量**：渲染质量与 NeRF 相当甚至更好
- 💾 **显式表示**：使用显式的 3D 高斯，易于理解和优化
- 🔧 **可微分**：完全可微分，支持梯度优化

## 🎯 学习目标

通过本项目，你将学习到：

1. 高斯函数的数学基础及其在图形学中的应用
2. 光栅化渲染的基本原理
3. 3D Gaussian Splatting 的核心算法
4. 如何从零实现一个简化版的高斯光栅化器
5. 性能优化和工程实现技巧

## 📖 学习路径

建议按以下顺序学习：

### 第一阶段：理论基础（1-2 周）
1. 阅读 [3DGS 简介](docs/01-introduction.md) - 了解技术背景
2. 学习 [高斯函数基础](docs/02-gaussian-basics.md) - 掌握数学基础
3. 理解 [光栅化原理](docs/03-rasterization.md) - 学习渲染流程

### 第二阶段：代码实践（2-3 周）
4. 运行 `code/basic/gaussian_2d.py` - 可视化 2D 高斯
5. 运行 `code/basic/gaussian_3d.py` - 理解 3D 高斯
6. 学习 `code/rasterization/simple_rasterizer.py` - 基础光栅化
7. 研究 `code/rasterization/gaussian_rasterizer.py` - 核心算法

### 第三阶段：综合应用（1 周）
8. 运行 `examples/simple_scene.py` - 完整场景渲染
9. 阅读 [实现细节](docs/04-implementation.md) - 深入理解
10. 查阅 [参考资料](docs/05-references.md) - 拓展学习

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 包管理器

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/huluoboge/3dgs-rasterization-learning.git
cd 3dgs-rasterization-learning

# 安装依赖
pip install -r requirements.txt
```

### 运行第一个示例

```bash
# 2D 高斯可视化
python code/basic/gaussian_2d.py

# 简单场景渲染
python examples/simple_scene.py
```

## 📁 项目结构

```
3dgs-rasterization-learning/
├── README.md                 # 项目介绍（本文件）
├── requirements.txt          # Python 依赖
├── docs/                     # 📖 文档目录
│   ├── 01-introduction.md    # 3DGS 技术简介
│   ├── 02-gaussian-basics.md # 高斯函数数学基础
│   ├── 03-rasterization.md   # 光栅化渲染原理
│   ├── 04-implementation.md  # 核心算法实现细节
│   └── 05-references.md      # 论文和参考资料
├── code/                     # 💻 代码实现
│   ├── basic/                # 基础示例
│   │   ├── gaussian_2d.py    # 2D 高斯分布可视化
│   │   └── gaussian_3d.py    # 3D 高斯可视化与变换
│   ├── rasterization/        # 光栅化实现
│   │   ├── simple_rasterizer.py      # 简单光栅化器
│   │   └── gaussian_rasterizer.py    # 高斯光栅化核心
│   └── utils/                # 工具函数
│       └── visualization.py  # 可视化辅助工具
└── examples/                 # 🎨 完整示例
    └── simple_scene.py       # 简单场景渲染示例
```

## 📚 文档概览

| 文档 | 内容 | 难度 |
|------|------|------|
| [01-introduction.md](docs/01-introduction.md) | 3DGS 技术背景、优势、应用场景 | ⭐ |
| [02-gaussian-basics.md](docs/02-gaussian-basics.md) | 高斯函数、协方差、参数化 | ⭐⭐ |
| [03-rasterization.md](docs/03-rasterization.md) | 投影、变换、深度排序 | ⭐⭐⭐ |
| [04-implementation.md](docs/04-implementation.md) | 算法步骤、数据结构、优化 | ⭐⭐⭐⭐ |
| [05-references.md](docs/05-references.md) | 论文、开源项目、学习资源 | ⭐ |

## 🔗 相关资源

### 官方资源
- 📄 [原始论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- 💻 [官方实现](https://github.com/graphdeco-inria/gaussian-splatting)
- 🎥 [项目主页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

### 社区资源
- [Awesome 3D Gaussian Splatting](https://github.com/MrNeRF/awesome-3D-gaussian-splatting)
- [3DGS 论文解读](https://github.com/graphdeco-inria/gaussian-splatting/issues)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

如果你发现文档错误、代码 bug，或者有改进建议：
1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📝 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 INRIA 和 Max Planck Institute 团队的开创性工作
- 感谢所有为 3DGS 技术发展做出贡献的研究者
- 感谢开源社区的支持

## 📧 联系方式

如有问题或建议，欢迎通过 Issue 与我们交流。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个星标支持！**

Made with ❤️ for 3DGS learners

</div>
