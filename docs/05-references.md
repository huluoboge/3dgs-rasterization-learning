# 参考资料和论文

本章汇总了学习 3D Gaussian Splatting 的重要参考资料、论文和资源。

## 原始论文

### 主要论文

**3D Gaussian Splatting for Real-Time Radiance Field Rendering**

- **作者**：Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- **机构**：Inria, Université Côte d'Azur, Max Planck Institute for Informatics
- **发表**：SIGGRAPH 2023
- **论文链接**：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **项目主页**：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- **arXiv**：https://arxiv.org/abs/2308.04079

**摘要要点**：
- 提出使用 3D 高斯作为辐射场的显式表示
- 实现实时渲染（1080p @ 100+ FPS）
- 质量与最先进的 NeRF 方法相当
- 完全可微分的光栅化管线

**核心贡献**：
1. 3D 高斯的各向异性表示
2. 快速可微分光栅化算法
3. 自适应密度控制的优化策略
4. tile-based 渲染架构

## 相关论文

### NeRF 和辐射场

#### 1. NeRF: Representing Scenes as Neural Radiance Fields
- **作者**：Ben Mildenhall et al.
- **发表**：ECCV 2020
- **链接**：https://www.matthewtancik.com/nerf
- **重要性**：神经辐射场的开创性工作

#### 2. Instant Neural Graphics Primitives
- **作者**：Thomas Müller et al.
- **发表**：SIGGRAPH 2022
- **链接**：https://nvlabs.github.io/instant-ngp/
- **重要性**：使用哈希编码加速 NeRF 训练

#### 3. Mip-NeRF: A Multiscale Representation
- **作者**：Jonathan T. Barron et al.
- **发表**：ICCV 2021
- **重要性**：改进 NeRF 的抗锯齿

### 点云渲染

#### 4. Differential Point Rendering
- **作者**：Jan Eric Lenssen et al.
- **发表**：ECCV 2020
- **重要性**：可微分点云渲染

#### 5. Neural Point-Based Graphics
- **作者**：Kara-Ali Aliev et al.
- **发表**：ECCV 2020
- **重要性**：使用点云表示的神经渲染

### Splatting 技术

#### 6. Surface Splatting
- **作者**：Hanspeter Pfister et al.
- **发表**：SIGGRAPH 2001
- **重要性**：经典的 splatting 技术

#### 7. EWA Volume Splatting
- **作者**：Matthias Zwicker et al.
- **发表**：VIS 2001
- **重要性**：椭圆加权平均 splatting

## 3DGS 的扩展和改进

### 质量改进

#### 1. Mip-Splatting
- **论文**：Mip-Splatting: Alias-free 3D Gaussian Splatting
- **作者**：Zehao Yu et al.
- **发表**：2023
- **arXiv**：https://arxiv.org/abs/2311.16493
- **改进**：更好的抗锯齿，减少渲染伪影

#### 2. Scaffold-GS
- **论文**：Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering
- **作者**：Tao Lu et al.
- **发表**：2024
- **改进**：结构化场景表示，提高效率

#### 3. 2D Gaussian Splatting
- **论文**：2D Gaussian Splatting for Geometrically Accurate Radiance Fields
- **作者**：Binbin Huang et al.
- **发表**：2024
- **arXiv**：https://arxiv.org/abs/2403.17888
- **改进**：使用 2D 高斯，更准确的几何表示

### 动态场景

#### 4. 4D Gaussian Splatting
- **论文**：4D Gaussian Splatting for Real-Time Dynamic Scene Rendering
- **作者**：Guanjun Wu et al.
- **发表**：2023
- **arXiv**：https://arxiv.org/abs/2310.08528
- **扩展**：支持动态场景和时间维度

#### 5. Deformable 3D Gaussians
- **论文**：Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction
- **作者**：Ziyi Yang et al.
- **发表**：2023
- **扩展**：可变形高斯用于动态重建

### 编辑和控制

#### 6. GaussianEditor
- **论文**：GaussianEditor: Swift and Controllable 3D Editing
- **发表**：2023
- **功能**：交互式 3D 场景编辑

#### 7. GauHuman
- **论文**：GauHuman: Articulated Gaussian Splatting from Monocular Human Videos
- **发表**：2023
- **应用**：人体建模和动画

### 压缩和效率

#### 8. Compact 3D Gaussians
- **论文**：Compact 3D Gaussian Representation
- **发表**：2024
- **改进**：压缩高斯表示，减少存储

#### 9. LightGaussian
- **论文**：LightGaussian: Unbounded 3D Gaussian Compression
- **发表**：2024
- **改进**：高效压缩算法

## 开源实现

### 官方实现

**gaussian-splatting**
- **GitHub**：https://github.com/graphdeco-inria/gaussian-splatting
- **语言**：Python + CUDA
- **许可**：Inria Software License
- **特点**：
  - 官方参考实现
  - 包含训练和渲染代码
  - CUDA 优化
  - 完整的工具链

### 社区实现

#### 1. gsplat
- **GitHub**：https://github.com/nerfstudio-project/gsplat
- **语言**：Python + CUDA
- **特点**：
  - 作为库使用
  - 集成到 Nerfstudio
  - 模块化设计

#### 2. taichi-splatting
- **GitHub**：https://github.com/wanmeihuali/taichi_3d_gaussian_splatting
- **语言**：Python + Taichi
- **特点**：
  - 使用 Taichi 框架
  - 跨平台
  - 易于学习

#### 3. diff-gaussian-rasterization
- **GitHub**：https://github.com/graphdeco-inria/diff-gaussian-rasterization
- **语言**：C++ + CUDA
- **特点**：
  - 可微分光栅化核心
  - PyTorch 绑定
  - 高性能

#### 4. Unity Implementation
- **GitHub**：https://github.com/aras-p/UnityGaussianSplatting
- **语言**：C#
- **特点**：
  - Unity 引擎集成
  - 实时查看器
  - 交互式

#### 5. WebGL Viewer
- **GitHub**：https://github.com/antimatter15/splat
- **语言**：JavaScript
- **特点**：
  - 浏览器内运行
  - 无需安装
  - 快速预览

## 学习资源

### 视频教程

#### 1. 官方视频
- **SIGGRAPH 2023 Talk**：[YouTube](https://www.youtube.com/watch?v=T_kXY43VZnk)
- **项目演示**：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

#### 2. 社区教程
- **Two Minute Papers**：3D Gaussian Splatting 解读
- **Károly Zsolnai-Fehér**：技术分析视频

### 博客文章

#### 1. 技术深度解析
- **Understanding 3D Gaussian Splatting**
  - 链接：https://huggingface.co/blog/gaussian-splatting
  - 作者：Hugging Face Team
  - 内容：详细的技术讲解

#### 2. 实现笔记
- **Implementing 3DGS from Scratch**
  - 各种开发者的实现笔记
  - 常见问题和解决方案

### 在线资源

#### 1. Awesome 3D Gaussian Splatting
- **GitHub**：https://github.com/MrNeRF/awesome-3D-gaussian-splatting
- **内容**：
  - 论文列表（100+）
  - 实现代码
  - 应用案例
  - 持续更新

#### 2. Papers with Code
- **链接**：https://paperswithcode.com/method/3d-gaussian-splatting
- **内容**：
  - 论文 + 代码
  - 性能对比
  - 数据集

### 交互式演示

#### 1. 在线 Viewer
- **Luma AI**：https://lumalabs.ai/
- **Polycam**：https://poly.cam/
- 可以上传和查看 3DGS 场景

## 数学和理论背景

### 必备数学知识

#### 1. 线性代数
- **推荐书籍**：《Linear Algebra Done Right》by Sheldon Axler
- **在线课程**：MIT 18.06 Linear Algebra

#### 2. 概率统计
- **推荐书籍**：《Pattern Recognition and Machine Learning》by Christopher Bishop
- **重点**：多元高斯分布

#### 3. 计算机图形学
- **推荐书籍**：
  - 《Fundamentals of Computer Graphics》by Steve Marschner
  - 《Real-Time Rendering》by Tomas Akenine-Möller
- **在线课程**：
  - GAMES101：现代计算机图形学入门（中文）

### 相关技术

#### 1. 体积渲染（Volume Rendering）
- **经典论文**：Max, N. "Optical Models for Direct Volume Rendering"
- **关联**：理解透射率和 alpha 混合

#### 2. 球谐函数（Spherical Harmonics）
- **教程**：Spherical Harmonic Lighting by Robin Green
- **应用**：视角相关外观建模

## 数据集

### 常用数据集

#### 1. NeRF Synthetic Dataset
- **链接**：https://github.com/bmild/nerf
- **内容**：8 个合成场景
- **用途**：基准测试

#### 2. Mip-NeRF 360 Dataset
- **链接**：https://jonbarron.info/mipnerf360/
- **内容**：真实世界场景
- **特点**：360 度视角

#### 3. Tanks and Temples
- **链接**：https://www.tanksandtemples.org/
- **内容**：大规模真实场景
- **用途**：评估重建质量

#### 4. DTU Dataset
- **链接**：https://roboimagedata.compute.dtu.dk/
- **内容**：多视角立体数据
- **用途**：几何重建评估

## 工具和软件

### 数据采集

#### 1. COLMAP
- **链接**：https://colmap.github.io/
- **功能**：Structure from Motion，相机标定
- **用途**：从图片序列重建稀疏点云和相机参数

#### 2. Reality Capture
- **链接**：https://www.capturingreality.com/
- **功能**：专业摄影测量软件
- **特点**：商业软件，高精度

### 可视化

#### 1. MeshLab
- **链接**：https://www.meshlab.net/
- **功能**：点云和网格查看
- **免费**：开源

#### 2. CloudCompare
- **链接**：https://www.cloudcompare.org/
- **功能**：点云处理和可视化
- **免费**：开源

## 社区和讨论

### GitHub Discussions
- **官方仓库 Issues**：https://github.com/graphdeco-inria/gaussian-splatting/issues
- 技术问题讨论

### Discord/Slack
- **Nerfstudio Discord**：3DGS 相关讨论频道
- 实时交流

### Reddit
- **/r/computergraphics**：计算机图形学讨论
- **/r/MachineLearning**：ML 相关讨论

## 进阶主题

### 研究方向

1. **更快的训练**：减少优化时间
2. **更好的几何**：提取精确的表面
3. **大规模场景**：城市级别重建
4. **动态场景**：实时动态捕捉
5. **语义理解**：结合语义信息
6. **物理仿真**：可交互的物理场景

### 应用领域

1. **VR/AR**：虚拟现实内容创建
2. **自动驾驶**：场景理解和仿真
3. **电影制作**：特效和虚拟制片
4. **游戏开发**：真实场景捕捉
5. **文化遗产**：文物数字化保护
6. **房地产**：虚拟看房
7. **电子商务**：3D 产品展示

## 推荐学习路径

### 初学者路径（1-2 个月）

1. **Week 1-2：基础理论**
   - 阅读本仓库的文档
   - 理解高斯函数和光栅化
   - 运行代码示例

2. **Week 3-4：代码实践**
   - 研究简化实现
   - 尝试修改参数
   - 可视化结果

3. **Week 5-6：原始论文**
   - 精读 3DGS 原始论文
   - 理解完整算法
   - 查看官方实现

4. **Week 7-8：项目实践**
   - 采集自己的数据
   - 训练自己的场景
   - 分享结果

### 进阶路径（3-6 个月）

1. **深入实现**
   - 学习 CUDA 编程
   - 实现 GPU 加速
   - 优化性能

2. **扩展研究**
   - 阅读改进论文
   - 实现新特性
   - 尝试新应用

3. **贡献社区**
   - 提交 PR
   - 分享经验
   - 写教程

## 总结

### 核心资源清单

**必读：**
- ✅ 3DGS 原始论文
- ✅ 官方实现代码
- ✅ 本仓库文档

**推荐阅读：**
- ⭐ NeRF 相关论文
- ⭐ Mip-Splatting 等改进
- ⭐ Awesome 3DGS 列表

**实践工具：**
- 🔧 COLMAP（数据准备）
- 🔧 官方 3DGS 代码（训练）
- 🔧 在线 Viewer（查看结果）

### 持续学习

3DGS 是一个快速发展的领域：
- 每月都有新论文
- 社区非常活跃
- 应用不断扩展

**建议：**
- 关注 arXiv 新论文
- 参与社区讨论
- 尝试实际项目

---

## 联系和贡献

如果你发现好的资源或有建议：
- 提交 Issue
- 发起 Pull Request
- 分享你的学习心得

**祝学习顺利！**

---

**导航：**
- [← 上一章：实现细节](04-implementation.md)
- [→ 返回首页](../README.md)
