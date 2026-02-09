# 光栅化原理

本章将介绍光栅化的基本原理，以及 3D Gaussian Splatting 中的特殊光栅化技术。

## 什么是光栅化？

**光栅化（Rasterization）** 是将矢量图形（如点、线、三角形）转换为光栅图像（像素网格）的过程。

### 传统图形管线

```
3D 场景 → 顶点变换 → 投影 → 光栅化 → 片段着色 → 帧缓冲
```

### 3DGS 的光栅化管线

```
3D 高斯 → 投影到 2D → 排序 → Splatting → Alpha 混合 → 图像
```

## 坐标系统和变换

### 坐标系统

在 3DGS 中涉及多个坐标系：

1. **世界坐标系（World Space）**
   - 全局 3D 坐标系
   - 高斯的位置在世界坐标系中定义

2. **相机坐标系（Camera/View Space）**
   - 相机为原点
   - Z 轴指向相机前方

3. **屏幕坐标系（Screen/Image Space）**
   - 2D 像素坐标
   - 范围 [0, width] × [0, height]

### 投影变换

#### 世界 → 相机

使用视图矩阵（View Matrix）$V$：

$$
\mathbf{p}_{\text{cam}} = V \mathbf{p}_{\text{world}}
$$

其中 $V$ 由相机的位置和朝向定义：

$$
V = \begin{bmatrix}
R & -R\mathbf{t} \\
0 & 1
\end{bmatrix}
$$

- $R$：相机旋转（3×3）
- $\mathbf{t}$：相机位置

#### 相机 → 屏幕（透视投影）

使用投影矩阵（Projection Matrix）$P$：

$$
\begin{bmatrix}
x' \\ y' \\ z' \\ w'
\end{bmatrix} = P \begin{bmatrix}
x_{\text{cam}} \\ y_{\text{cam}} \\ z_{\text{cam}} \\ 1
\end{bmatrix}
$$

透视投影后进行齐次除法：

$$
x_{\text{screen}} = \frac{x'}{w'}, \quad y_{\text{screen}} = \frac{y'}{w'}
$$

#### 简化的针孔相机模型

对于简单的针孔相机：

$$
\begin{bmatrix}
u \\ v
\end{bmatrix} = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y
\end{bmatrix} \begin{bmatrix}
x_{\text{cam}} / z_{\text{cam}} \\
y_{\text{cam}} / z_{\text{cam}} \\
1
\end{bmatrix}
$$

其中：
- $(f_x, f_y)$：焦距（像素单位）
- $(c_x, c_y)$：主点（图像中心）
- $(u, v)$：屏幕像素坐标

## 3D 高斯的投影

### 核心思想

将 3D 高斯投影到 2D，得到 2D 高斯：

$$
G_{3D}(x, y, z) \xrightarrow{\text{投影}} G_{2D}(u, v)
$$

### 投影公式

给定 3D 高斯：
- 均值：$\boldsymbol{\mu}_{3D} = [x, y, z]^T$
- 协方差：$\Sigma_{3D} \in \mathbb{R}^{3 \times 3}$

投影后的 2D 高斯：
- 均值：$\boldsymbol{\mu}_{2D} = \text{Project}(\boldsymbol{\mu}_{3D})$
- 协方差：$\Sigma_{2D} = J \Sigma_{3D} J^T$

其中 $J$ 是投影的雅可比矩阵（Jacobian）。

### 雅可比矩阵

对于针孔相机模型，雅可比矩阵：

$$
J = \frac{\partial (u, v)}{\partial (x, y, z)} = \begin{bmatrix}
f_x / z & 0 & -f_x \cdot x / z^2 \\
0 & f_y / z & -f_y \cdot y / z^2
\end{bmatrix}
$$

这是一个 2×3 矩阵，将 3D 空间的微小变化映射到 2D 屏幕空间。

### 2D 协方差计算

$$
\Sigma_{2D} = J \Sigma_{3D} J^T
$$

这是一个 2×2 矩阵，定义了投影后 2D 高斯的形状。

**物理意义：**
- 3D 椭球体投影到 2D 平面成为椭圆
- 协方差矩阵变换确保形状正确

## Gaussian Splatting

### 什么是 Splatting？

**Splatting** 是一种渲染技术，将 3D 点"泼溅"到 2D 图像上：

```
每个高斯贡献 = 在屏幕上绘制一个 2D 高斯核
```

### 2D 高斯核

投影后的 2D 高斯函数：

$$
G_{2D}(u, v) = \exp\left(-\frac{1}{2}(\mathbf{p}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{p}-\boldsymbol{\mu})\right)
$$

其中：
- $\mathbf{p} = [u, v]^T$：屏幕像素坐标
- $\boldsymbol{\mu}$：高斯中心在屏幕上的位置
- $\Sigma$：2D 协方差矩阵

### Splatting 范围

高斯函数理论上在整个平面都有值，但实践中：
- 在距离中心 $3\sigma$ 外，值接近 0（<1%）
- 只需要在有限的像素范围内计算

**边界框（Bounding Box）：**

$$
u_{\min}, u_{\max} = \mu_u \pm 3\sqrt{\Sigma_{00}}
$$
$$
v_{\min}, v_{\max} = \mu_v \pm 3\sqrt{\Sigma_{11}}
$$

## 深度排序

### 为什么需要排序？

多个高斯可能投影到同一像素，需要正确的混合顺序。

### 画家算法（Painter's Algorithm）

```
从后往前绘制（远 → 近）
```

**原理：**
- 后面的物体先画
- 前面的物体后画，覆盖后面的

**在 3DGS 中：**
- 按照深度（相机坐标系中的 z 值）排序
- 深度大的（远）先渲染
- 深度小的（近）后渲染

### 排序准则

使用高斯中心的深度：

$$
z = \boldsymbol{\mu}_{\text{cam}} \cdot [0, 0, 1]^T
$$

排序：

$$
z_1 > z_2 > z_3 > \cdots
$$

## Alpha 混合

### 基本概念

**Alpha 混合**：根据透明度组合多个颜色。

每个高斯有：
- 颜色：$\mathbf{c}_i = [r, g, b]$
- 不透明度：$\alpha_i \in [0, 1]$

### 前向混合公式

从后往前混合：

$$
C = C_{\text{bg}} \cdot T_N + \sum_{i=1}^{N} \mathbf{c}_i \alpha_i G_i \cdot T_{i-1}
$$

其中：
- $C_{\text{bg}}$：背景颜色
- $G_i$：第 $i$ 个高斯在该像素的值（归一化后）
- $T_i$：透射率（transmittance）

$$
T_i = \prod_{j=1}^{i} (1 - \alpha_j G_j)
$$

**物理意义：**
- $\alpha_i G_i$：第 $i$ 个高斯的有效不透明度
- $T_{i-1}$：到达第 $i$ 个高斯前，光线剩余的强度

### 简化的逐像素混合

对于单个像素，按深度顺序混合：

```python
color = background_color
alpha_accumulated = 0

for gaussian in sorted_gaussians:  # 从远到近
    # 计算高斯在该像素的权重
    weight = gaussian.opacity * gaussian.evaluate(pixel)
    
    # 混合颜色
    color = color * (1 - weight) + gaussian.color * weight
    
    # 累积 alpha
    alpha_accumulated += weight
    
    # 提前终止（完全不透明）
    if alpha_accumulated > 0.99:
        break
```

### 可微分性

整个混合过程是可微分的：
- 可以对颜色、不透明度、位置、协方差求梯度
- 支持基于梯度的优化（训练）

## 渲染算法流程

### 完整的渲染流程

```
输入：
  - N 个 3D 高斯 {G₁, G₂, ..., Gₙ}
  - 相机参数（位置、朝向、内参）
  
步骤 1：视锥剔除（Frustum Culling）
  - 移除在相机视野外的高斯
  
步骤 2：变换到相机坐标系
  - 对每个高斯：μ_cam = V * μ_world
  
步骤 3：投影到 2D
  - 对每个高斯：
    - 计算 2D 中心：μ_2D = Project(μ_cam)
    - 计算 2D 协方差：Σ_2D = J * Σ_3D * Jᵀ
  
步骤 4：深度排序
  - 按 z_cam 降序排序
  
步骤 5：光栅化（Splatting）
  - 对每个像素 (u, v)：
    - 初始化：color = bg_color, T = 1.0
    - 对每个高斯（按深度顺序）：
      - 计算高斯值：G = exp(-0.5 * r² / σ²)
      - 计算贡献：contrib = T * α * G * c
      - 累加颜色：color += contrib
      - 更新透射率：T *= (1 - α * G)
      - 如果 T < 0.01：提前终止
  
输出：
  - 渲染图像（W × H × 3）
```

### 优化技巧

#### 1. Tile-Based 渲染

将屏幕分成小块（tiles）：
- 每个 tile 单独处理
- 减少内存访问
- 便于并行化

#### 2. 视锥剔除

只处理视野内的高斯：
```python
if not in_frustum(gaussian):
    continue  # 跳过
```

#### 3. 提前终止

当透射率很小时停止：
```python
if transmittance < 0.01:
    break  # 后面的高斯贡献很小
```

#### 4. 自适应半径

根据高斯大小动态调整 splatting 范围：
```python
radius = min(3 * sigma, max_radius)
```

## 传统光栅化 vs 高斯光栅化

### 三角形光栅化（传统）

```
特点：
- 渲染离散的三角面片
- 硬边界
- 需要纹理映射
- 需要复杂的几何
```

### 高斯光栅化（3DGS）

```
特点：
- 渲染连续的高斯点
- 软边界（自然混合）
- 颜色直接存储在高斯上
- 简单的点表示
```

### 对比

| 特性 | 传统光栅化 | 高斯光栅化 |
|------|------------|------------|
| 基本单元 | 三角形 | 高斯点 |
| 边界 | 硬边界 | 软边界 |
| 混合 | Z-buffer | Alpha 混合 |
| 纹理 | 需要 UV 映射 | 直接存储颜色 |
| 几何复杂度 | 高（需要网格） | 低（点集） |

## 实践建议

### 调试技巧

1. **可视化深度**：
   ```python
   # 将深度映射到颜色
   depth_color = (depth - min_depth) / (max_depth - min_depth)
   ```

2. **可视化单个高斯**：
   ```python
   # 只渲染第 i 个高斯
   render_single_gaussian(i)
   ```

3. **可视化 Alpha**：
   ```python
   # 显示每个像素的总 alpha
   alpha_map = sum(alpha * gaussian_value)
   ```

### 常见问题

**问题 1：高斯太大，覆盖整个屏幕**
- 解决：限制协方差矩阵的特征值
- 代码：`eigenvalues = clip(eigenvalues, min_val, max_val)`

**问题 2：颜色异常（过亮或过暗）**
- 解决：检查颜色范围 [0, 1]
- 代码：`color = np.clip(color, 0, 1)`

**问题 3：渲染太慢**
- 解决：实现视锥剔除和提前终止
- 解决：使用 tile-based 渲染

## 总结

### 关键要点

1. **投影**：3D 高斯 → 2D 高斯，通过雅可比矩阵变换协方差

2. **Splatting**：在屏幕上绘制 2D 高斯核

3. **排序**：按深度从后往前

4. **混合**：使用 Alpha 混合公式，累积颜色和透射率

5. **可微分**：整个流程可微，支持优化

### 下一步

1. 📖 阅读 [实现细节](04-implementation.md)，学习工程实现
2. 💻 运行 `code/rasterization/gaussian_rasterizer.py`
3. 🔍 尝试修改参数，观察渲染效果

---

**导航：**
- [← 上一章：高斯函数基础](02-gaussian-basics.md)
- [→ 下一章：实现细节](04-implementation.md)
