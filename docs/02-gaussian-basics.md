# 高斯函数基础

本章将深入讲解高斯函数的数学基础，这是理解 3D Gaussian Splatting 的关键。

## 1D 高斯分布

### 基本公式

一维高斯函数（也称正态分布）的概率密度函数：

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

其中：
- $\mu$ (mu)：均值，控制高斯的中心位置
- $\sigma$ (sigma)：标准差，控制高斯的宽度
- $\sigma^2$：方差

### 参数的影响

```python
# μ (均值) 的影响
μ = 0: 高斯中心在原点
μ = 5: 高斯中心向右移动到 x=5

# σ (标准差) 的影响
σ = 1: 标准宽度
σ = 0.5: 更窄更高的高斯
σ = 2: 更宽更矮的高斯
```

### 重要性质

1. **归一化**：整个函数下的面积为 1
   $$\int_{-\infty}^{\infty} f(x)dx = 1$$

2. **对称性**：关于 $\mu$ 对称

3. **68-95-99.7 规则**：
   - 68% 的数据在 $[\mu-\sigma, \mu+\sigma]$
   - 95% 的数据在 $[\mu-2\sigma, \mu+2\sigma]$
   - 99.7% 的数据在 $[\mu-3\sigma, \mu+3\sigma]$

## 2D 高斯分布

### 基本公式

二维高斯函数：

$$
f(x, y) = \frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_x)^2}{\sigma_x^2} - \frac{2\rho(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y} + \frac{(y-\mu_y)^2}{\sigma_y^2}\right]\right)
$$

### 使用协方差矩阵表示

更简洁的矩阵形式：

$$
f(\mathbf{x}) = \frac{1}{2\pi|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

其中：
- $\mathbf{x} = [x, y]^T$：2D 坐标向量
- $\boldsymbol{\mu} = [\mu_x, \mu_y]^T$：均值向量（中心位置）
- $\Sigma$：协方差矩阵

### 协方差矩阵

协方差矩阵 $\Sigma$ 定义了高斯的形状和方向：

$$
\Sigma = \begin{bmatrix}
\sigma_x^2 & \rho\sigma_x\sigma_y \\
\rho\sigma_x\sigma_y & \sigma_y^2
\end{bmatrix}
$$

其中：
- $\sigma_x^2$：x 方向的方差
- $\sigma_y^2$：y 方向的方差
- $\rho$：相关系数（范围 [-1, 1]）

### 协方差矩阵的作用

1. **对角协方差**（$\rho = 0$）：
   ```
   Σ = [[σ_x², 0    ]
        [0,    σ_y²]]
   ```
   - 高斯轴对齐（沿 x 和 y 轴）
   - 无旋转

2. **非对角协方差**（$\rho \neq 0$）：
   ```
   Σ = [[σ_x², ρσ_xσ_y]
        [ρσ_xσ_y, σ_y²]]
   ```
   - 高斯有旋转
   - 椭圆形分布

### 特征值分解

协方差矩阵可以分解为：

$$
\Sigma = R \Lambda R^T
$$

其中：
- $R$：旋转矩阵（特征向量）
- $\Lambda = \text{diag}(\lambda_1, \lambda_2)$：对角矩阵（特征值）

**物理意义：**
- 特征值 $\lambda_1, \lambda_2$：椭圆的两个主轴长度的平方
- 特征向量：椭圆的主轴方向

## 3D 高斯分布

### 基本公式

三维高斯函数：

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{3/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

其中：
- $\mathbf{x} = [x, y, z]^T$：3D 坐标向量
- $\boldsymbol{\mu} = [\mu_x, \mu_y, \mu_z]^T$：均值向量（3D 位置）
- $\Sigma \in \mathbb{R}^{3 \times 3}$：3×3 协方差矩阵

### 3D 协方差矩阵

$$
\Sigma = \begin{bmatrix}
\sigma_x^2 & \sigma_{xy} & \sigma_{xz} \\
\sigma_{xy} & \sigma_y^2 & \sigma_{yz} \\
\sigma_{xz} & \sigma_{yz} & \sigma_z^2
\end{bmatrix}
$$

**性质：**
- 对称矩阵：$\Sigma = \Sigma^T$
- 半正定：所有特征值 $\geq 0$
- 定义了 3D 椭球体

### 几何解释

3D 高斯定义了一个椭球体：
- **中心**：$\boldsymbol{\mu}$ (位置)
- **形状和方向**：$\Sigma$ (协方差)

椭球体的主轴：
- 通过协方差矩阵的特征值分解得到
- 三个主轴长度：$2\sqrt{\lambda_1}, 2\sqrt{\lambda_2}, 2\sqrt{\lambda_3}$
- 三个主轴方向：对应的特征向量

## 协方差矩阵的参数化

在 3DGS 中，我们需要一种**可优化且稳定**的方式来表示协方差矩阵。

### 问题

直接优化 $\Sigma$ 的 6 个参数存在问题：
- ❌ 难以保证对称性
- ❌ 难以保证半正定性（可能出现负特征值）
- ❌ 数值不稳定

### 解决方案：旋转+缩放分解

将协方差矩阵分解为：

$$
\Sigma = R S S^T R^T
$$

其中：
- $R \in SO(3)$：3×3 旋转矩阵
- $S = \text{diag}(s_x, s_y, s_z)$：缩放矩阵

**参数化方法：**

1. **旋转 R**：使用四元数 $\mathbf{q} = (q_w, q_x, q_y, q_z)$
   - 4 个参数
   - 自动满足旋转约束（归一化后）

2. **缩放 S**：使用 3 个缩放因子 $\mathbf{s} = (s_x, s_y, s_z)$
   - 3 个参数
   - 使用 $s_i = \exp(\tilde{s}_i)$ 保证正值

**优势：**
- ✅ 总共 7 个参数（4 个四元数 + 3 个缩放）
- ✅ 自动保证 $\Sigma$ 对称且半正定
- ✅ 数值稳定
- ✅ 几何意义清晰

### 四元数表示旋转

四元数 $\mathbf{q} = (w, x, y, z)$ 转换为旋转矩阵：

$$
R = \begin{bmatrix}
1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\
2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\
2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2)
\end{bmatrix}
$$

其中四元数需要归一化：$w^2 + x^2 + y^2 + z^2 = 1$

### 完整的高斯参数

一个 3D 高斯需要以下参数：

```python
class Gaussian3D:
    position: [x, y, z]           # 3 参数 - 位置
    rotation: [qw, qx, qy, qz]    # 4 参数 - 旋转（四元数）
    scale: [sx, sy, sz]           # 3 参数 - 缩放
    color: [r, g, b] or SH        # 3+ 参数 - 颜色
    opacity: α                     # 1 参数 - 不透明度
```

**总计：** 至少 14 个参数/高斯（使用 RGB 颜色）

## 球谐函数（进阶）

为了表示视角相关的外观，3DGS 使用球谐函数（Spherical Harmonics, SH）：

### 0 阶球谐（常数）

最简单的情况，等价于 RGB 颜色：
- 1 个球谐基函数
- 3 个通道 (RGB)
- 总共 3 个参数

### 高阶球谐

使用更高阶的球谐函数可以表示更复杂的视角相关效果：
- 1 阶：4 个基函数 × 3 通道 = 12 参数（总共 15 参数）
- 2 阶：9 个基函数 × 3 通道 = 27 参数（总共 30 参数）
- 3 阶：16 个基函数 × 3 通道 = 48 参数（总共 51 参数）

球谐系数 $\mathbf{c}$ 用于计算特定方向 $\mathbf{d}$ 的颜色：

$$
\text{color}(\mathbf{d}) = \sum_{l,m} c_{l,m} Y_l^m(\mathbf{d})
$$

## 实践示例

### 示例 1：创建一个简单的 2D 高斯

```python
import numpy as np

# 参数
mu = np.array([0, 0])        # 中心在原点
sigma = np.array([[1, 0],     # 协方差矩阵（圆形）
                  [0, 1]])

# 评估函数
def gaussian_2d(x, y):
    pos = np.array([x, y])
    diff = pos - mu
    exponent = -0.5 * diff.T @ np.linalg.inv(sigma) @ diff
    coeff = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    return coeff * np.exp(exponent)
```

### 示例 2：旋转的椭圆高斯

```python
import numpy as np

# 旋转 45 度
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 缩放
S = np.diag([2, 0.5])  # x 方向长，y 方向短

# 协方差矩阵
Sigma = R @ S @ S.T @ R.T
```

## 关键要点总结

1. **高斯函数是连续的平滑函数**，适合表示空间分布

2. **协方差矩阵** $\Sigma$ 完全定义了高斯的形状和方向

3. **参数化很重要**：使用四元数+缩放保证数值稳定性

4. **3D 高斯 → 2D 投影**：这是下一章的重点

5. **球谐函数**：用于视角相关的外观建模

## 练习建议

1. 运行 `code/basic/gaussian_2d.py`：
   - 观察不同 $\sigma$ 的效果
   - 尝试不同的协方差矩阵
   - 可视化旋转的椭圆高斯

2. 运行 `code/basic/gaussian_3d.py`：
   - 理解 3D 高斯的形状
   - 观察旋转和缩放的效果
   - 尝试投影到 2D

3. 手动计算：
   - 给定协方差矩阵，计算特征值和特征向量
   - 给定四元数，转换为旋转矩阵

## 下一步

现在你已经掌握了高斯函数的数学基础，接下来：

1. 📖 阅读 [光栅化原理](03-rasterization.md)，学习如何渲染高斯
2. 💻 运行 2D 和 3D 高斯可视化代码
3. 🧮 尝试修改参数，加深理解

---

**导航：**
- [← 上一章：3DGS 简介](01-introduction.md)
- [→ 下一章：光栅化原理](03-rasterization.md)
