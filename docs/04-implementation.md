# 实现细节

本章深入讲解 3D Gaussian Splatting 的工程实现，包括数据结构、算法细节和优化技巧。

## 核心数据结构

### 3D 高斯表示

```python
class Gaussian3D:
    """表示一个 3D 高斯"""
    
    def __init__(self):
        # 几何属性
        self.position = np.zeros(3)        # [x, y, z] 位置
        self.rotation = np.array([1,0,0,0]) # [w, x, y, z] 四元数
        self.scale = np.ones(3)            # [sx, sy, sz] 缩放
        
        # 外观属性
        self.color = np.ones(3)            # [r, g, b] 颜色
        self.opacity = 1.0                 # α ∈ [0, 1] 不透明度
        
        # 可选：球谐系数（用于视角相关外观）
        self.sh_coeffs = None              # (K, 3) 球谐系数
        
    def get_covariance_matrix(self):
        """计算 3D 协方差矩阵 Σ = R S Sᵀ Rᵀ"""
        R = quaternion_to_rotation_matrix(self.rotation)
        S = np.diag(self.scale)
        return R @ S @ S.T @ R.T
```

### 场景表示

```python
class GaussianScene:
    """3D 高斯场景"""
    
    def __init__(self):
        self.gaussians = []  # List[Gaussian3D]
        
    def add_gaussian(self, position, color, scale, rotation=None, opacity=1.0):
        """添加一个高斯"""
        g = Gaussian3D()
        g.position = position
        g.color = color
        g.scale = scale
        g.rotation = rotation if rotation is not None else np.array([1,0,0,0])
        g.opacity = opacity
        self.gaussians.append(g)
        
    def get_positions(self):
        """获取所有高斯的位置（N×3）"""
        return np.array([g.position for g in self.gaussians])
    
    def get_colors(self):
        """获取所有高斯的颜色（N×3）"""
        return np.array([g.color for g in self.gaussians])
```

### 相机参数

```python
class Camera:
    """相机参数"""
    
    def __init__(self, width, height, fov=60):
        self.width = width
        self.height = height
        
        # 外参（相机在世界中的位置和朝向）
        self.position = np.array([0, 0, 5])  # 相机位置
        self.target = np.array([0, 0, 0])     # 看向的目标
        self.up = np.array([0, 1, 0])         # 上方向
        
        # 内参（投影参数）
        self.fov = fov  # 视场角（度）
        self.focal_length = self.width / (2 * np.tan(np.radians(fov) / 2))
        self.cx = width / 2   # 主点 x
        self.cy = height / 2  # 主点 y
        
    def get_view_matrix(self):
        """计算视图矩阵（世界 → 相机）"""
        # 计算相机坐标系的三个轴
        z_axis = normalize(self.position - self.target)  # 相机朝向的反方向
        x_axis = normalize(np.cross(self.up, z_axis))    # 右方向
        y_axis = np.cross(z_axis, x_axis)                # 上方向
        
        # 构建旋转矩阵
        R = np.array([x_axis, y_axis, z_axis])
        
        # 构建完整的视图矩阵（4×4）
        V = np.eye(4)
        V[:3, :3] = R
        V[:3, 3] = -R @ self.position
        return V
    
    def get_projection_matrix(self):
        """计算投影矩阵（针孔相机）"""
        K = np.array([
            [self.focal_length, 0, self.cx],
            [0, self.focal_length, self.cy],
            [0, 0, 1]
        ])
        return K
```

## 核心算法实现

### 1. 四元数到旋转矩阵

```python
def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    
    Args:
        q: (4,) 四元数 [w, x, y, z]
        
    Returns:
        R: (3, 3) 旋转矩阵
    """
    # 归一化
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    # 构建旋转矩阵
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R
```

### 2. 3D 高斯投影到 2D

```python
def project_gaussian_to_2d(gaussian, camera):
    """
    将 3D 高斯投影到 2D
    
    Args:
        gaussian: Gaussian3D 对象
        camera: Camera 对象
        
    Returns:
        mean_2d: (2,) 2D 中心位置
        cov_2d: (2, 2) 2D 协方差矩阵
        depth: float 深度值
    """
    # 1. 变换到相机坐标系
    V = camera.get_view_matrix()
    pos_homo = np.append(gaussian.position, 1)  # 齐次坐标
    pos_cam = (V @ pos_homo)[:3]
    
    depth = pos_cam[2]
    
    # 2. 投影到屏幕
    K = camera.get_projection_matrix()
    pos_2d_homo = K @ (pos_cam / depth)
    mean_2d = pos_2d_homo[:2]
    
    # 3. 计算 2D 协方差矩阵
    # 3D 协方差
    cov_3d = gaussian.get_covariance_matrix()
    
    # 变换到相机坐标系
    R = V[:3, :3]
    cov_3d_cam = R @ cov_3d @ R.T
    
    # 雅可比矩阵
    fx, fy = camera.focal_length, camera.focal_length
    z = pos_cam[2]
    x, y = pos_cam[0], pos_cam[1]
    
    J = np.array([
        [fx / z, 0, -fx * x / (z**2)],
        [0, fy / z, -fy * y / (z**2)]
    ])
    
    # 投影协方差
    cov_2d = J @ cov_3d_cam @ J.T
    
    # 确保对称性（数值稳定性）
    cov_2d = (cov_2d + cov_2d.T) / 2
    
    return mean_2d, cov_2d, depth
```

### 3. 评估 2D 高斯

```python
def evaluate_gaussian_2d(pixel, mean, cov_inv):
    """
    评估 2D 高斯在某个像素的值
    
    Args:
        pixel: (2,) 像素坐标 [u, v]
        mean: (2,) 高斯中心
        cov_inv: (2, 2) 协方差矩阵的逆
        
    Returns:
        value: float 高斯值（未归一化）
    """
    diff = pixel - mean
    exponent = -0.5 * diff @ cov_inv @ diff
    return np.exp(exponent)
```

### 4. 光栅化（单像素）

```python
def render_pixel(pixel_coord, gaussians_2d, sorted_indices):
    """
    渲染单个像素
    
    Args:
        pixel_coord: (2,) 像素坐标 [u, v]
        gaussians_2d: List of (mean, cov, color, opacity)
        sorted_indices: 按深度排序的索引（远到近）
        
    Returns:
        color: (3,) RGB 颜色
    """
    color = np.array([0.0, 0.0, 0.0])  # 背景色
    transmittance = 1.0
    
    for idx in sorted_indices:
        mean, cov, gauss_color, opacity = gaussians_2d[idx]
        
        # 计算高斯值
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue  # 奇异矩阵，跳过
        
        gaussian_value = evaluate_gaussian_2d(pixel_coord, mean, cov_inv)
        
        # 有效 alpha
        alpha = opacity * gaussian_value
        alpha = np.clip(alpha, 0, 1)
        
        # 累积颜色
        weight = alpha * transmittance
        color += weight * gauss_color
        
        # 更新透射率
        transmittance *= (1 - alpha)
        
        # 提前终止
        if transmittance < 0.01:
            break
    
    return np.clip(color, 0, 1)
```

### 5. 完整渲染

```python
def render(scene, camera):
    """
    渲染完整场景
    
    Args:
        scene: GaussianScene 对象
        camera: Camera 对象
        
    Returns:
        image: (H, W, 3) 渲染图像
    """
    H, W = camera.height, camera.width
    image = np.zeros((H, W, 3))
    
    # 1. 投影所有高斯到 2D
    gaussians_2d = []
    depths = []
    
    for gaussian in scene.gaussians:
        mean_2d, cov_2d, depth = project_gaussian_to_2d(gaussian, camera)
        
        # 视锥剔除
        if depth <= 0:  # 在相机后面
            continue
        if mean_2d[0] < -100 or mean_2d[0] > W + 100:  # 太远的边界
            continue
        if mean_2d[1] < -100 or mean_2d[1] > H + 100:
            continue
        
        gaussians_2d.append((mean_2d, cov_2d, gaussian.color, gaussian.opacity))
        depths.append(depth)
    
    # 2. 深度排序
    sorted_indices = np.argsort(depths)[::-1]  # 从远到近
    
    # 3. 光栅化每个像素
    for v in range(H):
        for u in range(W):
            pixel_coord = np.array([u, v])
            color = render_pixel(pixel_coord, gaussians_2d, sorted_indices)
            image[v, u] = color
    
    return image
```

## 性能优化技巧

### 1. Tile-Based 渲染

```python
def render_with_tiles(scene, camera, tile_size=16):
    """使用分块渲染优化"""
    H, W = camera.height, camera.width
    image = np.zeros((H, W, 3))
    
    # 分块
    for tile_y in range(0, H, tile_size):
        for tile_x in range(0, W, tile_size):
            # 获取 tile 边界
            y_start, y_end = tile_y, min(tile_y + tile_size, H)
            x_start, x_end = tile_x, min(tile_x + tile_size, W)
            
            # 找出影响这个 tile 的高斯
            tile_gaussians = filter_gaussians_for_tile(
                gaussians_2d, x_start, x_end, y_start, y_end
            )
            
            # 渲染 tile
            for v in range(y_start, y_end):
                for u in range(x_start, x_end):
                    color = render_pixel([u, v], tile_gaussians, sorted_indices)
                    image[v, u] = color
    
    return image
```

### 2. 边界框剔除

```python
def get_gaussian_bounding_box(mean, cov, threshold=3.0):
    """
    计算高斯的边界框
    
    Args:
        mean: (2,) 中心
        cov: (2, 2) 协方差
        threshold: 截断阈值（几个标准差）
        
    Returns:
        bbox: (x_min, y_min, x_max, y_max)
    """
    # 特征值分解
    eigenvalues = np.linalg.eigvalsh(cov)
    
    # 最大半径
    max_radius = threshold * np.sqrt(eigenvalues.max())
    
    x_min = int(mean[0] - max_radius)
    y_min = int(mean[1] - max_radius)
    x_max = int(mean[0] + max_radius)
    y_max = int(mean[1] + max_radius)
    
    return (x_min, y_min, x_max, y_max)
```

### 3. 球谐函数评估

```python
def evaluate_spherical_harmonics(sh_coeffs, view_direction):
    """
    评估球谐函数以获得视角相关颜色
    
    Args:
        sh_coeffs: (K, 3) 球谐系数
        view_direction: (3,) 归一化的视线方向
        
    Returns:
        color: (3,) RGB 颜色
    """
    # 0 阶（常数）
    color = sh_coeffs[0]
    
    if len(sh_coeffs) > 1:
        # 1 阶
        x, y, z = view_direction
        Y_1_neg1 = y
        Y_1_0 = z
        Y_1_pos1 = x
        
        color += sh_coeffs[1] * Y_1_neg1
        color += sh_coeffs[2] * Y_1_0
        color += sh_coeffs[3] * Y_1_pos1
    
    # 更高阶...（省略）
    
    return np.clip(color, 0, 1)
```

## CUDA 实现要点（概述）

虽然本项目使用 Python 实现，但了解 CUDA 优化思路很重要：

### 核心思想

1. **并行化**：
   - 每个像素一个线程
   - 或每个 tile 一个 thread block

2. **共享内存**：
   - Tile 内的高斯数据存储在共享内存
   - 减少全局内存访问

3. **原子操作**：
   - 用于累积颜色（如果需要）

### 伪代码

```cuda
__global__ void render_kernel(
    const Gaussian* gaussians,  // 高斯数据
    const int* sorted_indices,  // 排序索引
    float* output_image,         // 输出图像
    int width, int height
) {
    // 每个线程处理一个像素
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (u >= width || v >= height) return;
    
    float3 color = make_float3(0, 0, 0);
    float T = 1.0f;
    
    // 遍历排序后的高斯
    for (int i = 0; i < num_gaussians; i++) {
        int idx = sorted_indices[i];
        Gaussian g = gaussians[idx];
        
        // 计算高斯贡献
        float alpha = compute_gaussian_alpha(g, u, v);
        
        // 累积颜色
        color += T * alpha * g.color;
        T *= (1 - alpha);
        
        // 提前终止
        if (T < 0.01f) break;
    }
    
    // 写入输出
    int pixel_idx = v * width + u;
    output_image[pixel_idx * 3 + 0] = color.x;
    output_image[pixel_idx * 3 + 1] = color.y;
    output_image[pixel_idx * 3 + 2] = color.z;
}
```

## 数值稳定性

### 问题和解决方案

#### 1. 协方差矩阵奇异

**问题**：协方差矩阵可能接近奇异（不可逆）

**解决**：添加小的正则化项
```python
cov_2d += np.eye(2) * 1e-6
```

#### 2. 指数溢出

**问题**：$e^x$ 可能溢出

**解决**：限制指数范围
```python
exponent = np.clip(exponent, -50, 50)
```

#### 3. 颜色范围

**问题**：颜色可能超出 [0, 1]

**解决**：裁剪
```python
color = np.clip(color, 0, 1)
```

## 调试和可视化

### 调试技巧

```python
def debug_render(scene, camera):
    """调试渲染流程"""
    
    # 1. 可视化高斯位置
    positions = scene.get_positions()
    plt.scatter(positions[:, 0], positions[:, 2])
    plt.title("Top-down view of Gaussians")
    plt.show()
    
    # 2. 可视化深度
    depths = compute_depths(scene, camera)
    depth_image = depths_to_image(depths)
    plt.imshow(depth_image, cmap='gray')
    plt.title("Depth Map")
    plt.show()
    
    # 3. 渲染单个高斯
    for i, gaussian in enumerate(scene.gaussians):
        single_gaussian_scene = GaussianScene()
        single_gaussian_scene.gaussians = [gaussian]
        image = render(single_gaussian_scene, camera)
        plt.imshow(image)
        plt.title(f"Gaussian {i}")
        plt.show()
```

### 性能分析

```python
import time

def profile_render(scene, camera):
    """性能分析"""
    
    times = {}
    
    # 投影
    start = time.time()
    gaussians_2d = project_all_gaussians(scene, camera)
    times['projection'] = time.time() - start
    
    # 排序
    start = time.time()
    sorted_indices = sort_by_depth(gaussians_2d)
    times['sorting'] = time.time() - start
    
    # 光栅化
    start = time.time()
    image = rasterize(gaussians_2d, sorted_indices, camera)
    times['rasterization'] = time.time() - start
    
    # 打印结果
    for stage, t in times.items():
        print(f"{stage}: {t:.3f}s")
    
    return image
```

## 总结

### 关键实现要点

1. **数据结构**：
   - 高斯用四元数+缩放表示协方差
   - 场景是高斯的集合
   - 相机包含内外参

2. **核心算法**：
   - 投影：3D 协方差 → 2D 协方差
   - 排序：按深度从后往前
   - 渲染：alpha 混合

3. **优化**：
   - 视锥剔除
   - Tile-based 渲染
   - 提前终止
   - 边界框

4. **数值稳定性**：
   - 正则化协方差矩阵
   - 裁剪指数和颜色值
   - 检查奇异情况

### 实践建议

1. 先实现简单版本，再优化
2. 每个阶段独立测试
3. 可视化中间结果
4. 使用小场景调试

## 下一步

1. 📖 阅读 [参考资料](05-references.md)，深入学习
2. 💻 运行完整示例 `examples/simple_scene.py`
3. 🔧 尝试实现自己的优化

---

**导航：**
- [← 上一章：光栅化原理](03-rasterization.md)
- [→ 下一章：参考资料](05-references.md)
