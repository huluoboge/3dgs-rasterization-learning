"""
高斯光栅化器
Gaussian Rasterizer

实现 3D Gaussian Splatting 的核心光栅化算法
"""

import numpy as np
import matplotlib.pyplot as plt


def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])
    return R


class Gaussian3D:
    """3D 高斯"""
    
    def __init__(self, position, color, scale, rotation=None, opacity=1.0):
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.rotation = rotation if rotation is not None else np.array([1, 0, 0, 0], dtype=float)
        self.opacity = opacity
        
    def get_covariance_3d(self):
        """计算 3D 协方差矩阵"""
        R = quaternion_to_rotation_matrix(self.rotation)
        S = np.diag(self.scale)
        return R @ S @ S.T @ R.T


class GaussianCamera:
    """相机"""
    
    def __init__(self, width, height, fov=60):
        self.width = width
        self.height = height
        self.fov = fov
        
        # 焦距
        self.focal_length = self.width / (2 * np.tan(np.radians(fov) / 2))
        
        # 主点
        self.cx = width / 2
        self.cy = height / 2
        
        # 外参（简化：相机在原点，朝向+Z）
        self.position = np.array([0, 0, 0], dtype=float)
        
    def project_point(self, point_3d):
        """投影 3D 点到 2D"""
        x, y, z = point_3d
        
        if z <= 0:
            return None, -1
        
        u = self.focal_length * x / z + self.cx
        v = self.focal_length * y / z + self.cy
        
        return np.array([u, v]), z


def project_gaussian_to_2d(gaussian, camera):
    """
    将 3D 高斯投影到 2D
    
    Returns:
        mean_2d: (2,) 2D 中心
        cov_2d: (2, 2) 2D 协方差
        depth: float 深度
    """
    # 投影中心
    mean_2d, depth = camera.project_point(gaussian.position)
    
    if mean_2d is None:
        return None, None, -1
    
    # 获取 3D 协方差
    cov_3d = gaussian.get_covariance_3d()
    
    # 雅可比矩阵（透视投影的导数）
    x, y, z = gaussian.position
    fx, fy = camera.focal_length, camera.focal_length
    
    J = np.array([
        [fx / z, 0, -fx * x / (z**2)],
        [0, fy / z, -fy * y / (z**2)]
    ])
    
    # 投影协方差：Σ_2d = J Σ_3d J^T
    cov_2d = J @ cov_3d @ J.T
    
    # 确保对称性和数值稳定性
    cov_2d = (cov_2d + cov_2d.T) / 2
    cov_2d += np.eye(2) * 1e-6  # 正则化
    
    return mean_2d, cov_2d, depth


def evaluate_gaussian_2d(pixel, mean, cov_inv):
    """评估 2D 高斯在某个像素的值"""
    diff = pixel - mean
    exponent = -0.5 * diff @ cov_inv @ diff
    # 限制指数范围，避免溢出
    exponent = np.clip(exponent, -50, 0)
    return np.exp(exponent)


class GaussianRasterizer:
    """高斯光栅化器"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        """重置缓冲区"""
        self.color_buffer = np.zeros((self.height, self.width, 3))
        
    def get_gaussian_bbox(self, mean, cov, threshold=3.0):
        """计算高斯的边界框"""
        eigenvalues = np.linalg.eigvalsh(cov)
        max_radius = threshold * np.sqrt(eigenvalues.max())
        
        u_min = int(max(0, mean[0] - max_radius))
        u_max = int(min(self.width, mean[0] + max_radius + 1))
        v_min = int(max(0, mean[1] - max_radius))
        v_max = int(min(self.height, mean[1] + max_radius + 1))
        
        return u_min, u_max, v_min, v_max
    
    def render(self, gaussians, camera):
        """
        渲染高斯场景
        
        Args:
            gaussians: List[Gaussian3D]
            camera: GaussianCamera
            
        Returns:
            image: (H, W, 3)
        """
        self.reset()
        
        # 1. 投影所有高斯
        gaussians_2d = []
        for g in gaussians:
            mean_2d, cov_2d, depth = project_gaussian_to_2d(g, camera)
            
            if mean_2d is None or depth <= 0:
                continue
            
            # 视锥剔除（简单版本）
            if (mean_2d[0] < -100 or mean_2d[0] > self.width + 100 or
                mean_2d[1] < -100 or mean_2d[1] > self.height + 100):
                continue
            
            gaussians_2d.append({
                'mean': mean_2d,
                'cov': cov_2d,
                'color': g.color,
                'opacity': g.opacity,
                'depth': depth
            })
        
        if len(gaussians_2d) == 0:
            return self.color_buffer
        
        # 2. 深度排序（从后往前）
        gaussians_2d.sort(key=lambda x: x['depth'], reverse=True)
        
        # 3. 光栅化（逐像素）
        for v in range(self.height):
            for u in range(self.width):
                pixel = np.array([u, v], dtype=float)
                
                # 累积颜色
                color = np.zeros(3)
                transmittance = 1.0
                
                # 从后往前混合
                for g2d in gaussians_2d:
                    try:
                        cov_inv = np.linalg.inv(g2d['cov'])
                    except np.linalg.LinAlgError:
                        continue
                    
                    # 计算高斯值
                    gaussian_value = evaluate_gaussian_2d(pixel, g2d['mean'], cov_inv)
                    
                    # 有效 alpha
                    alpha = g2d['opacity'] * gaussian_value
                    alpha = np.clip(alpha, 0, 1)
                    
                    # 跳过贡献很小的高斯
                    if alpha < 0.001:
                        continue
                    
                    # 累积颜色
                    weight = alpha * transmittance
                    color += weight * g2d['color']
                    
                    # 更新透射率
                    transmittance *= (1 - alpha)
                    
                    # 提前终止
                    if transmittance < 0.01:
                        break
                
                self.color_buffer[v, u] = np.clip(color, 0, 1)
        
        return self.color_buffer
    
    def render_optimized(self, gaussians, camera):
        """
        优化版本的渲染（使用边界框）
        """
        self.reset()
        
        # 投影
        gaussians_2d = []
        for g in gaussians:
            mean_2d, cov_2d, depth = project_gaussian_to_2d(g, camera)
            
            if mean_2d is None or depth <= 0:
                continue
            
            gaussians_2d.append({
                'mean': mean_2d,
                'cov': cov_2d,
                'color': g.color,
                'opacity': g.opacity,
                'depth': depth
            })
        
        if len(gaussians_2d) == 0:
            return self.color_buffer
        
        # 深度排序
        gaussians_2d.sort(key=lambda x: x['depth'], reverse=True)
        
        # 对每个高斯，只在其边界框内光栅化
        transmittance_map = np.ones((self.height, self.width))
        
        for g2d in gaussians_2d:
            try:
                cov_inv = np.linalg.inv(g2d['cov'])
            except np.linalg.LinAlgError:
                continue
            
            # 获取边界框
            u_min, u_max, v_min, v_max = self.get_gaussian_bbox(g2d['mean'], g2d['cov'])
            
            # 在边界框内光栅化
            for v in range(v_min, v_max):
                for u in range(u_min, u_max):
                    pixel = np.array([u, v], dtype=float)
                    
                    gaussian_value = evaluate_gaussian_2d(pixel, g2d['mean'], cov_inv)
                    alpha = g2d['opacity'] * gaussian_value
                    alpha = np.clip(alpha, 0, 1)
                    
                    if alpha < 0.001:
                        continue
                    
                    weight = alpha * transmittance_map[v, u]
                    self.color_buffer[v, u] += weight * g2d['color']
                    transmittance_map[v, u] *= (1 - alpha)
        
        return np.clip(self.color_buffer, 0, 1)


def demo_single_gaussian():
    """演示单个高斯的渲染"""
    
    # 创建高斯
    gaussian = Gaussian3D(
        position=[0, 0, 5],
        color=[1, 0.5, 0],
        scale=[0.5, 0.3, 0.4],
        rotation=[np.cos(np.pi/8), 0, np.sin(np.pi/8), 0],
        opacity=1.0
    )
    
    # 相机
    camera = GaussianCamera(width=256, height=256, fov=60)
    
    # 渲染
    rasterizer = GaussianRasterizer(256, 256)
    image = rasterizer.render([gaussian], camera)
    
    # 可视化
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title('单个高斯渲染', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gaussian_rasterizer_single.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_rasterizer_single.png'")
    plt.show()


def demo_multiple_gaussians():
    """演示多个高斯的渲染"""
    
    # 创建多个高斯
    gaussians = [
        # 红色球
        Gaussian3D([0, 0, 8], [1, 0, 0], [0.6, 0.6, 0.6], opacity=0.9),
        # 绿色椭球
        Gaussian3D([1.5, 0, 6], [0, 1, 0], [0.4, 0.8, 0.3], 
                  rotation=[np.cos(np.pi/6), 0, 0, np.sin(np.pi/6)], opacity=0.8),
        # 蓝色扁平
        Gaussian3D([-1, 0.5, 7], [0, 0, 1], [0.7, 0.7, 0.2], 
                  rotation=[np.cos(np.pi/4), np.sin(np.pi/4), 0, 0], opacity=0.7),
        # 黄色小球
        Gaussian3D([0, -1, 5], [1, 1, 0], [0.3, 0.3, 0.3], opacity=1.0),
    ]
    
    camera = GaussianCamera(width=512, height=512, fov=60)
    
    # 对比两种渲染方法
    rasterizer = GaussianRasterizer(512, 512)
    
    import time
    
    start = time.time()
    image1 = rasterizer.render(gaussians, camera)
    time1 = time.time() - start
    
    start = time.time()
    image2 = rasterizer.render_optimized(gaussians, camera)
    time2 = time.time() - start
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].imshow(image1)
    axes[0].set_title(f'标准渲染\n时间: {time1:.3f}s', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(image2)
    axes[1].set_title(f'优化渲染（边界框）\n时间: {time2:.3f}s', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('gaussian_rasterizer_multiple.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_rasterizer_multiple.png'")
    print(f"加速比: {time1/time2:.2f}x")
    plt.show()


def demo_opacity_effects():
    """演示不透明度的效果"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('不透明度效果', fontsize=16)
    
    opacities = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    
    for i, opacity in enumerate(opacities):
        ax = axes[i // 3, i % 3]
        
        # 两个重叠的高斯
        gaussians = [
            Gaussian3D([0, 0, 8], [1, 0, 0], [0.8, 0.8, 0.8], opacity=opacity),
            Gaussian3D([0.5, 0, 7], [0, 0, 1], [0.8, 0.8, 0.8], opacity=opacity),
        ]
        
        camera = GaussianCamera(width=256, height=256, fov=60)
        rasterizer = GaussianRasterizer(256, 256)
        image = rasterizer.render_optimized(gaussians, camera)
        
        ax.imshow(image)
        ax.set_title(f'Opacity = {opacity}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('gaussian_rasterizer_opacity.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_rasterizer_opacity.png'")
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("高斯光栅化器演示")
    print("=" * 60)
    
    print("\n1. 演示单个高斯...")
    demo_single_gaussian()
    
    print("\n2. 演示多个高斯...")
    demo_multiple_gaussians()
    
    print("\n3. 演示不透明度效果...")
    demo_opacity_effects()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
