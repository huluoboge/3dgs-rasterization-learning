"""
简单光栅化器
Simple Rasterizer

演示基础的光栅化概念：
1. 点的光栅化
2. 简单的混合模式
3. 深度排序
"""

import numpy as np
import matplotlib.pyplot as plt


class SimplePoint:
    """简单的点表示"""
    
    def __init__(self, position, color, size=1.0, opacity=1.0):
        """
        Args:
            position: (3,) 3D 位置 [x, y, z]
            color: (3,) RGB 颜色 [r, g, b]
            size: float 点的大小（半径）
            opacity: float 不透明度 [0, 1]
        """
        self.position = np.array(position, dtype=float)
        self.color = np.array(color, dtype=float)
        self.size = size
        self.opacity = opacity


class SimpleCamera:
    """简单的相机"""
    
    def __init__(self, width, height, focal_length=500):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.position = np.array([0.0, 0.0, 0.0])
        
    def project(self, point_3d):
        """
        投影 3D 点到 2D
        
        Args:
            point_3d: (3,) 3D 位置
            
        Returns:
            point_2d: (2,) 2D 位置 [u, v]
            depth: float 深度
        """
        # 相机坐标（简化：假设相机在原点，朝向 +Z）
        x, y, z = point_3d
        
        if z <= 0:
            return None, -1  # 在相机后面
        
        # 透视投影
        u = self.focal_length * x / z + self.width / 2
        v = self.focal_length * y / z + self.height / 2
        
        return np.array([u, v]), z


class SimpleRasterizer:
    """简单的光栅化器"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()
        
    def reset(self):
        """重置图像缓冲区"""
        self.color_buffer = np.zeros((self.height, self.width, 3))
        self.depth_buffer = np.full((self.height, self.width), np.inf)
        
    def rasterize_point(self, center_2d, depth, color, size, opacity=1.0):
        """
        光栅化一个点（圆形）
        
        Args:
            center_2d: (2,) 2D 中心 [u, v]
            depth: float 深度值
            color: (3,) RGB 颜色
            size: float 半径（像素）
            opacity: float 不透明度
        """
        u_center, v_center = center_2d
        
        # 计算边界框
        u_min = int(max(0, u_center - size))
        u_max = int(min(self.width, u_center + size + 1))
        v_min = int(max(0, v_center - size))
        v_max = int(min(self.height, v_center + size + 1))
        
        # 遍历边界框内的像素
        for v in range(v_min, v_max):
            for u in range(u_min, u_max):
                # 计算距离
                dist = np.sqrt((u - u_center)**2 + (v - v_center)**2)
                
                # 检查是否在圆内
                if dist <= size:
                    # 简单的圆形权重（可以改为高斯）
                    weight = 1.0 - (dist / size)
                    weight = max(0, min(1, weight))
                    
                    # Alpha 混合
                    alpha = opacity * weight
                    self.color_buffer[v, u] = (
                        self.color_buffer[v, u] * (1 - alpha) +
                        color * alpha
                    )
        
    def rasterize_point_with_depth(self, center_2d, depth, color, size, opacity=1.0):
        """
        带深度测试的点光栅化
        
        Args:
            center_2d: (2,) 2D 中心
            depth: float 深度值
            color: (3,) RGB 颜色
            size: float 半径
            opacity: float 不透明度
        """
        u_center, v_center = center_2d
        
        u_min = int(max(0, u_center - size))
        u_max = int(min(self.width, u_center + size + 1))
        v_min = int(max(0, v_center - size))
        v_max = int(min(self.height, v_center + size + 1))
        
        for v in range(v_min, v_max):
            for u in range(u_min, u_max):
                dist = np.sqrt((u - u_center)**2 + (v - v_center)**2)
                
                if dist <= size:
                    # 深度测试
                    if depth < self.depth_buffer[v, u]:
                        weight = 1.0 - (dist / size)
                        weight = max(0, min(1, weight))
                        
                        alpha = opacity * weight
                        self.color_buffer[v, u] = (
                            self.color_buffer[v, u] * (1 - alpha) +
                            color * alpha
                        )
                        
                        # 更新深度（如果完全不透明）
                        if alpha > 0.99:
                            self.depth_buffer[v, u] = depth
    
    def render_scene(self, points, camera, use_depth_test=False):
        """
        渲染整个场景
        
        Args:
            points: List[SimplePoint]
            camera: SimpleCamera
            use_depth_test: bool 是否使用深度测试
            
        Returns:
            image: (H, W, 3) 渲染图像
        """
        self.reset()
        
        # 投影所有点
        projected = []
        for point in points:
            pos_2d, depth = camera.project(point.position)
            if pos_2d is not None:
                projected.append({
                    'pos_2d': pos_2d,
                    'depth': depth,
                    'color': point.color,
                    'size': point.size,
                    'opacity': point.opacity
                })
        
        # 深度排序（从后往前）
        projected.sort(key=lambda x: x['depth'], reverse=True)
        
        # 光栅化每个点
        for p in projected:
            if use_depth_test:
                self.rasterize_point_with_depth(
                    p['pos_2d'], p['depth'], p['color'], 
                    p['size'], p['opacity']
                )
            else:
                self.rasterize_point(
                    p['pos_2d'], p['depth'], p['color'], 
                    p['size'], p['opacity']
                )
        
        return np.clip(self.color_buffer, 0, 1)


def demo_simple_points():
    """演示简单的点光栅化"""
    
    # 创建场景
    points = [
        SimplePoint([0, 0, 10], [1, 0, 0], size=50, opacity=1.0),    # 红色
        SimplePoint([1, 0, 8], [0, 1, 0], size=40, opacity=0.8),     # 绿色
        SimplePoint([-1, 0, 12], [0, 0, 1], size=60, opacity=0.6),   # 蓝色
    ]
    
    # 相机
    camera = SimpleCamera(width=512, height=512, focal_length=400)
    
    # 光栅化
    rasterizer = SimpleRasterizer(512, 512)
    
    # 不使用深度测试
    image_no_depth = rasterizer.render_scene(points, camera, use_depth_test=False)
    
    # 使用深度测试
    image_with_depth = rasterizer.render_scene(points, camera, use_depth_test=True)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image_no_depth)
    axes[0].set_title('无深度测试\n（按深度排序 + Alpha 混合）', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(image_with_depth)
    axes[1].set_title('有深度测试\n（Z-buffer）', fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_rasterizer_points.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'simple_rasterizer_points.png'")
    plt.show()


def demo_alpha_blending():
    """演示 Alpha 混合"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Alpha 混合演示', fontsize=16)
    
    camera = SimpleCamera(width=256, height=256, focal_length=200)
    
    opacities = [1.0, 0.7, 0.5, 0.3, 0.1, 0.05]
    
    for i, opacity in enumerate(opacities):
        ax = axes[i // 3, i % 3]
        
        # 创建重叠的点
        points = [
            SimplePoint([0, 0, 10], [1, 0, 0], size=40, opacity=opacity),
            SimplePoint([0.5, 0, 10], [0, 1, 0], size=40, opacity=opacity),
            SimplePoint([0.25, 0.5, 10], [0, 0, 1], size=40, opacity=opacity),
        ]
        
        rasterizer = SimpleRasterizer(256, 256)
        image = rasterizer.render_scene(points, camera)
        
        ax.imshow(image)
        ax.set_title(f'Opacity = {opacity}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_rasterizer_alpha.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'simple_rasterizer_alpha.png'")
    plt.show()


def demo_depth_ordering():
    """演示深度排序的重要性"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('深度排序演示', fontsize=16)
    
    camera = SimpleCamera(width=256, height=256, focal_length=200)
    
    # 创建三个不同深度的点
    points_correct = [
        SimplePoint([0, 0, 15], [1, 0, 0], size=50, opacity=0.7),  # 红色，远
        SimplePoint([0, 0, 10], [0, 1, 0], size=50, opacity=0.7),  # 绿色，中
        SimplePoint([0, 0, 5], [0, 0, 1], size=50, opacity=0.7),   # 蓝色，近
    ]
    
    # 错误的顺序
    points_wrong = points_correct[::-1]
    
    # 1. 正确排序
    rasterizer = SimpleRasterizer(256, 256)
    image_correct = rasterizer.render_scene(points_correct, camera)
    axes[0].imshow(image_correct)
    axes[0].set_title('正确排序\n（远 → 近）', fontsize=12)
    axes[0].axis('off')
    
    # 2. 错误排序（不排序）
    rasterizer = SimpleRasterizer(256, 256)
    for p in points_wrong:
        pos_2d, depth = camera.project(p.position)
        if pos_2d is not None:
            rasterizer.rasterize_point(pos_2d, depth, p.color, p.size, p.opacity)
    image_wrong = rasterizer.color_buffer
    axes[1].imshow(image_wrong)
    axes[1].set_title('错误排序\n（近 → 远）', fontsize=12)
    axes[1].axis('off')
    
    # 3. 使用深度测试
    rasterizer = SimpleRasterizer(256, 256)
    image_depth_test = rasterizer.render_scene(points_wrong, camera, use_depth_test=True)
    axes[2].imshow(image_depth_test)
    axes[2].set_title('深度测试\n（任意顺序）', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_rasterizer_depth.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'simple_rasterizer_depth.png'")
    plt.show()


if __name__ == '__main__':
    print("=" * 60)
    print("简单光栅化器演示")
    print("=" * 60)
    
    print("\n1. 演示简单的点光栅化...")
    demo_simple_points()
    
    print("\n2. 演示 Alpha 混合...")
    demo_alpha_blending()
    
    print("\n3. 演示深度排序...")
    demo_depth_ordering()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("关键概念：")
    print("  - 光栅化：将矢量图形转换为像素")
    print("  - Alpha 混合：根据透明度组合颜色")
    print("  - 深度排序：从后往前绘制")
    print("  - 深度测试：使用 Z-buffer 处理遮挡")
    print("=" * 60)
