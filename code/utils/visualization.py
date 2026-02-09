"""
可视化工具函数
Visualization Utilities

提供各种可视化辅助函数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_gaussian_contour(mean, cov, ax=None, color='blue', alpha=0.5, levels=5):
    """
    绘制 2D 高斯的等高线
    
    Args:
        mean: (2,) 均值
        cov: (2, 2) 协方差
        ax: matplotlib axis
        color: 颜色
        alpha: 透明度
        levels: 等高线层数
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # 创建网格
    x_range = 4 * np.sqrt(cov[0, 0])
    y_range = 4 * np.sqrt(cov[1, 1])
    
    x = np.linspace(mean[0] - x_range, mean[0] + x_range, 200)
    y = np.linspace(mean[1] - y_range, mean[1] + y_range, 200)
    X, Y = np.meshgrid(x, y)
    
    # 计算高斯值
    pos = np.dstack([X, Y])
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    
    diff = pos - mean
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, cov_inv, diff)
    Z = np.exp(exponent) / (2 * np.pi * np.sqrt(cov_det))
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, levels=levels, colors=color, alpha=alpha)
    ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=alpha * 0.3)
    
    return ax


def plot_gaussian_ellipse(mean, cov, ax=None, color='red', alpha=0.5, n_std=1):
    """
    绘制高斯的椭圆边界
    
    Args:
        mean: (2,) 均值
        cov: (2, 2) 协方差
        ax: matplotlib axis
        color: 颜色
        alpha: 透明度
        n_std: 几个标准差（1-sigma, 2-sigma, etc.）
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # 椭圆参数
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    
    # 绘制椭圆
    ellipse = Ellipse(mean, width, height, angle=angle,
                     facecolor=color, alpha=alpha * 0.3,
                     edgecolor=color, linewidth=2)
    ax.add_patch(ellipse)
    
    # 标记中心
    ax.plot(mean[0], mean[1], 'o', color=color, markersize=8)
    
    return ax


def visualize_depth_map(depths, title="Depth Map"):
    """
    可视化深度图
    
    Args:
        depths: (H, W) 深度值
        title: 标题
    """
    plt.figure(figsize=(10, 8))
    
    # 归一化深度
    depths_normalized = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
    
    plt.imshow(depths_normalized, cmap='gray')
    plt.colorbar(label='Normalized Depth')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_alpha_map(alpha_map, title="Alpha Map"):
    """
    可视化 alpha 通道
    
    Args:
        alpha_map: (H, W) alpha 值
        title: 标题
    """
    plt.figure(figsize=(10, 8))
    
    plt.imshow(alpha_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Alpha')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_images(images, titles, suptitle="Image Comparison"):
    """
    对比多张图像
    
    Args:
        images: List of (H, W, 3) 图像
        titles: List of str 标题
        suptitle: 总标题
    """
    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(suptitle, fontsize=16)
    
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_3d_points(points, colors=None, title="3D Points", s=50):
    """
    绘制 3D 点云
    
    Args:
        points: (N, 3) 点坐标
        colors: (N, 3) RGB 颜色或 None
        title: 标题
        s: 点的大小
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is None:
        colors = 'blue'
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=colors, s=s, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)
    
    # 设置相等的轴比例
    set_axes_equal_3d(ax)
    
    plt.tight_layout()
    plt.show()


def set_axes_equal_3d(ax):
    """设置 3D 坐标轴等比例"""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def plot_camera_frustum(ax, camera_pos, camera_dir, fov, depth=5, color='black'):
    """
    绘制相机视锥
    
    Args:
        ax: matplotlib 3D axis
        camera_pos: (3,) 相机位置
        camera_dir: (3,) 相机朝向
        fov: 视场角（度）
        depth: 视锥深度
        color: 颜色
    """
    # 计算视锥的四个角
    fov_rad = np.radians(fov)
    half_width = depth * np.tan(fov_rad / 2)
    
    # 简化：假设相机朝向 +Z
    corners = [
        camera_pos + depth * camera_dir + np.array([half_width, half_width, 0]),
        camera_pos + depth * camera_dir + np.array([half_width, -half_width, 0]),
        camera_pos + depth * camera_dir + np.array([-half_width, -half_width, 0]),
        camera_pos + depth * camera_dir + np.array([-half_width, half_width, 0]),
    ]
    
    # 绘制视锥边缘
    for corner in corners:
        ax.plot([camera_pos[0], corner[0]],
               [camera_pos[1], corner[1]],
               [camera_pos[2], corner[2]],
               color=color, linestyle='--', alpha=0.5)
    
    # 绘制远平面
    for i in range(4):
        next_i = (i + 1) % 4
        ax.plot([corners[i][0], corners[next_i][0]],
               [corners[i][1], corners[next_i][1]],
               [corners[i][2], corners[next_i][2]],
               color=color, alpha=0.5)


def create_color_wheel(n_colors):
    """
    创建均匀分布的颜色
    
    Args:
        n_colors: 颜色数量
        
    Returns:
        colors: (n_colors, 3) RGB 颜色
    """
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        # 简单的 HSV 到 RGB 转换
        if hue < 1/6:
            r, g, b = 1, 6*hue, 0
        elif hue < 2/6:
            r, g, b = 2-6*hue, 1, 0
        elif hue < 3/6:
            r, g, b = 0, 1, 6*hue-2
        elif hue < 4/6:
            r, g, b = 0, 4-6*hue, 1
        elif hue < 5/6:
            r, g, b = 6*hue-4, 0, 1
        else:
            r, g, b = 1, 0, 6-6*hue
        
        colors.append([r, g, b])
    
    return np.array(colors)


def save_comparison_grid(images, titles, filename, rows=2, cols=2):
    """
    保存图像对比网格
    
    Args:
        images: List of images
        titles: List of titles
        filename: 保存文件名
        rows: 行数
        cols: 列数
    """
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"图像已保存为 '{filename}'")
    plt.close()


# 示例使用
if __name__ == '__main__':
    print("可视化工具函数示例")
    
    # 1. 2D 高斯可视化
    print("\n1. 2D 高斯可视化")
    mean = np.array([0, 0])
    cov = np.array([[2, 1], [1, 1.5]])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_2d_gaussian_contour(mean, cov, ax=ax)
    plot_gaussian_ellipse(mean, cov, ax=ax, n_std=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D 高斯可视化', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig('visualization_utils_demo.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'visualization_utils_demo.png'")
    plt.show()
    
    # 2. 颜色轮
    print("\n2. 颜色轮示例")
    colors = create_color_wheel(12)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, color in enumerate(colors):
        angle = 2 * np.pi * i / len(colors)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.scatter(x, y, c=[color], s=500)
        ax.text(x*1.2, y*1.2, f'{i+1}', ha='center', va='center', fontsize=12)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('颜色轮（12 色）', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('color_wheel_demo.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'color_wheel_demo.png'")
    plt.show()
    
    print("\n工具函数演示完成！")
