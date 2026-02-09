"""
3D 高斯可视化
Visualization of 3D Gaussian distributions

这个脚本演示：
1. 3D 高斯的参数化（位置、旋转、缩放）
2. 3D 高斯到 2D 的投影
3. 椭球体的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse


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


def get_covariance_matrix(rotation_quat, scale):
    """
    计算 3D 协方差矩阵
    
    Args:
        rotation_quat: (4,) 旋转四元数 [w, x, y, z]
        scale: (3,) 缩放 [sx, sy, sz]
        
    Returns:
        Sigma: (3, 3) 协方差矩阵
    """
    R = quaternion_to_rotation_matrix(rotation_quat)
    S = np.diag(scale)
    Sigma = R @ S @ S.T @ R.T
    return Sigma


def plot_3d_ellipsoid(ax, center, covariance, color='blue', alpha=0.3, n_points=30):
    """
    绘制 3D 椭球体
    
    Args:
        ax: matplotlib 3D axis
        center: (3,) 椭球中心
        covariance: (3, 3) 协方差矩阵
        color: 颜色
        alpha: 透明度
        n_points: 采样点数
    """
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    
    # 生成单位球面上的点
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # 堆叠成 (3, n_points, n_points)
    sphere = np.stack([x, y, z], axis=0)
    
    # 缩放和旋转
    # 椭球 = center + eigenvectors @ diag(sqrt(eigenvalues)) @ sphere
    for i in range(n_points):
        for j in range(n_points):
            point = sphere[:, i, j]
            # 缩放
            scaled = np.sqrt(eigenvalues) * point
            # 旋转
            rotated = eigenvectors @ scaled
            # 平移
            sphere[:, i, j] = center + rotated
    
    # 绘制
    ax.plot_surface(sphere[0], sphere[1], sphere[2], 
                   color=color, alpha=alpha, edgecolor='none')
    
    # 绘制主轴
    for i in range(3):
        direction = eigenvectors[:, i] * np.sqrt(eigenvalues[i])
        ax.plot([center[0], center[0] + direction[0]],
               [center[1], center[1] + direction[1]],
               [center[2], center[2] + direction[2]],
               'r-', linewidth=2)


def demo_3d_gaussian_basic():
    """演示基本的 3D 高斯椭球"""
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 球形高斯
    ax1 = fig.add_subplot(131, projection='3d')
    center1 = np.array([0, 0, 0])
    rotation1 = np.array([1, 0, 0, 0])  # 无旋转
    scale1 = np.array([1, 1, 1])  # 各向同性
    cov1 = get_covariance_matrix(rotation1, scale1)
    
    plot_3d_ellipsoid(ax1, center1, cov1, color='blue', alpha=0.3)
    ax1.scatter(*center1, color='red', s=100, marker='*')
    ax1.set_title('球形高斯\nScale=[1,1,1]', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    set_axes_equal(ax1)
    
    # 2. 拉伸的椭球
    ax2 = fig.add_subplot(132, projection='3d')
    center2 = np.array([0, 0, 0])
    rotation2 = np.array([1, 0, 0, 0])  # 无旋转
    scale2 = np.array([2, 1, 0.5])  # 不同缩放
    cov2 = get_covariance_matrix(rotation2, scale2)
    
    plot_3d_ellipsoid(ax2, center2, cov2, color='green', alpha=0.3)
    ax2.scatter(*center2, color='red', s=100, marker='*')
    ax2.set_title('拉伸椭球\nScale=[2,1,0.5]', fontsize=12)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    set_axes_equal(ax2)
    
    # 3. 旋转的椭球
    ax3 = fig.add_subplot(133, projection='3d')
    center3 = np.array([0, 0, 0])
    # 绕 Z 轴旋转 45 度
    angle = np.pi / 4
    rotation3 = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    scale3 = np.array([2, 0.5, 1])
    cov3 = get_covariance_matrix(rotation3, scale3)
    
    plot_3d_ellipsoid(ax3, center3, cov3, color='orange', alpha=0.3)
    ax3.scatter(*center3, color='red', s=100, marker='*')
    ax3.set_title('旋转椭球\n45° around Z', fontsize=12)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    set_axes_equal(ax3)
    
    plt.tight_layout()
    plt.savefig('gaussian_3d_basic.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_3d_basic.png'")
    plt.show()


def demo_projection():
    """演示 3D 高斯投影到 2D"""
    
    fig = plt.figure(figsize=(16, 6))
    
    # 定义 3D 高斯
    center_3d = np.array([0, 0, 5])  # 在相机前方
    rotation = np.array([np.cos(np.pi/8), 0, np.sin(np.pi/8), 0])  # 稍微旋转
    scale = np.array([1.5, 0.8, 1.0])
    cov_3d = get_covariance_matrix(rotation, scale)
    
    # 1. 3D 视图
    ax1 = fig.add_subplot(131, projection='3d')
    plot_3d_ellipsoid(ax1, center_3d, cov_3d, color='blue', alpha=0.3)
    ax1.scatter(*center_3d, color='red', s=100, marker='*')
    
    # 绘制相机
    camera_pos = np.array([0, 0, 0])
    ax1.scatter(*camera_pos, color='black', s=200, marker='^', label='Camera')
    
    # 视线
    ax1.plot([camera_pos[0], center_3d[0]],
            [camera_pos[1], center_3d[1]],
            [camera_pos[2], center_3d[2]],
            'k--', alpha=0.5)
    
    ax1.set_title('3D 高斯', fontsize=12)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    set_axes_equal(ax1)
    
    # 2. 简化的投影计算（针孔相机）
    # 相机参数
    focal_length = 500  # 像素
    
    # 投影中心
    z = center_3d[2]
    center_2d = np.array([
        focal_length * center_3d[0] / z,
        focal_length * center_3d[1] / z
    ])
    
    # 简化的 2D 协方差（仅考虑 XY 平面的投影）
    # 这是一个简化版本，完整版本需要雅可比矩阵
    cov_2d_simple = cov_3d[:2, :2] * (focal_length / z) ** 2
    
    # 3. 侧视图（XZ 平面）
    ax2 = fig.add_subplot(132)
    
    # 绘制椭圆的轮廓（XZ 平面）
    eigenvalues_xz = np.linalg.eigvalsh(cov_3d[[0, 2]][:, [0, 2]])
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_points = np.array([
        np.sqrt(eigenvalues_xz[0]) * np.cos(theta),
        np.sqrt(eigenvalues_xz[1]) * np.sin(theta)
    ])
    
    # 旋转
    R_xz = quaternion_to_rotation_matrix(rotation)[[0, 2]][:, [0, 2]]
    ellipse_rotated = R_xz @ ellipse_points
    ellipse_rotated[0] += center_3d[0]
    ellipse_rotated[1] += center_3d[2]
    
    ax2.plot(ellipse_rotated[0], ellipse_rotated[1], 'b-', linewidth=2, label='3D Gaussian')
    ax2.scatter(center_3d[0], center_3d[2], color='red', s=100, marker='*')
    ax2.scatter(0, 0, color='black', s=200, marker='^', label='Camera')
    
    # 投影线
    ax2.plot([0, center_3d[0]], [0, center_3d[2]], 'k--', alpha=0.5)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z (depth)')
    ax2.set_title('侧视图 (XZ平面)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # 4. 投影后的 2D 高斯
    ax3 = fig.add_subplot(133)
    
    # 绘制 2D 高斯
    eigenvalues_2d, eigenvectors_2d = np.linalg.eig(cov_2d_simple)
    angle = np.degrees(np.arctan2(eigenvectors_2d[1, 0], eigenvectors_2d[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues_2d)
    
    ellipse = Ellipse(center_2d, width, height, angle=angle,
                     facecolor='blue', alpha=0.3, edgecolor='blue', linewidth=2)
    ax3.add_patch(ellipse)
    ax3.scatter(*center_2d, color='red', s=100, marker='*', zorder=5)
    
    # 设置范围
    margin = max(width, height) * 1.5
    ax3.set_xlim(center_2d[0] - margin, center_2d[0] + margin)
    ax3.set_ylim(center_2d[1] - margin, center_2d[1] + margin)
    
    ax3.set_xlabel('u (pixels)')
    ax3.set_ylabel('v (pixels)')
    ax3.set_title('投影后的 2D 高斯', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.tight_layout()
    plt.savefig('gaussian_3d_projection.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_3d_projection.png'")
    plt.show()


def demo_multiple_3d_gaussians():
    """演示多个 3D 高斯"""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义多个高斯
    gaussians = [
        {
            'center': np.array([0, 0, 5]),
            'rotation': np.array([1, 0, 0, 0]),
            'scale': np.array([1, 1, 1]),
            'color': 'blue'
        },
        {
            'center': np.array([2, 1, 6]),
            'rotation': np.array([np.cos(np.pi/6), 0, 0, np.sin(np.pi/6)]),
            'scale': np.array([0.8, 0.5, 1.2]),
            'color': 'green'
        },
        {
            'center': np.array([-1.5, -1, 4]),
            'rotation': np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]),
            'scale': np.array([1.5, 0.6, 0.8]),
            'color': 'red'
        },
        {
            'center': np.array([0, 2, 7]),
            'rotation': np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0, 0]),
            'scale': np.array([0.7, 1.0, 0.5]),
            'color': 'orange'
        },
    ]
    
    # 绘制所有高斯
    for i, g in enumerate(gaussians):
        cov = get_covariance_matrix(g['rotation'], g['scale'])
        plot_3d_ellipsoid(ax, g['center'], cov, color=g['color'], alpha=0.3)
        ax.scatter(*g['center'], color='black', s=100, marker='*')
        ax.text(g['center'][0], g['center'][1], g['center'][2] + 0.3, 
               f'G{i+1}', fontsize=12)
    
    # 相机位置
    camera_pos = np.array([0, 0, 0])
    ax.scatter(*camera_pos, color='black', s=300, marker='^', label='Camera')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('多个 3D 高斯场景', fontsize=14)
    ax.legend()
    set_axes_equal(ax)
    
    plt.tight_layout()
    plt.savefig('gaussian_3d_multiple.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_3d_multiple.png'")
    plt.show()


def set_axes_equal(ax):
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


if __name__ == '__main__':
    print("=" * 60)
    print("3D 高斯可视化演示")
    print("=" * 60)
    
    print("\n1. 演示基本的 3D 高斯形状...")
    demo_3d_gaussian_basic()
    
    print("\n2. 演示 3D 到 2D 的投影...")
    demo_projection()
    
    print("\n3. 演示多个 3D 高斯...")
    demo_multiple_3d_gaussians()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
