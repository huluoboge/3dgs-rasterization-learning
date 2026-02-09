"""
2D 高斯可视化
Visualization of 2D Gaussian distributions

这个脚本演示：
1. 如何创建和可视化 2D 高斯分布
2. 协方差矩阵对形状的影响
3. 旋转和缩放的效果
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

def gaussian_2d(x, y, mu, sigma):
    """
    计算 2D 高斯函数值
    
    Args:
        x, y: 坐标点（可以是标量或数组）
        mu: (2,) 均值向量 [mu_x, mu_y]
        sigma: (2, 2) 协方差矩阵
        
    Returns:
        高斯函数值
    """
    pos = np.dstack([x, y])
    mu = np.array(mu)
    sigma = np.array(sigma)
    
    # 计算协方差矩阵的逆和行列式
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    
    # 计算归一化系数
    norm = 1.0 / (2 * np.pi * np.sqrt(sigma_det))
    
    # 计算指数部分
    diff = pos - mu
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, sigma_inv, diff)
    
    return norm * np.exp(exponent)


def plot_gaussian_2d(mu, sigma, title="2D Gaussian", ax=None):
    """
    绘制 2D 高斯分布
    
    Args:
        mu: (2,) 均值向量
        sigma: (2, 2) 协方差矩阵
        title: 图表标题
        ax: matplotlib axis，如果为 None 则创建新图
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建网格
    x = np.linspace(mu[0] - 4, mu[0] + 4, 200)
    y = np.linspace(mu[1] - 4, mu[1] + 4, 200)
    X, Y = np.meshgrid(x, y)
    
    # 计算高斯值
    Z = gaussian_2d(X, Y, mu, sigma)
    
    # 绘制等高线
    contour = ax.contour(X, Y, Z, levels=8, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 绘制填充等高线
    ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
    
    # 绘制椭圆（1-sigma 边界）
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(eigenvalues)  # 1-sigma 边界
    
    ellipse = Ellipse(mu, width, height, angle=angle,
                     facecolor='none', edgecolor='red', linewidth=2)
    ax.add_patch(ellipse)
    
    # 标记中心点
    ax.plot(mu[0], mu[1], 'r*', markersize=15, label='Center')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    return ax


def demo_covariance_effects():
    """演示不同协方差矩阵的效果"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('协方差矩阵对高斯形状的影响', fontsize=16, fontproperties='SimHei')
    
    mu = [0, 0]  # 中心点
    
    # 1. 圆形高斯（各向同性）
    sigma1 = np.array([[1, 0], [0, 1]])
    plot_gaussian_2d(mu, sigma1, 
                     title='圆形高斯\nΣ = [[1, 0], [0, 1]]', 
                     ax=axes[0, 0])
    
    # 2. 水平拉伸
    sigma2 = np.array([[4, 0], [0, 1]])
    plot_gaussian_2d(mu, sigma2, 
                     title='水平拉伸\nΣ = [[4, 0], [0, 1]]', 
                     ax=axes[0, 1])
    
    # 3. 垂直拉伸
    sigma3 = np.array([[1, 0], [0, 4]])
    plot_gaussian_2d(mu, sigma3, 
                     title='垂直拉伸\nΣ = [[1, 0], [0, 4]]', 
                     ax=axes[0, 2])
    
    # 4. 正相关
    sigma4 = np.array([[2, 1.5], [1.5, 2]])
    plot_gaussian_2d(mu, sigma4, 
                     title='正相关\nΣ = [[2, 1.5], [1.5, 2]]', 
                     ax=axes[1, 0])
    
    # 5. 负相关
    sigma5 = np.array([[2, -1.5], [-1.5, 2]])
    plot_gaussian_2d(mu, sigma5, 
                     title='负相关\nΣ = [[2, -1.5], [-1.5, 2]]', 
                     ax=axes[1, 1])
    
    # 6. 旋转的椭圆
    theta = np.pi / 4  # 45度旋转
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    S = np.array([[3, 0], [0, 0.5]])
    sigma6 = R @ S @ S.T @ R.T
    plot_gaussian_2d(mu, sigma6, 
                     title='45°旋转椭圆\nΣ = R S Sᵀ Rᵀ', 
                     ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('gaussian_2d_covariance_effects.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_2d_covariance_effects.png'")
    plt.show()


def demo_rotation():
    """演示旋转效果"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('高斯旋转演示', fontsize=16)
    
    mu = [0, 0]
    S = np.array([[2, 0], [0, 0.5]])  # 基础缩放
    
    angles = [0, 30, 60, 90, 120, 150]
    
    for i, angle_deg in enumerate(angles):
        ax = axes[i // 3, i % 3]
        
        # 计算旋转矩阵
        theta = np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        # 计算协方差矩阵
        sigma = R @ S @ S.T @ R.T
        
        plot_gaussian_2d(mu, sigma, 
                        title=f'旋转 {angle_deg}°', 
                        ax=ax)
    
    plt.tight_layout()
    plt.savefig('gaussian_2d_rotation.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_2d_rotation.png'")
    plt.show()


def demo_multiple_gaussians():
    """演示多个高斯的混合"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建网格
    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)
    
    # 定义多个高斯
    gaussians = [
        {'mu': [0, 0], 'sigma': [[1, 0], [0, 1]], 'weight': 1.0},
        {'mu': [2, 2], 'sigma': [[0.5, 0.2], [0.2, 0.5]], 'weight': 0.8},
        {'mu': [-2, 2], 'sigma': [[0.8, -0.3], [-0.3, 0.8]], 'weight': 0.6},
        {'mu': [0, -2.5], 'sigma': [[1.5, 0], [0, 0.3]], 'weight': 0.7},
    ]
    
    # 计算混合高斯
    Z = np.zeros_like(X)
    for g in gaussians:
        Z += g['weight'] * gaussian_2d(X, Y, g['mu'], g['sigma'])
    
    # 绘制
    contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    contour = ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.4, linewidths=0.5)
    
    # 标记中心点
    for i, g in enumerate(gaussians):
        ax.plot(g['mu'][0], g['mu'][1], 'r*', markersize=15)
        ax.text(g['mu'][0], g['mu'][1] + 0.3, f'G{i+1}', 
               ha='center', fontsize=12, color='red', weight='bold')
    
    plt.colorbar(contourf, ax=ax, label='Intensity')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('多个高斯的混合', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('gaussian_2d_mixture.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'gaussian_2d_mixture.png'")
    plt.show()


def interactive_demo():
    """
    交互式演示（简化版）
    可以修改参数来观察效果
    """
    print("\n=== 交互式 2D 高斯演示 ===\n")
    
    # 默认参数
    mu = [0, 0]
    sigma_xx = 2.0
    sigma_yy = 1.0
    sigma_xy = 0.5
    
    print(f"当前参数：")
    print(f"  均值 μ = {mu}")
    print(f"  协方差 Σ = [[{sigma_xx}, {sigma_xy}], [{sigma_xy}, {sigma_yy}]]")
    
    sigma = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])
    
    # 绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_gaussian_2d(mu, sigma, 
                    title=f'2D 高斯分布\nμ={mu}, Σ=[[{sigma_xx}, {sigma_xy}], [{sigma_xy}, {sigma_yy}]]',
                    ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    print("\n提示：修改代码中的 sigma_xx, sigma_yy, sigma_xy 参数来观察不同效果")


if __name__ == '__main__':
    print("=" * 60)
    print("2D 高斯可视化演示")
    print("=" * 60)
    
    # 运行各种演示
    print("\n1. 演示不同协方差矩阵的效果...")
    demo_covariance_effects()
    
    print("\n2. 演示旋转效果...")
    demo_rotation()
    
    print("\n3. 演示多个高斯的混合...")
    demo_multiple_gaussians()
    
    print("\n4. 交互式演示...")
    interactive_demo()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
