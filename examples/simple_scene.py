"""
简单场景渲染示例
Simple Scene Rendering Example

这是一个完整的示例，展示如何：
1. 创建一个简单的 3D 场景
2. 使用高斯表示场景
3. 渲染图像
4. 展示完整流程
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加代码路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

from rasterization.gaussian_rasterizer import (
    Gaussian3D, GaussianCamera, GaussianRasterizer,
    quaternion_to_rotation_matrix
)
from utils.visualization import compare_images, create_color_wheel


def create_simple_scene():
    """创建一个简单的场景"""
    
    gaussians = []
    
    # 1. 地面（扁平的大高斯）
    for x in np.linspace(-3, 3, 5):
        for z in np.linspace(5, 15, 7):
            gaussians.append(Gaussian3D(
                position=[x, -1.5, z],
                color=[0.3, 0.5, 0.3],  # 绿色地面
                scale=[0.8, 0.1, 0.8],
                opacity=0.8
            ))
    
    # 2. 几个彩色球体
    spheres = [
        {'pos': [0, 0, 8], 'color': [1, 0.2, 0.2], 'scale': [0.8, 0.8, 0.8]},  # 红
        {'pos': [2, 0, 10], 'color': [0.2, 1, 0.2], 'scale': [0.6, 0.6, 0.6]},  # 绿
        {'pos': [-2, 0.5, 12], 'color': [0.2, 0.2, 1], 'scale': [0.7, 0.7, 0.7]},  # 蓝
        {'pos': [0, 1, 6], 'color': [1, 1, 0.2], 'scale': [0.5, 0.5, 0.5]},  # 黄
    ]
    
    for sphere in spheres:
        gaussians.append(Gaussian3D(
            position=sphere['pos'],
            color=sphere['color'],
            scale=sphere['scale'],
            opacity=0.95
        ))
    
    return gaussians


def create_colorful_scene():
    """创建一个彩色场景"""
    
    gaussians = []
    colors = create_color_wheel(12)
    
    # 圆形排列的高斯
    radius = 3
    n_gaussians = 12
    
    for i in range(n_gaussians):
        angle = 2 * np.pi * i / n_gaussians
        x = radius * np.cos(angle)
        z = 8 + radius * np.sin(angle)
        
        # 随机旋转
        rot_angle = np.random.rand() * 2 * np.pi
        rotation = np.array([np.cos(rot_angle/2), 0, np.sin(rot_angle/2), 0])
        
        # 随机缩放
        scale = 0.3 + np.random.rand(3) * 0.5
        
        gaussians.append(Gaussian3D(
            position=[x, 0, z],
            color=colors[i],
            scale=scale,
            rotation=rotation,
            opacity=0.9
        ))
    
    # 中心的白色高斯
    gaussians.append(Gaussian3D(
        position=[0, 0, 8],
        color=[1, 1, 1],
        scale=[0.8, 0.8, 0.8],
        opacity=1.0
    ))
    
    return gaussians


def create_layered_scene():
    """创建一个分层场景（测试深度）"""
    
    gaussians = []
    
    # 三层不同深度的高斯
    depths = [15, 10, 5]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 红绿蓝
    
    for depth, color in zip(depths, colors):
        gaussians.append(Gaussian3D(
            position=[0, 0, depth],
            color=color,
            scale=[1.5, 1.5, 0.5],
            opacity=0.6
        ))
    
    return gaussians


def render_from_multiple_views(gaussians, views_config):
    """从多个视角渲染场景"""
    
    images = []
    titles = []
    
    for config in views_config:
        # 创建相机
        camera = GaussianCamera(
            width=config['width'],
            height=config['height'],
            fov=config.get('fov', 60)
        )
        
        # 设置相机位置（简化：通过修改高斯位置实现）
        # 在完整实现中应该使用视图矩阵
        
        # 渲染
        rasterizer = GaussianRasterizer(config['width'], config['height'])
        image = rasterizer.render_optimized(gaussians, camera)
        
        images.append(image)
        titles.append(config['title'])
    
    return images, titles


def demo_simple_scene():
    """演示简单场景"""
    
    print("创建简单场景...")
    gaussians = create_simple_scene()
    print(f"场景包含 {len(gaussians)} 个高斯")
    
    # 渲染
    camera = GaussianCamera(width=512, height=512, fov=60)
    rasterizer = GaussianRasterizer(512, 512)
    
    import time
    start = time.time()
    image = rasterizer.render_optimized(gaussians, camera)
    render_time = time.time() - start
    
    print(f"渲染时间: {render_time:.3f}s")
    
    # 显示
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f'简单场景渲染\n{len(gaussians)} 个高斯, 渲染时间: {render_time:.2f}s', 
             fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('simple_scene_render.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'simple_scene_render.png'")
    plt.show()


def demo_colorful_scene():
    """演示彩色场景"""
    
    print("\n创建彩色场景...")
    gaussians = create_colorful_scene()
    print(f"场景包含 {len(gaussians)} 个高斯")
    
    camera = GaussianCamera(width=512, height=512, fov=60)
    rasterizer = GaussianRasterizer(512, 512)
    image = rasterizer.render_optimized(gaussians, camera)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f'彩色场景渲染\n{len(gaussians)} 个彩色高斯', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('colorful_scene_render.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'colorful_scene_render.png'")
    plt.show()


def demo_layered_scene():
    """演示分层场景（深度测试）"""
    
    print("\n创建分层场景...")
    gaussians = create_layered_scene()
    
    camera = GaussianCamera(width=512, height=512, fov=60)
    rasterizer = GaussianRasterizer(512, 512)
    image = rasterizer.render_optimized(gaussians, camera)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title('分层场景渲染\n三层高斯：红（远）、绿（中）、蓝（近）', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('layered_scene_render.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'layered_scene_render.png'")
    plt.show()


def demo_resolution_comparison():
    """演示不同分辨率的渲染"""
    
    print("\n分辨率对比...")
    gaussians = create_simple_scene()
    
    resolutions = [
        {'width': 128, 'height': 128, 'title': '128x128'},
        {'width': 256, 'height': 256, 'title': '256x256'},
        {'width': 512, 'height': 512, 'title': '512x512'},
    ]
    
    images = []
    titles = []
    times = []
    
    for res in resolutions:
        camera = GaussianCamera(width=res['width'], height=res['height'], fov=60)
        rasterizer = GaussianRasterizer(res['width'], res['height'])
        
        import time
        start = time.time()
        image = rasterizer.render_optimized(gaussians, camera)
        render_time = time.time() - start
        
        images.append(image)
        titles.append(f"{res['title']}\n{render_time:.3f}s")
        times.append(render_time)
    
    # 显示对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('不同分辨率渲染对比', fontsize=16)
    
    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('resolution_comparison.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'resolution_comparison.png'")
    plt.show()


def demo_complete_pipeline():
    """演示完整的渲染管线"""
    
    print("\n=== 完整渲染管线演示 ===\n")
    
    # 1. 场景创建
    print("步骤 1: 创建场景...")
    gaussians = create_simple_scene()
    print(f"  - 创建了 {len(gaussians)} 个高斯")
    
    # 2. 相机设置
    print("\n步骤 2: 设置相机...")
    camera = GaussianCamera(width=512, height=512, fov=60)
    print(f"  - 分辨率: {camera.width}x{camera.height}")
    print(f"  - 视场角: {camera.fov}°")
    print(f"  - 焦距: {camera.focal_length:.1f} 像素")
    
    # 3. 光栅化器
    print("\n步骤 3: 初始化光栅化器...")
    rasterizer = GaussianRasterizer(512, 512)
    
    # 4. 渲染
    print("\n步骤 4: 渲染...")
    import time
    start = time.time()
    image = rasterizer.render_optimized(gaussians, camera)
    render_time = time.time() - start
    print(f"  - 渲染完成！耗时: {render_time:.3f}s")
    
    # 5. 显示结果
    print("\n步骤 5: 显示结果...")
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    plt.title(f'完整渲染管线\n' + 
             f'场景: {len(gaussians)} 高斯 | ' + 
             f'分辨率: {camera.width}x{camera.height} | ' +
             f'渲染时间: {render_time:.3f}s',
             fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('complete_pipeline.png', dpi=150, bbox_inches='tight')
    print("图像已保存为 'complete_pipeline.png'")
    plt.show()
    
    print("\n=== 管线演示完成 ===")


if __name__ == '__main__':
    print("=" * 70)
    print("3D Gaussian Splatting - 简单场景渲染示例")
    print("=" * 70)
    
    # 运行所有演示
    demo_simple_scene()
    demo_colorful_scene()
    demo_layered_scene()
    demo_resolution_comparison()
    demo_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
    
    print("\n总结：")
    print("  ✓ 学习了如何创建 3D 高斯场景")
    print("  ✓ 理解了相机参数的设置")
    print("  ✓ 掌握了光栅化渲染流程")
    print("  ✓ 观察了不同参数的影响")
    
    print("\n下一步：")
    print("  - 尝试修改场景参数")
    print("  - 实验不同的高斯配置")
    print("  - 优化渲染性能")
    print("  - 阅读官方 3DGS 论文和实现")
