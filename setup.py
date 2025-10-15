from setuptools import setup, find_packages
import pathlib

# 获取项目当前目录
here = pathlib.Path(__file__).parent.resolve()

# 读取项目描述（从README.md）
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # 项目名称
    name='ml_builder',
    # 版本号（遵循语义化版本规范）
    version='0.1.0',
    # 项目描述
    description='机器学习模型构建脚手架',
    # 详细描述（从README文件读取）
    long_description=long_description,
    long_description_content_type='text/markdown',
    # 项目URL（如果有GitHub等仓库）
    url='https://github.com/yourusername/ml_builder',  # 请替换为实际仓库URL
    # 作者信息
    author='Your Name',  # 请替换为你的名字
    author_email='your.email@example.com',  # 请替换为你的邮箱
    # 项目分类
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  # 假设使用MIT许可证
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    # 项目关键词
    keywords='machine-learning, transformer, neural-network, scaffold',
    # 包发现配置
    packages=find_packages(include=['ml_builder', 'ml_builder.*']),
    # Python版本要求
    python_requires='>=3.8',
    # 项目依赖
    install_requires=[
        # 'torch>=1.9.0',  # 根据transformer.py中的import torch判断
    ],
    # 可选依赖（如果有的话）
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=4.0',
            'mypy>=0.900',
        ],
        'test': [
            'pytest>=6.0',
            'coverage>=5.0',
        ],
    },
    # 包数据（如果有的话）
    package_data={
        'ml_builder': ['py.typed'],  # 可选，如果你使用类型提示
    },
    # 数据文件（如果有的话）
    data_files=[('share/info', ['README.md'])],
    # 入口点（如果有命令行工具）
    entry_points={
        # 例如：'console_scripts': ['ml_builder=ml_builder.cli:main'],
    },
    # 项目许可证
    license='MIT',
    # 项目平台
    platforms=['any'],
    # 项目维护状态
    maintainer='Your Name',  # 请替换为你的名字
    maintainer_email='your.email@example.com',  # 请替换为你的邮箱
    # 项目开发状态
    development_status='3 - Alpha',
)