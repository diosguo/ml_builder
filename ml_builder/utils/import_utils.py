"""
import module by str 
"""
import importlib
import importlib.util
import os
import sys
from typing import Union

def load_class(import_path: str) -> type:
    """
    通过路径字符串动态加载类，支持两种格式：
    1. 文件路径 + 类名（如 "/path/to/module.py:MyClass"）
    2. 模块路径 + 类名（如 "my_package.module:MyClass"）

    :param import_path: 格式为 "path.to.module:ClassName" 或 "/abs/path/module.py:ClassName"
    :return: 类对象
    """
    if ":" not in import_path:
        raise ValueError("路径格式错误，应为 'path.to.module:ClassName' 或 '/path/module.py:ClassName'")

    module_path, class_name = import_path.rsplit(":", 1)
    
    # 情况1：模块路径（如 "my_package.module:MyClass"）
    if os.path.sep not in module_path and "." in module_path:
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"无法导入模块 '{module_path}': {e}")
    
    # 情况2：文件路径（如 "/path/to/module.py:MyClass"）
    else:
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"文件不存在: {module_path}")
        
        # 生成唯一模块名防止冲突
        module_name = f"dynamic_{hash(module_path)}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ImportError(f"无法从文件加载模块: {module_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, class_name):
            raise AttributeError(f"模块中未找到类 '{class_name}'")
        return getattr(module, class_name)

# 示例用法
if __name__ == "__main__":
    # 示例1：从文件路径加载
    MyClass1 = load_class("/absolute/path/to/my_module.py:MyClass")
    obj1 = MyClass1()

    # 示例2：从模块路径加载（需在PYTHONPATH中）
    MyClass2 = load_class("my_package.submodule:MyClass")
    obj2 = MyClass2()