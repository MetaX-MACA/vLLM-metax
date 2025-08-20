# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Callable, Any, Optional
import sys
from .hook_proxy import get_proxy

def safe_import(module_path: str, import_item: Optional[str] = None) -> Any:
    """
    安全导入模块或属性，处理可能的后续重定向
    
    使用示例:
      safe_import("vllm.model_executor")  # 导入模块
      safe_import("vllm.model_executor", "ModelRunner")  # 导入类
    """
    # 如果模块已在sys.modules中，直接使用
    if module_path in sys.modules:
        module = sys.modules[module_path]
        return getattr(module, import_item) if import_item else module
    
    # 否则使用代理系统
    proxy = get_proxy(module_path)
    if import_item:
        return getattr(proxy, import_item)
    return proxy

def deferred_import(func: Callable) -> Callable:
    """装饰器：延迟执行导入操作直到函数被调用"""
    def wrapper(*args, **kwargs):
        # 在实际调用时执行导入
        return func(*args, **kwargs)
    return wrapper