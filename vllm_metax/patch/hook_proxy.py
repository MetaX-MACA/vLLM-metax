# SPDX-License-Identifier: Apache-2.0

import sys
import types
from typing import Any, Callable, Optional

class ModuleProxy:
    """模块代理，延迟访问真实模块"""
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._real_module = None
        self._attributes = {}
    
    def __getattr__(self, name: str) -> Any:
        # 首先检查本地属性
        if name in self._attributes:
            return self._attributes[name]
        
        # 延迟加载真实模块
        if self._real_module is None:
            # 只有在访问时才导入
            if self._module_name in sys.modules:
                self._real_module = sys.modules[self._module_name]
            else:
                # 延迟导入模块
                self._real_module = __import__(self._module_name, fromlist=["*"])
        
        # 从真实模块获取属性
        try:
            return getattr(self._real_module, name)
        except AttributeError:
            raise AttributeError(
                f"Module '{self._module_name}' has no attribute '{name}'"
            )
    
    def add_attribute(self, name: str, value: Any):
        """添加或覆盖属性"""
        self._attributes[name] = value
    
    def __repr__(self) -> str:
        return f"<ModuleProxy for {self._module_name}>"

def create_proxy(module_name: str) -> ModuleProxy:
    """创建模块代理"""
    return ModuleProxy(module_name)

# 全局代理缓存
_PROXY_CACHE = {}

def get_proxy(module_name: str) -> ModuleProxy:
    """获取或创建模块代理"""
    if module_name not in _PROXY_CACHE:
        _PROXY_CACHE[module_name] = create_proxy(module_name)
    return _PROXY_CACHE[module_name]