# SPDX-License-Identifier: Apache-2.0
"""
Global registry for module redirects, function/method patches, and import-time hooks.

This module provides APIs to register and retrieve different types of hooks:
- Module redirects: map original module names to replacement module names.
- Function/method/variable patches: map module names and qualified names to new callables or values.
- Execution hooks: callbacks to run immediately after a module is (re)loaded.
"""

import sys
import importlib
import types
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_MODULE_REDIRECTS: dict[str, str] = {}
_PATCH_REGISTRY: dict[str, dict[str, Any]] = {}
_VLLM_BACKUP: dict[str, dict[str, Any]] = {}
_ORIGINAL_VALUES: dict[str, dict[str, Any]] = {}

def register_vllm_backup(module: str, path: str, replacement: Any):
    _VLLM_BACKUP.setdefault(module, {})[path] = replacement

# redirect
def register_module_redirect(original_name: str, target_name: str) -> None:
    """
    Register a module redirection so that imports of `src` load `dst` instead.

    Args:
        src (str): Fully qualified original module name.
        dst (str): Fully qualified replacement module name.
    """
    _MODULE_REDIRECTS[original_name] = target_name
    logger.info(f"Registered redirect: {original_name} -> {target_name}")

    # 保存原始模块的引用
    if original_name in sys.modules:
        _ORIGINAL_VALUES[original_name] = sys.modules[original_name].__dict__.copy()
    
    # 如果模块已经存在且要求立即替换
    if original_name in sys.modules:
        replace_existing_module(original_name, target_name)


# patch
def register_patch(module: str, path: str, replacement: Any, immediate: bool = True):
    """
    注册属性补丁（变量/函数/类/方法）
    
    Args:
        module: 模块名
        path: 属性路径（如 "variable" 或 "Class.method"）
        replacement: 替换值
        immediate: 是否立即应用到已存在的模块
    """
    # 初始化模块注册表
    if module not in _PATCH_REGISTRY:
        _PATCH_REGISTRY[module] = {}
        _ORIGINAL_VALUES[module] = {}
    
    # 保存原始值
    if module in sys.modules:
        try:
            original_value = _get_nested_attr(sys.modules[module], path)
            _ORIGINAL_VALUES[module][path] = original_value
        except AttributeError:
            _ORIGINAL_VALUES[module][path] = None
    
    # 注册补丁
    _PATCH_REGISTRY[module][path] = replacement
    logger.info(f"Registered patch: {module}.{path}")
    
    # 立即应用到已存在的模块
    if immediate and module in sys.modules:
        _apply_single_patch(sys.modules[module], module, path, replacement)

def replace_existing_module(original_name: str, target_name: str):
    """立即替换已存在的模块"""
    try:
        # 导入目标模块
        target_module = importlib.import_module(target_name)
        
        # 创建代理模块保留原始模块名
        class ProxyModule(types.ModuleType):
            def __init__(self, target, name):
                super().__init__(name)
                self.__dict__.update(target.__dict__)
                self.__name__ = name
                self._target = target
            
            def __getattr__(self, name):
                # 特殊处理：访问原始引用
                if name == "__original__":
                    return _ORIGINAL_VALUES.get(original_name, {})
                
                return getattr(self._target, name)
        
        # 创建代理实例
        proxy = ProxyModule(target_module, original_name)
        
        # 替换系统模块缓存
        sys.modules[original_name] = proxy
        logger.debug(f"Immediately replaced {original_name} with {target_name}")
        
        # 应用任何已注册的补丁
        apply_patches_to_module(proxy, original_name)
            
    except ImportError as e:
        logger.error(f"Failed to replace {original_name}: {e}")


def _get_nested_attr(obj: object, path: str) -> Any:
    """获取嵌套属性"""
    parts = path.split('.')
    current = obj
    
    for part in parts:
        current = getattr(current, part)
    
    return current

def _set_nested_attr(obj: object, path: str, value: Any):
    """设置嵌套属性"""
    parts = path.split('.')
    current = obj
    
    # 遍历到目标对象的父级
    for part in parts[:-1]:
        current = getattr(current, part)
    
    # 设置最终属性
    logger.debug(f"Setting  {parts[-1]} to {value}")
    setattr(current, parts[-1], value)

def _apply_single_patch(module: types.ModuleType, module_name: str, 
                        path: str, replacement: Any):
    """应用单个补丁到模块"""
    try:
        _set_nested_attr(module, path, replacement)
        logger.debug(f"Applied patch: {module_name}.{path} -> {replacement}")
    except AttributeError as e:
        logger.error(f"Patch application failed: {e}")
        logger.error(f"{module}.{path} -> {replacement}")
        

def apply_patches_to_module(module: types.ModuleType, module_name: str):
    """应用所有补丁到指定模块"""
    if module_name in _PATCH_REGISTRY:
        for path, replacement in _PATCH_REGISTRY[module_name].items():
            _apply_single_patch(module, module_name, path, replacement)

def apply_all_patches():
    """应用所有补丁到已存在模块（实现）"""
    for module_path in list(_PATCH_REGISTRY.keys()):
        if module_path in sys.modules:
            apply_patches_to_module(sys.modules[module_path], module_path)
            