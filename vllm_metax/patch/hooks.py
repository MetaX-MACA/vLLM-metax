# SPDX-License-Identifier: Apache-2.0
"""
PEP 302 MetaPathFinder + Loader that intercepts imports,
applies module redirections, function/method/variable patches,
fixes alias references, and executes import-time hooks.
"""
import sys
import importlib
import importlib.util
import types
from importlib.abc import MetaPathFinder, Loader
from vllm.logger import init_logger

logger = init_logger(__name__)

from .hook_registry import (
    _MODULE_REDIRECTS, 
    _PATCH_REGISTRY,
    apply_patches_to_module,
    apply_all_patches
)

from typing import Any, Optional, List, Callable

_REDIRECT_FINDER=None
_PATCH_FINDER=None

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


class RedirectLoader(Loader):
    """解决重定向模块名问题的加载器"""
    def __init__(self, loader, target_name):
        self.loader = loader
        self.target_name = target_name
    
    def create_module(self, spec):
        return self.loader.create_module(spec)
    
    def exec_module(self, module):
        # 临时修改模块名以通过加载器验证
        original_name = module.__name__
        module.__name__ = self.target_name
        
        try:
            # 执行目标模块的加载
            self.loader.exec_module(module)
        finally:
            # 恢复原始模块名
            module.__name__ = original_name
        
        # 确保模块规范正确
        module.__spec__.name = original_name

class RedirectFinder(MetaPathFinder):
    """
    MetaPathFinder that intercepts imports for:
    - Full module redirections (source -> target).
    - Patch-only modules (apply patches/hooks without redirect).
    Returns a ModuleSpec using RedirectLoader when a match is found.
    """
    def __init__(self):
        self._redirect_map: dict[str, str] = _MODULE_REDIRECTS

    def find_spec(self, fullname: str, path: Optional[List[str]] = None, 
                 target: Optional[types.ModuleType] = None) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname in self._redirect_map:
            target_name = self._redirect_map[fullname]
            
            # 查找目标模块的规范
            target_spec = importlib.util.find_spec(target_name)
            if not target_spec:
                return None
            
            # 创建重定向加载器
            loader = RedirectLoader(target_spec.loader, target_name)
            
            # 返回重定向后的规范
            return importlib.util.spec_from_loader(
                fullname,
                loader,
                origin=target_spec.origin,
                is_package=target_spec.submodule_search_locations is not None
            )
        return None


class PatchLoader(Loader):
    """在模块加载后应用补丁的加载器"""
    def __init__(self, original_loader: importlib.abc.Loader, module_name: str):
        self.original_loader = original_loader
        self.module_name = module_name
    
    def create_module(self, spec):
        return self.original_loader.create_module(spec)
    
    def exec_module(self, module: types.ModuleType):
        # 执行原始加载
        self.original_loader.exec_module(module)
        
        # 应用补丁
        apply_patches_to_module(module, self.module_name)

class PatchFinder(MetaPathFinder):
    """处理模块内补丁的查找器"""
    def find_spec(self, fullname: str, path: Optional[List[str]] = None, 
                 target: Optional[types.ModuleType] = None) -> Optional[importlib.machinery.ModuleSpec]:
        # 只处理有注册补丁的模块
        if fullname in _PATCH_REGISTRY:
            # 查找原始规范
            spec = importlib.util.find_spec(fullname, path)
            if spec is None:
                return None
            
            # 使用自定义加载器应用补丁
            if spec.loader is not None:
                spec.loader = PatchLoader(spec.loader, fullname)
            
            return spec
        return None

def install_hooks():
    """安装全局导入钩子"""
    global _REDIRECT_FINDER, _PATCH_FINDER
    
    # 如果钩子已安装，先卸载
    uninstall_hooks()
    
    apply_all_patches()
    _REDIRECT_FINDER = RedirectFinder()
    sys.meta_path.insert(0, _REDIRECT_FINDER)
    
    # 安装补丁查找器
    _PATCH_FINDER = PatchFinder()
    sys.meta_path.insert(1, _PATCH_FINDER)
    
    logger.info("Import hooks installed")

def uninstall_hooks():
    """卸载全局导入钩子"""
    global _REDIRECT_FINDER, _PATCH_FINDER
    
    # 移除重定向查找器
    if _REDIRECT_FINDER and _REDIRECT_FINDER in sys.meta_path:
        sys.meta_path.remove(_REDIRECT_FINDER)
    
    # 移除补丁查找器
    if _PATCH_FINDER and _PATCH_FINDER in sys.meta_path:
        sys.meta_path.remove(_PATCH_FINDER)
    
    _REDIRECT_FINDER = None
    _PATCH_FINDER = None
    logger.info("Import hooks uninstalled")
