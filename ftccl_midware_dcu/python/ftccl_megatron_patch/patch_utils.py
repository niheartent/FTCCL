import importlib
from typing import Callable


def patch_attr(module_name: str, attr_name: str, replacement):
    module = importlib.import_module(module_name)
    original = getattr(module, attr_name)
    setattr(module, attr_name, replacement)
    return original


def wrap_attr(module_name: str, attr_name: str, wrapper: Callable):
    module = importlib.import_module(module_name)
    original = getattr(module, attr_name)
    replacement = wrapper(original)
    setattr(module, attr_name, replacement)
    return original
