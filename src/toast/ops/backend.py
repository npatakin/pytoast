from contextvars import ContextVar
from contextlib import contextmanager

from enum import Enum

import torch

class BackendType(str, Enum):
    torch = 'torch'
    torch_compile = 'torch_compile'
    cpp = 'cpp'


_current_backend = ContextVar("current_backend", default=BackendType.cpp.value)


class Backend:
    def __init__(self, name: str):
        self.name = name
        self._ops = {}

    def __getattr__(self, item):
        if item in self._ops:
            return self._ops[item]
        else:
            raise ValueError(
                'Operation "{}" is not registered for "{}" backend'.format(
                    item, self.name
                ))

    def register_op(self, fn_name, fn):
        self._ops[fn_name] = fn


class TorchCompileBackend(Backend):
    def __getattr__(self, item):
        if item in self._ops:
            return self._ops[item]
        if item in _backends['torch']._ops:
            op = torch.compile(_backends['torch']._ops[item])
            self._ops[item] = op
            return op

        raise ValueError('Operation "{}" is not registered for "{}" backend '
                         '(should be either directly registered in '
                         '"torch_compile" or be registered '
                         'in "torch" backend)'.format(item, self.name))


_backends = {
    'torch': Backend('torch'),
    'torch_compile': TorchCompileBackend('torch_compile'),
    'cpp': Backend('cpp')
}


@contextmanager
def use_backend(name: BackendType | str):
    if not isinstance(name, str):
        name = name.value

    if name not in _backends:
        raise ValueError("Unknown backend: {}".format(name))

    token = _current_backend.set(name)
    try:
        yield
    finally:
        _current_backend.reset(token)


def set_backend(name: BackendType | str):
    if not isinstance(name, str):
        name = name.value

    if name not in _backends:
        raise ValueError("Unknown backend: {}".format(name))

    _current_backend.set(name)


def get_backend(name: str = None):
    if name is None:
        return _backends[_current_backend.get()]
    return _backends[name]



def register_op(backend: str, name: str = None):
    def decorator(fn):
        get_backend(backend).register_op(name or fn.__name__, fn)
        return fn
    return decorator


def torch_op(name: str = None):
    return register_op('torch', name)

def cpp_op(name: str = None):
    return register_op('cpp', name)

