import os
import sys
import shutil

from unittest.mock import MagicMock


sys.path.insert(0, os.path.abspath('../src'))

_torch_mock = MagicMock()
for _mod in ('torch', 'torch.nn', 'torch.nn.functional', 'torch.autograd'):
    sys.modules[_mod] = _torch_mock
sys.modules['toast._C'] = MagicMock()

def _torch_stub(name):
    return type(name, (), {'__module__': 'torch', '__qualname__': name})

for _name in ('Tensor', 'device', 'dtype', 'Size', 'Generator'):
    setattr(_torch_mock, _name, _torch_stub(_name))


class _TorchConst:
    def __init__(self, qualname):
        self._qualname = qualname
    def __repr__(self):
        return self._qualname

for _name in ('float16', 'float32', 'float64', 'int16', 'int32', 'int64', 'uint8', 'bool'):
    setattr(_torch_mock, _name, _TorchConst(f'torch.{_name}'))

import toast

project = 'pytoast'
author = 'pytoast contributors'
release = '0.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

autodoc_mock_imports = ['torch']

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
    'special-members': '__matmul__',
}
autodoc_member_order = 'bysource'
autodoc_typehints = 'signature'

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = 'furo'
html_theme_options = {}
html_static_path = ['_static']
html_css_files = ['custom.css']

# Images live in ../assets: the header logo, and the benchmark charts generated
# by tests/run_benchmarks.py.  Mirror them into _static at build time so the docs
# can reference them without keeping a second copy under version control.

_here = os.path.dirname(os.path.abspath(__file__))
_assets_dir = os.path.join(_here, os.pardir, 'assets')
_bench_dir = os.path.join(_here, '_static', 'benchmarks')
if os.path.isdir(_assets_dir):
    os.makedirs(_bench_dir, exist_ok=True)
    for _name in sorted(os.listdir(_assets_dir)):
        if _name.startswith('bench-') and _name.endswith('.png'):
            shutil.copy2(os.path.join(_assets_dir, _name),
                         os.path.join(_bench_dir, _name))

    _logo = os.path.join(_assets_dir, 'pytoast-header.svg')
    if os.path.isfile(_logo):
        shutil.copy2(_logo, os.path.join(_here, '_static', 'pytoast-header.svg'))

templates_path = []
exclude_patterns = ['_build']
