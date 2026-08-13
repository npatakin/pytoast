Installation
============

The build compiles a C++/CUDA extension against your existing environment, so it
needs:

* **PyTorch 2.x** already installed.
* **GCC/G++ with C++17 support** and OpenMP.
* **A CUDA toolkit with** ``nvcc`` matching your PyTorch build (tested on 12.6+)
  — only if you want the CUDA kernels; see :ref:`cpu-only-install` below.

.. code-block:: bash

    git clone https://github.com/npatakin/pytoast
    cd pytoast
    pip install --no-build-isolation .

On a machine with a low RAM-to-core ratio the parallel compile can run out of
memory; cap the number of processes with:

.. code-block:: bash

    MAX_JOBS=8 pip install --no-build-isolation .

Check the installation with an import:

.. code-block:: python

    import toast

.. _cpu-only-install:

CPU-only install
----------------

A CUDA toolkit is **not** required. If no usable ``nvcc`` is found the build
falls back to a CPU-only extension automatically and prints which flavour it is
building; neither ``nvcc`` nor the CUDA headers are needed on that path. A C++17
compiler and a PyTorch install are still required.

.. code-block:: bash

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    TOAST_CPU_ONLY=1 pip install --no-build-isolation .   # or just let it autodetect

A CPU-only build keeps the fused multi-threaded CPU kernels for every operation.
Passing a CUDA tensor to one raises a clear error rather than crashing.

Which flavour you ended up with:

.. code-block:: python

    import toast

    toast.has_cuda_support()   # False in a CPU-only build

``TOAST_FORCE_CUDA=1`` forces the CUDA path if your toolkit lives somewhere the
autodetection misses.
