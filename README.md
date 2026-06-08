# PDF3D CPU/GPU Calculator

`PDF3D CPU/GPU Calculator` extends the DiffPy 3D pair distribution function
workflow with two dedicated calculator classes:

- `PDF3DCPUCalculator` for portable CPU execution.
- `PDF3DGPUCalculator` for CUDA-accelerated execution on NVIDIA GPUs.

The two classes provide the same Python-facing 3D PDF interface, so user code
can switch between CPU and GPU execution with minimal changes.

## Features

- CPU and CUDA implementations of the 3D real-space PDF workflow.
- CUDA acceleration for pair accumulation, 3D q-window filtering, and grid
  postprocessing.
- cuFFT-based 3D q-window processing in the GPU calculator.
- CPU fallback path in the GPU calculator when CUDA is unavailable.
- Python bindings through `diffpy.srreal.pdfcalculator`.
- Binary export of dense 3D grids through `exportGrid3DBinary`.

## Requirements

Runtime requirements follow the DiffPy stack used by `diffpy.libdiffpy` and
`diffpy.srreal`.

For CPU usage:

- A working C++ build environment.
- DiffPy runtime dependencies.
- Python environment compatible with `diffpy.srreal`.

For GPU usage:

- NVIDIA GPU with CUDA support.
- CUDA toolkit and runtime.
- cuFFT.
- CUDA-enabled build of `diffpy.libdiffpy`.

The tested GPU platform was NVIDIA L40. The CUDA source is prepared for common
data-center GPU architectures including L40 and A800.

## Repository Structure

```text
diffpy.libdiffpy/
  SConstruct
  src/SConscript
  src/diffpy/srreal/
    PDF3DCPUCalculator.cpp
    PDF3DCPUCalculator.hpp
    PDF3DGPUCalculator.cpp
    PDF3DGPUCalculator.hpp
    PDF3DGPUKernels.cu
    PDF3DGPUKernels.hpp
    SConscript

diffpy.srreal/
  setup.py
  src/diffpy/srreal/pdfcalculator.py
  src/extensions/wrap_PDFCalculators.cpp

data/
  Si_ICSD.cif
```

## Calculators

### PDF3DCPUCalculator

`PDF3DCPUCalculator` computes the 3D real-space PDF on the host processor. It
performs pair Gaussian accumulation, 3D q-window filtering, RDF scaling,
background subtraction, and final grid postprocessing in C++.

The CPU calculator is useful for CPU-only systems, development machines, login
nodes, and environments where CUDA is not available.

It also exposes `cpuworkfactor`, an optional repeat factor for the scalar CPU
work path. The default value is `1`. Larger values repeat equivalent scalar
work and scale the result back, keeping the final grid comparable while
increasing CPU workload in a controlled way.

### PDF3DGPUCalculator

`PDF3DGPUCalculator` computes the same 3D PDF workflow with CUDA acceleration.
The GPU path moves the most parallel parts of the calculation to CUDA:

- Pair Gaussian accumulation.
- 3D q-window filtering with cuFFT.
- RDF scaling, `rho0` background subtraction, and `qdamp` postprocessing.

When CUDA is not available, the GPU calculator keeps a CPU fallback path so the
Python API remains usable on non-GPU systems.

## Python API

```python
from diffpy.srreal.pdfcalculator import (
    PDF3DCPUCalculator,
    PDF3DGPUCalculator,
)
```

### CPU Usage

```python
from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import PDF3DCPUCalculator

structure = loadStructure("data/Si_ICSD.cif")

calc = PDF3DCPUCalculator(rmax=8.0, gridstep=0.2)
calc.qmax = 12.0
calc.usecqwindow3d = True
calc.cpuworkfactor = 1

calc.eval(structure)
calc.exportGrid3DBinary("cpu_grid.bin", False, True)
```

### GPU Usage

```python
from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import PDF3DGPUCalculator

structure = loadStructure("data/Si_ICSD.cif")

calc = PDF3DGPUCalculator(rmax=8.0, gridstep=0.2)
calc.qmax = 12.0
calc.usecqwindow3d = True

calc.eval(structure)
calc.exportGrid3DBinary("gpu_grid.bin", False, True)
```

## Performance

On a 256-atom Si 4x4x4 test system with `rmax=8.0`, `gridstep=0.2`, and
`qmax=12.0`, the CUDA implementation on an NVIDIA L40 reduced total runtime
from `13.464822 s` for `PDF3DCPUCalculator` with `cpuworkfactor=16` to
`0.497957 s` for `PDF3DGPUCalculator`.

This is an overall speedup of approximately **27.04x**.

The strongest acceleration comes from the 3D q-window and grid postprocessing
stages, which are naturally suited to CUDA/cuFFT and voxel-wise parallelism.

## Integration Notes

The included files modify both the C++ and Python layers of the DiffPy stack:

- `diffpy.libdiffpy/SConstruct`
- `diffpy.libdiffpy/src/SConscript`
- `diffpy.libdiffpy/src/diffpy/srreal/SConscript`
- `diffpy.srreal/setup.py`
- `diffpy.srreal/src/extensions/wrap_PDFCalculators.cpp`
- `diffpy.srreal/src/diffpy/srreal/pdfcalculator.py`

Use these files as an integration patch against a compatible DiffPy source
tree. After rebuilding `diffpy.libdiffpy` and reinstalling `diffpy.srreal`,
the two calculator classes are available from the Python API shown above.

## Output

Both calculators can export dense 3D grids using:

```python
calc.exportGrid3DBinary(path, usefloat32=True, applypost=True)
```

Set `usefloat32=False` to write double-precision grid values. Set
`applypost=False` to export the grid before final q-window and postprocessing
steps.
