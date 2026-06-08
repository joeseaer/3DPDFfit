from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="pdf3d-cpu-gpu-calculator",
    version="0.1.0",
    description="CPU and CUDA calculators for 3D real-space PDF calculation.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="3DPDFfit contributors",
    license="BSD-3-Clause",
    python_requires=">=3.9",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pdf3d_calculators": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
