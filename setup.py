from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "ptu_decoder.stream_decoder",  # IMPORTANT: Include the package prefix
        ["ptu_decoder/stream_decoder.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="ptu_decoder",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["numpy"],
    ext_modules=cythonize(extensions, language_level=3),
    package_data={
        "ptu_decoder": ["*.pyx", "*.pxd"],
    },
    zip_safe=False,  # Required for Cython extensions
)
