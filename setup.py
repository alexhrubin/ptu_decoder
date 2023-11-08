from setuptools import setup
from Cython.Build import cythonize

setup(
    name="ptu_decoder",
    ext_modules=cythonize("src/ptu_decoder/stream_decoder.pyx", language="c++")
)
