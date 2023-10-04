from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("ptu_decoder/stream_decoder.pyx")
)
