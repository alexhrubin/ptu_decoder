from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "ptu_decoder.ptu_decoder",  # Name of the Python extension to be built
        ["ptu_decoder/ptu_decoder.pyx", "ptu_decoder/stream_decoder.c"],  # Source files
        include_dirs=["ptu_decoder"],  # Directory of your header files
    )
]

setup(
    name='ptu_decoder',
    ext_modules=cythonize(extensions, annotate=True),
)
