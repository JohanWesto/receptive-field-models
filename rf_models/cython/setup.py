from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[Extension("rf_cython",
             ["rf_cython.pyx"],
             libraries=["m"],
             extra_compile_args =["-ffast-math", "-fopenmp"],
             extra_link_args=["-fopenmp"])]

setup(
  name = "rf_cython",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)