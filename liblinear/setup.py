from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

liblinear_sources = ['my_liblinear.pyx',
                     'src/linear.cpp',
                     'src/tron.cpp',
                     'src/blas/daxpy.c',
                     'src/blas/ddot.c',
                     'src/blas/dnrm2.c',
                     'src/blas/dscal.c']

liblinear_depends = ['src/linear.h',
                     'src/tron.h',
                     'src/liblinear_helper.c',
                     'src/blas/blas.h'
                     'src/blas/blasp.h']

ext_modules = [Extension("my_liblinear",
                         liblinear_sources,
                         libraries=["m"],
                         depends=liblinear_depends,
                         extra_compile_args=["-ffast-math", "-O3", "-fPIC", "-fopenmp"],
                         extra_link_args=['-lgomp'])]

setup(
    name="my_liblinear",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules)



