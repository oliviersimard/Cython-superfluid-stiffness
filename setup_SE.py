from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[
    Extension("SEvec_fig_util",
              sources=["SEvec_fig_util.pyx"],
              libraries=["m"]
              )
    ]


setup(
    name = "figures",
    ext_modules = cythonize(ext_modules)
    )
