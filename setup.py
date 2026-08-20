import sys

import numpy
from setuptools import Extension, setup

if sys.platform == "win32":
    compile_args = ["/O2", "/DNDEBUG", "/std:c++17"]
else:
    compile_args = ["-O3", "-DNDEBUG", "-std=c++17"]

ext_modules = [
    Extension(
        "fastlabelrle._core",
        ["src/fastlabelrle/_core.cpp"],
        include_dirs=[numpy.get_include()],
        language="c++",
        extra_compile_args=compile_args,
    )
]

setup(
    package_dir={"": "src"},
    packages=["fastlabelrle"],
    package_data={"fastlabelrle": ["py.typed"]},
    ext_modules=ext_modules,
    zip_safe=False,
)
