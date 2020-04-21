from setuptools import setup, find_packages

requirements = ["numpy>=1.18", "torch>=1.4.0", "mlcalcdriver>=1.0.0"]

setup(
    name="utils",
    author="Olivier Malenfant-Thuot",
    author_email="malenfantthuotolivier@gmail.com",
    description="Utilities for the scripts_raman repo.",
    packages=["utils"],
    install_requires=requirements,
    classifiers=["Programming Language :: Python :: 3.7"],
)
