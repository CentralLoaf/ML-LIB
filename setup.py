from setuptools import setup, find_packages

setup(
    name="mlib",
    version="0.1.0",
    description="A simple machine learning library",
    author="Harrison D",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib"
    ],
)