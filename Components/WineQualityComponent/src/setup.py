__author__ = "lalitdutt parsai"
from setuptools import setup,find_packages
packages = ['src/WineQualityClient']
setup(
    name="WineQualityClient",
    version="0.0.1",
    author="lalitdutt parsai",
    author_email="lalitdutt.parsai@clustr.co.in",
    description="WineQuality Prediction ",
    long_description="WineQualityClient...",
    long_description_content_type="text/markdown",
    packages=packages,
    install_requires=[
        "scikit-learn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

