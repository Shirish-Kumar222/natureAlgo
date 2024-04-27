from setuptools import setup, find_packages

from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="natureAlgo",
    version="0.1.0",
    description="Nature Based Algorithm Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://natureAlgo.readthedocs.io/",
    author="OpenAI's ChatGPT, Shirish Kumar",
    author_email="shirishk222@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent"
    ],
    packages=["natureAlgo"],
    include_package_data=True,
    install_requires=[]
)