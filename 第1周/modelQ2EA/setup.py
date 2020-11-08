from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="PSpincalc",
    version="0.2.7",
    author="Jose Gama",
    author_email="josephgama@yahoo.com",
    maintainer="tuxcell",
    maintainer_email="josephgama@yahoo.com",
    description="Package for converting between attitude representations (DCM, Euler angles, Quaternions, and Euler vectors)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuxcell/PSpincalc",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)

