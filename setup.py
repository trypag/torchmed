import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
description = "A compagnon library for deep learning on medical imaging"

# Read in requirements
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setuptools.setup(
    name="torchmed",
    version="0.0.1a",
    author="Pierre-Antoine Ganaye",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trypag/pytorch-med",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=requirements,
    license='GNU GPLv3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX",
    ],
)
