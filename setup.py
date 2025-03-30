from setuptools import setup, find_packages

setup(
    name="GPT-SoVITS-Batches",
    version="0.1",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    description="A package for GPT-SoVITS with batch support",
    author="Alyx",
    url="https://github.com/Alyxhelpme/GPT-SoVITS-Batches",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)