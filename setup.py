from setuptools import setup, find_packages

setup(
    name="GPT_SoVITS_Batches",
    version="0.1",
    packages=["GPT_SoVITS"],
    install_requires=open("requirements.txt").read().splitlines(),
    description="A package for GPT-SoVITS with batch support",
    author="The GPT-SoVITS dev team (not me)",
    url="https://github.com/Alyxhelpme/GPT-SoVITS-Batches",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
    ],
)