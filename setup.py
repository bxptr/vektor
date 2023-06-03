from setuptools import setup, find_packages

with open("requirements.txt") as handler:
    requirements = handler.readlines()

setup(
    name = "vektor",
    author = "Aarush Gupta",
    description = "a mini vector database implementation that intends to be educational and interpretable",
    packages = find_packages(),
    install_requires = requirements
)
