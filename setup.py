import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="karhunenloeve", # Replace with your own username
    version="0.0.1",
    author="Luciano Melodia",
    author_email="luciano.melodia@fau.de",
    description="A package for TDA on time series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karhunenloeve/Ajin",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)