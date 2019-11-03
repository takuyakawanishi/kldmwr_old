import setuptools

with open ("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kldmwr-pkg-takuyakawanishi",
    version="0.1.1",
    author="Takuya Kawanishi",
    description="Parameter estimation methods",
    long_description=long_description,
    long_description_type="text/markdown",
    url="https://github.com/takuyakawanishi/kldmwr",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy']
    # python_requires='>=3.1',
)
