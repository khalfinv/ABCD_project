import setuptools


setuptools.setup(
    name="networks_correlations",
    version = "0.2",
    author="Victoria Khalfin Fekson",
    author_email="skhalfin@campus.technion.ac.il",
    packages=['networks_correlations', 'networks_correlations.common_statistics','networks_correlations.statistics'],
    install_requires=['nilearn', 'numpy', 'matplotlib','seaborn','openpyxl'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
