from setuptools import setup, find_packages

setup(
    name="bayesian_pasa",
    version="1.0.0",
    author="Mohsen Mostafa",
    author_email="mohsen.mostafa.ai@outlook.com",
    description="Bayesian Probabilistic Adaptive Sigmoidal Activation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/BayesianPASA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
    ],
)
