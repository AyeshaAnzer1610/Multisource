"""Setup script for NeuroFormer package."""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuroformer",
    version="1.0.0",
    author="Ayesha Anzer, Abdul Bais",
    author_email="aag833@uregina.ca, abdul.bais@uregina.ca",
    description="Transformer-Based Multimodal Integration for Mental Health Diagnosis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuroformer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.1",
            "pytest-cov>=4.0.0",
            "black>=23.1.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.21.1",
            "ipywidgets>=8.0.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuroformer-train=scripts.train:main",
            "neuroformer-eval=scripts.evaluate:main",
            "neuroformer-preprocess=scripts.preprocess_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "neuroformer": ["configs/*.yaml"],
    },
    zip_safe=False,
)
