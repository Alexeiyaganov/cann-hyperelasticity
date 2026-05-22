"""
Setup configuration for CANN Hyperelasticity package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cann-hyperelasticity",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automatic discovery of hyperelastic material laws via physics-informed neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/cann-hyperelasticity",
    project_urls={
        "Bug Tracker": "https://github.com/YOUR_USERNAME/cann-hyperelasticity/issues",
        "Documentation": "https://cann-hyperelasticity.readthedocs.io",
        "Source Code": "https://github.com/YOUR_USERNAME/cann-hyperelasticity",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "fem": [
            "fenics-dolfinx>=0.10.0",
        ],
        "dev": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "cann-run=cann.cli:main",
        ],
    },
    keywords=[
        "neural networks",
        "physics-informed learning",
        "constitutive relations",
        "hyperelasticity",
        "materials science",
        "finite element method",
        "automatic differentiation",
    ],
)