from setuptools import setup, find_packages

setup(
    name="wctrs2026",
    version="0.1.0",
    description="Time series forecasting for U.S. agricultural trade with 11 model architectures",
    author="WCTRS2026",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "transformers>=4.20.0",
        "einops>=0.4.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": ["jupyter>=1.0.0", "ipywidgets>=7.6.0"],
    },
)
