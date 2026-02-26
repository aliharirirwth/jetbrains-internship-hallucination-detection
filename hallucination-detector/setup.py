from setuptools import setup, find_packages

setup(
    name="hallucination-detector",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn>=1.4",
        "pyyaml",
        "tqdm",
        "datasets>=2.18",
        "transformers>=4.40",
        "accelerate>=0.29",
    ],
    extras_require={
        "dev": ["pytest", "ruff"],
        "gpu": ["bitsandbytes>=0.43"],
    },
)
