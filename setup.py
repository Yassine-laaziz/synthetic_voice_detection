from setuptools import setup, find_packages

setup(
    name="ai-voice-detector",
    version="0.1.0",
    description="Research project for detecting AI-generated voices",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pydub>=0.25.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
