from setuptools import setup, find_packages

setup(
    name="oxrl",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "numpy",
        "transformers>=4.34",
        "datasets",
        "deepspeed>=0.12",
        "vllm>=0.4",
        "ray>=2.9",
        "pydantic>=2.0",
        "pyyaml",
        "tqdm",
        "peft",
        "bitsandbytes",
        "soundfile",
    ],
    author="oxRL Team",
    description="A lightweight post-training framework for LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/warlockee/oxRL",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
