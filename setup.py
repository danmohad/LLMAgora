"""Packaging metadata for editable installs."""

from setuptools import find_packages, setup

setup(
    name="llm-agora",
    version="0.1.0",
    description="Lightweight arena for LLM agents",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["httpx", "python-dotenv"],
    package_data={"agora": []},
    entry_points={
        "console_scripts": [
            "agora=agora.cli:main",
        ]
    },
    python_requires=">=3.12",
)
