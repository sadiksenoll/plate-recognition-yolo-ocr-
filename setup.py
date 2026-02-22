from setuptools import setup, find_packages

setup(
    name="plaka-okuma-sistemi",
    version="1.0.0",
    description="YOLO ve EasyOCR tabanlı akıllı plaka tanıma sistemi",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="[Your Name]",
    author_email="[your.email@example.com]",
    url="https://github.com/[your-username]/plaka-okuma-sistemi",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.7.0",
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "easyocr>=1.6.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "plaka-okuma=plaka_okuma:main",
        ],
    },
)
