from setuptools import find_packages
from setuptools import setup
from codecs import open
import os

current_path = os.getcwd()

with open(os.path.join(current_path, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='alphazerocode', 
    packages=find_packages(exclude=('*.pyc',)),
    version='1.10.2.0',
    license='MIT', 
    install_requires=[],
    author='M.Uchiyama',
    description='AlphaZeroCode', 
    long_description=long_description,
    long_description_content_type='text/markdown', 
    keywords='AlphaZeroCode', 

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
)
