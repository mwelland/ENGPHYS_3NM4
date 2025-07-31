from setuptools import setup, find_packages

setup(
    name='tools3nm4',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'ipywidgets',
        'IPython',
        'pyppeteer',
        'nbconvert',
        'scipy',
        'plotly'
    ],
    author='Joel',
    description='A Jupyter widgets for solving',
    url='https://github.com/mwelland/ENGPHYS_3NM4',  
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)