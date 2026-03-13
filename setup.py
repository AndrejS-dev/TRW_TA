from setuptools import setup, find_packages

setup(
    name='The Real World TA',
    version='1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'statsmodels',
        'requests',
        'ccxt',
        'scikit-learn',
        'scipy',
    ],
)
