from setuptools import setup, find_packages

setup(
    name='trw_ta',
    version='1.0.0',
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
