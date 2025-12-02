from setuptools import setup, find_packages

setup(
    name='trw_ta',
    version='0.2.2',
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
