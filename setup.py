from setuptools import setup, find_packages

setup(
    name='trw_ta',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'statsmodels',
    ],
)
