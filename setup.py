from setuptools import setup, find_packages

setup(
    name='NN_for_polynomial_integration',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scikit-learn',
        'scipy',
        'torch',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'integral_interpolator = src/integral_interpolator.py',
        ],
    },
    author='Leonardo Santiago',
    author_email='l201292@dac.unicamp.br',
    description='Using neural networks for numerical integration',
    url='https://gitlab.unicamp.br/201292/high_performance_integration',
)
