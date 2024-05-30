from setuptools import setup, find_packages

setup(
    name='mfusampler',
    version='0.1.0',
    description='Multivariate-from-Univariate (MfU) Markov Chain Monte Carlo Sampler',
    author='Alireza S. Mahani',
    author_email='alireza.s.mahani@gmail.com',
    url='https://github.com/asmahani/MfUSampler',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)