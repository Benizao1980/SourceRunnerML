from setuptools import setup, find_packages

setup(
    name='SourceRunnerML',
    version='4.4.0',
    author='Ben Pascoe',
    description='A machine learning pipeline for microbial source attribution using (cg)MLST data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Benizao1980/SourceRunnerML',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas>=1.1.0',
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'xgboost>=1.3.0',
        'lightgbm>=3.2.0',
        'catboost>=0.26',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'scipy>=1.5.0',
        'tqdm>=4.50.0',
        'imbalanced-learn>=0.8.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'sourcerunnerml=SourceRunnerML_v4_4_0:main',
        ],
    },
)
