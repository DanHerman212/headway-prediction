import setuptools

setuptools.setup(
    name='headway-prediction-pipeline',
    version='0.1.0',
    install_requires=[
        'apache-beam[gcp]',
        'pyarrow',
        # 'statistics' is part of stdlib in python 3, so no need to lsit it
    ],
    packages=setuptools.find_packages(),
)