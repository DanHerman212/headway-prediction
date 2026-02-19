import setuptools

setuptools.setup(
    name='headway-prediction-pipeline',
    version='0.1.0',
    install_requires=[
        'apache-beam[gcp]',
        'pyarrow',
        'google-cloud-aiplatform>=1.38.0',
        'google-cloud-firestore>=2.13.0',
        'google-cloud-storage>=2.9.0',
    ],
    packages=setuptools.find_packages(),
)