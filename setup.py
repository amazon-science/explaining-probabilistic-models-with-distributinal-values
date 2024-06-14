from setuptools import setup
setup(
    name='DistributionalValues',
    version='0.1',
    packages=['dvals', 'dvals.examples'],
    url='',
    license='Apache-2.0',
    author='Luca Franceschi',
    author_email='franuluc@amazon.de',
    description='Distributional values for XAI',
    install_requires=open('requirements.txt').read().splitlines()
)
