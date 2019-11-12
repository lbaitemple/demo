from setuptools import setup

setup(
    name='deepnet',
    version='1.0.0',
    packages=['deepnet'],
    url='',
    license='',
    author='Li Bai',
    author_email='lbai@temple.edu',
    description='',
    install_requires=['torch', 'torch-vision'],
    package_data={'data', ['*.txt']}
)
