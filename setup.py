from distutils.core import setup

INSTALL_REQ = [
    'scikit-learn>=0.21.2',
    'numpy>=1.16.4',
    'pandas>=0.24.2',
]

setup(
    name='pulearn',
    version='0.1dev',
    url = 'https://github.com/tomastokar/pulearn_package.git',
    author = 'Tomas Tokar',
    author_email = 'tomastokar@gmail.com',
    license='',
    descriptions = 'Positive-unlabelled learning',
    long_description=open('README.md').read(), 
    python_requires = '>=3.7', 
    install_requires = INSTALL_REQ
)