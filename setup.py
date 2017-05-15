from distutils.core import setup

setup(
    name='Cukiverb',
    version='1.0.0',
    author='J. Nistal Hurle',
    author_email='j.nistalhurle@gmail.com',
    packages=['measurement'],
    scripts=['measure_ir.py'],
    url='http://pypi.python.org/pypi/Cukiverb/',
    license='LICENSE.txt',
    description='Real-time convolutional reverb for REAPER.',
    long_description=open('README.md').read(),
)
