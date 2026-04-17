import setuptools

with open('README.rst') as file:
        long_description = file.read()

### Crap to be able to create a binary installer for Windows
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)

setuptools.setup(
    name='spatialstats',
    packages=setuptools.find_packages(),
    package_data={'spatialstats': ['reference_data/*.txt']},
    include_package_data=True,
    description='Toolbox for spatial point statistics',
    long_description=long_description,
    author='Jack Featherstone',
    author_email='jack.featherstone@proton.me',
    url='http://www.github.com/jfeatherstone/spatialstats',
    install_requires=['scipy', 'numpy', 'matplotlib'],
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research'
    ]
)
