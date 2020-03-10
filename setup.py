from setuptools import setup, find_packages

setup(
    name='pynn_object_serialisation',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/pabogdan/pynn_object_serialisation',
    license="GNU GPLv3.0",
    author='Petrut Antoniu Bogdan',
    author_email='petrut.bogdan@manchester.ac.uk',
    description='Write PyNN object to disk for later use',
    # Requirements
    dependency_links=[],

    install_requires=["numpy",
                      "scipy",
                      "pynn",
                      "keras",
                      "keras_rewiring",
                      "matplotlib",
                      "tensorflow",
                      "argparse",
                      "pillow",
                      "colorama"],
    classifiers=[
        "Development Status :: 3 - Alpha",

        "Intended Audience :: Science/Research",

        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2"
        "Programming Language :: Python :: 2.7"


        "Programming Language :: Python :: 3"
        "Programming Language :: Python :: 3.6"
        "Programming Language :: Python :: 3.6"
        "Programming Language :: Python :: 3.7"

        "Topic :: Scientific/Engineering",
    ]
)
