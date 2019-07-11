#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

__author__ = 'Kiran Karra, Chace Ashcraft, Nat Kavaler, Cash Costello, Nathan Drenkow'
__email__ = 'kiran.karra@jhuapl.edu, chace.ashcraft@jhuapl.edu, nathaniel.kavaler@jhaupl.edu, ' \
            'cash.costello@jhuapl.edu, nathan.drenkow@jhuapl.edu'
__version__ = '0.1.0'


setuptools.setup(
    name='trojai',
    version=__version__,

    description='TrojAI model and dataset generation library',
    long_description=long_description,
    long_description_content_type="text/markdown",

    url = 'https://github.com/trojai/trojai',

    author=__author__,
    author_email=__email__,

    license='Apache License 2.0',

    python_requires='>=3',
    packages=['trojai', 'trojai.datagen'],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='deep-learning trojan adversarial',

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy',
                      'pandas',
                      'scikit-image',
                      'joblib',
                      'scipy',
                      'pillow',
                      'scikit-learn==0.20.3',  # choose this version to be compatible with nosetests
                      'tqdm',
                      'opencv-python',
                      ],

    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
          'test': ['nose']
    },

    scripts=['scripts/datagen/mnist.py',
             'scripts/datagen/mnist_utils.py',
             'scripts/datagen/mnist_badnets.py',
             'scripts/datagen/mnist_badnets2.py'
             ],

    zip_safe=False
)