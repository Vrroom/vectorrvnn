import setuptools

with open('requirements.txt') as fd : 
    INSTALL_REQUIRES = fd.readlines()

INSTALL_REQUIRES = [r.strip() for r in INSTALL_REQUIRES]
TEST_REQUIRES = ['pytest', 'coverage', 'pytest-cov']

setuptools.setup(
    name='vectorrvnn',
    version='0.1.0',
    packages=setuptools.find_packages(),
    license='MIT License',
    long_description="""
    Library for uncovering visual similarity
    between vector elements. Includes both hand
    written and learnt heuristics.
    """,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
    },

)
