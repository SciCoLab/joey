import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

reqs = []
with open('requirements.txt', 'r') as requirements:
    for row in requirements:
        reqs.append(row)

setuptools.setup(
    name='joey',
    author='SciCoLab/Devito',
    description='High-performance machine learning using code generation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/scicolab/joey',
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        'Programming Language :: Python :: 3'
    ],
    use_scm_version=True,
    setup_requires=['setuptools_scm']
)
