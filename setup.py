from setuptools import find_packages, setup

install_requires = [
    'numpy>=1.23.4',
    'pandas==1.5.3',
    'databricks-connect==13.0.0',
    'matplotlib==3.4.2',
    'pmdarima==2.0.3'
]

setup(
    author="Dr. Boxiang Dong",
    name="boxiang-pylib",
    version='0.1.1',
    packages=find_packages(),
    description="Boxiang's personal boxiang_pylib in Python",
    install_requires=install_requires,
    license="MIT",
    url="https://github.com/bxdong7/library",
    include_package_data=True
)