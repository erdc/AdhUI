from setuptools import setup, find_packages

setup(
    name='adhui',
    version='0.2.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'panel>=0.6.0',
        'pyuit>=0.3.0a6',
        'holoviews',
        'geoviews',
        'xarray',
        'param',
        'numpy',
        'genesis>=0.0.5',
        'pandas',
        'geopandas',
        'cartopy',
        'xmsinterp',
        'datashader',
        'pytest',
        'pyflakes',
        'nbsmoke',
        'adhmodel>=0.4.0'
    ],
    description="UI Building blocks for Adaptive Hydraulics Model creation and visualization",
    url="https://github.com/erdc/AdhUI",
    )
