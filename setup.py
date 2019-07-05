from setuptools import setup, find_packages

setup(
    name='AdhUI',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'panel',
        'pyuit',
        'holoviews',
        'geoviews',
        'xarray',
        'param',
        'numpy',
        'genesis',
        'pandas',
        'geopandas',
        'cartopy',
        'xmsinterp',
        'datashader',
        'pytest',
        'pyflakes',
        'nbsmoke',
    ],
    description="UI Building blocks for Adaptive Hydraulics Model creation and visualization",
    url="https://github.com/kcpevey/AdhUI",
    )
