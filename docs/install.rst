AdhUI Installation
==================


To install this packaged from conda:
::

    conda install -c erdc -c erdc/label/dev -c pyviz/label/dev -c conda-forge -c aquaveo adhui


To install the repository from GitHub source (`AdhUI <https://github.com/erdc/AdhUI>`_):
::

    conda env create -f environment.yml -n adhui
    conda activate adhmodel
    jupyter labextension install @pyviz/jupyterlab_pyviz