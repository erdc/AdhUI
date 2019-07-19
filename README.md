# AdhUI - Visualization of AdH Simulation Data  
| All | Linux | Windows | Mac |
| --------- | --------- | --------- | ------------- |
|[![Build Status](https://dev.azure.com/kimberlycpevey/ERDC/_apis/build/status/erdc.AdhUI?branchName=master)](https://dev.azure.com/kimberlycpevey/ERDC/_build/latest?definitionId=8&branchName=master)|[![Build Status](https://dev.azure.com/kimberlycpevey/ERDC/_apis/build/status/erdc.AdhUI?branchName=master&jobName=ubuntu-16.04)](https://dev.azure.com/kimberlycpevey/ERDC/_build/latest?definitionId=8&branchName=master)|[![Build Status](https://dev.azure.com/kimberlycpevey/ERDC/_apis/build/status/erdc.AdhUI?branchName=master&jobName=vs2017-win2016)](https://dev.azure.com/kimberlycpevey/ERDC/_build/latest?definitionId=8&branchName=master)|[![Build Status](https://dev.azure.com/kimberlycpevey/ERDC/_apis/build/status/erdc.AdhUI?branchName=master&jobName=macOS-10.13)](https://dev.azure.com/kimberlycpevey/ERDC/_build/latest?definitionId=8&branchName=master)
   
   
The AdhUI package holds all the visualization tools for the AdhModel object. This package primarily utilizes the PyViz visualization stack which relies heavily on Panel, Holoviews, Geoviews, and Bokeh.  
  
In this package, users will find the individual building blocks for a UI as well as fully constructed dashboards from those building blocks.  
  
### Installation
To install this packaged from conda:  
`conda install -c erdc -c erdc/label/dev -c pyviz/label/dev -c conda-forge -c aquaveo adhui`  
  
### Developer Notes
To install the repository from source:  
`conda env create -f environment.yml -n adhui`   
`conda activate adhmodel`  
`jupyter labextension install @pyviz/jupyterlab_pyviz`  
   
To opt out of a Azure Pipelines CI build, add [skip ci] or [ci skip] to the commit message. 