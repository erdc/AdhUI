import panel as pn
import holoviews as hv
from earthsim.grabcut import GrabCutPanel, SelectRegionPanel
from adhui import CreateMesh, ConceptualModelEditor

hv.extension('bokeh')


stages = [
    ('Select Region', SelectRegionPanel),
    ('Grabcut', GrabCutPanel),
    ('Path Editor', ConceptualModelEditor),
    ('Mesh', CreateMesh)
]

# create the pipeline
pipeline = pn.pipeline.Pipeline(stages, debug=True)

# modify button width (not exposed)
pipeline.layout[0][1]._widget_box.width = 100
pipeline.layout[0][2]._widget_box.width = 100
    
# return a display of the pipeline
pipeline.layout.servable()

