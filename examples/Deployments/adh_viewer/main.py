import holoviews as hv
from adhui.adh_model_ui import results_dashboard

hv.extension('bokeh')

results_dashboard().servable()
