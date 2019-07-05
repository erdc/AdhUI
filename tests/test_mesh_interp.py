import unittest
import cartopy.crs as ccrs
import pandas as pd
from adhui.mesh_ui import CreateMesh, InterpolateMesh
import holoviews as hv
import holoviews.plotting.bokeh
import geoviews.plotting.bokeh

hv.extension('bokeh')


class TestMeshInterp(unittest.TestCase):

    def test_mesh_interp(self):
        # todo waiting on method to push data into streams without visualizations
        # projection = ccrs.GOOGLE_MERCATOR
        # sample_poly1 = dict(
        #     Longitude=[-10114986, -10123906, -10130333, -10121522, -10129889, -10122959],
        #     Latitude=[3806790, 3812413, 3807530, 3805407, 3798394, 3796693])
        # sample_poly2 = dict(
        #     Longitude=[-10095762, -10101582, -10104906],
        #     Latitude=[3804189, 3817180, 3799659])
        # sample_poly1['node_spacing'] = 1000
        # sample_poly2['node_spacing'] = 1000
        #
        # sample_points = pd.DataFrame({
        #             'Longitude': [-10100750],
        #             'Latitude': [3807009.333],
        #             'node_spacing': [400]})
        #
        # path = [sample_poly1, sample_poly2]
        #
        # mesh_panel = CreateMesh(polys=path, points=sample_points, crs=projection)
        #
        # mesh_panel.panel()  # the data won't get into the streams until rendering - this won't do that - how to test????
        #
        # mesh_panel._create()
        #
        # mesh = mesh_panel.adh_mod.mesh
        #
        # interp_panel = InterpolateMesh(adh_mod=mesh_panel.adh_mod)
        #
        # # frivolous display
        # interp_panel.panel()
        #
        # # trigger a bathy load
        # interp_panel._load()
        #
        # # run interpolation
        # interp_panel.interpolate()

        # todo go through all the options for interpolation

        # todo add various crs tests
        pass
