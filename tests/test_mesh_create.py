import os
import unittest
import cartopy.crs as ccrs
import pandas as pd
from adhui.mesh_ui import CreateMesh
from adhmodel import AdhModel
from pyviz_comms import Comm
import holoviews as hv
import geoviews as gv
import holoviews.plotting.bokeh
import geoviews.plotting.bokeh

hv.extension('bokeh')

ROOTDIR = os.path.dirname(os.path.dirname(__file__))


class TestMesh(unittest.TestCase):

    def test_input(self):
        # todo waiting on method to push data into streams without visualizations
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
        #     'Longitude': [-10100750],
        #     'Latitude': [3807009.333],
        #     'node_spacing': [400]})
        #
        # polys = gv.Polygons(data=[sample_poly1, sample_poly2], crs=ccrs.GOOGLE_MERCATOR)
        # points = gv.Points(data=sample_points, crs=ccrs.GOOGLE_MERCATOR)
        #
        # adh_mod = AdhModel(path_type=gv.Polygons, polys=polys, points=points,
        #                    point_columns=['node_spacing'], poly_columns=['node_spacing'],
        #                    crs=ccrs.GOOGLE_MERCATOR)
        #
        # mesh_panel = CreateMesh(adh_mod=adh_mod)
        #
        # mesh_panel.panel()
        #
        # mesh_panel._create()
        #
        # mesh = mesh_panel.adh_mod.mesh
        pass

    def test_instantiation(self):
        mesh_panel = CreateMesh()

        mesh_panel.panel()

    def test_load_data(self):
        mesh_panel = CreateMesh()

        mesh_panel.point_file = os.path.join(ROOTDIR, 'tests', 'test_files', 'vicksburg_pts.geojson')
        mesh_panel.poly_file = os.path.join(ROOTDIR, 'tests', 'test_files', 'vicksburg_polys.geojson')
        mesh_panel.import_projection.set_crs(ccrs.GOOGLE_MERCATOR)

        mesh_panel._load_data()
