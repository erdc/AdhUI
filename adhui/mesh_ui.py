import os
import logging
import numpy as np
import pandas as pd
import geopandas
import param
import colorcet as cc

from earthsim.annotators import PolyAndPointAnnotator
import xmsmesh
import xmsinterp
import cartopy.crs as ccrs
import holoviews as hv
import geoviews as gv
import datashader as ds
from holoviews import Table, Image, Curve
from holoviews.operation.datashader import datashade, rasterize
from holoviews import opts
import panel as pn

from adhmodel.mesh import AdhMesh, check_polygon, xmsmesh_to_dataframe
from adhmodel.adh_model import AdhModel
from genesis.util import Projection
from genesis.ui_util.map_display import ColormapOpts, DisplayRangeOpts
from genesis.ui_util.interface import StatusBar

log = logging.getLogger('es_workflows')

ROOTDIR = os.path.dirname(os.path.dirname(__file__))


def geodframe_to_geoviews(obj, geo_type, crs):
    """Converts the geometry in a GeoPandas GeoDataFrame into a list formatted for Geoviews
        only accepts one data type per object

        Parameters
        ----------
        obj - geopandas geodataframe with geometry column of shapely objects
        geo_type - string - type of geometry to extract from gdf 'point' or 'polygon'.
            Only one geometry type can be extracted from a gdf
        crs - cartopy.crs projection
        """

    if geo_type is 'point':
        data = [item.coords for item in obj.geometry]
        element = gv.Points(data, crs=crs)
        for col in obj.columns:
            if col is not 'geometry':
                element = element.add_dimension(col, 0, obj[col], True)

    elif geo_type is 'polygon':
        data = [item.coords for item in obj.geometry.boundary]
        element = gv.Polygons(data, crs=crs)
        for col in obj.columns:
            if col is not 'geometry':
                element = element.add_dimension(col, 0, obj[col], True)
    else:
        raise IOError('Shapely geometry type {} not supported'.format(obj.geometryType()))

    return element


class ConceptualModelEditor(param.Parameterized):
    adh_mod = param.ClassSelector(default=AdhModel(path_type=gv.Polygons, poly_columns=['node_spacing'],
                                                   point_columns=['node_spacing']), class_=AdhModel)

    polys = param.ClassSelector(class_=hv.Path)

    def __init__(self, polys, **params):
        super(ConceptualModelEditor, self).__init__(polys=polys, **params)

        self.adh_mod = AdhModel(path_type=gv.Polygons, polys=polys,
                                point_columns=['node_spacing'], poly_columns=['node_spacing'],
                                crs=polys.crs)
        self.adh_mod.projection.set_crs(polys.crs)
        self.adh_mod.wmts.source = gv.tile_sources.EsriImagery

        self.adh_mod.height = 600
        self.adh_mod.width = 600

    @param.output(adh_mod=AdhModel)
    def output(self):
        return self.adh_mod

    def panel(self):
        return self.adh_mod.panel()


class CreateMesh(param.Parameterized):
    """
    We need a UUID for each polygon created (https://github.com/pyviz/EarthSim/issues/292)
    and then we need to apply attirbutes to the UUID.
    This is an issue right now since polygons are currently not being deleted properly. So they
    will always be in poly_stream (either as empty list or a single node - I cant remember)
    """

    create = param.Action(lambda self: self.param.trigger('create'), label='Generate Mesh', precedence=0.2)
    status_bar = param.ClassSelector(default=StatusBar(), class_=StatusBar)

    # import data
    point_file = param.String(default=os.path.join(ROOTDIR, 'tests', 'test_files', 'vicksburg_pts.geojson'), label='Point file (*.geojson)', precedence=3)
    poly_file = param.String(default=os.path.join(ROOTDIR, 'tests', 'test_files', 'vicksburg_polys.geojson'), label='Polygon file (*geojson)', precedence=4)
    import_projection = param.ClassSelector(default=Projection(), class_=Projection, precedence=5)
    load_data = param.Action(lambda self: self.param.trigger('load_data'), label='Load', precedence=6)

    map_height = param.Number(default=600, bounds=(0, None), precedence=-1)
    map_width = param.Number(default=600, bounds=(0, None), precedence=-1)

    default_point_spacing = param.Number(default=1000, bounds=(0, None), precedence=-1)
    default_poly_spacing = param.Number(default=1000, bounds=(0, None), precedence=-1)

    adh_mod = param.ClassSelector(default=AdhModel(path_type=gv.Polygons, poly_columns=['node_spacing'],
                                                   point_columns=['node_spacing'], crs=ccrs.GOOGLE_MERCATOR),
                                  class_=AdhModel)

    def __init__(self, **params):
        super(CreateMesh, self).__init__(**params)

        # self.map.opts.get('style') # merged but not released. (https://github.com/pyviz/holoviews/pull/3440)
        self.mesh_io = None
        self.adh_mod.wmts.source = gv.tile_sources.EsriImagery
        self.import_projection.set_crs(ccrs.GOOGLE_MERCATOR)

    @param.depends('load_data', watch=True)
    def _load_data(self):
        polys = geopandas.read_file(self.poly_file)
        pts = geopandas.read_file(self.point_file)

        self.adh_mod.points = geodframe_to_geoviews(pts, 'point', crs=self.import_projection.get_crs())
        self.adh_mod.polys = geodframe_to_geoviews(polys, 'polygon', crs=self.import_projection.get_crs())

    @param.depends('create', watch=True)
    def _create(self):
        # print('create clicked')
        # todo modify to always reproject to mercator before creating the mesh
        self.status_bar.busy()

        refine_points = []

        if self.adh_mod.point_stream.data and 'node_spacing' in self.adh_mod.point_stream.element.columns():
            # for item in zip(element['Longitude'], element['Latitude'], element['node_spacing']):
            for idx in range(len(self.adh_mod.point_stream.data['Longitude'])):

                refine_points.append(
                    xmsmesh.meshing.RefinePoint(create_mesh_point=True,
                                                point=(self.adh_mod.point_stream.element['Longitude'][idx],
                                                       self.adh_mod.point_stream.element['Latitude'][idx],
                                                       0),
                                                size=float(self.adh_mod.point_stream.element['node_spacing'])))

        # ==============
        input_polygon = []
        if self.adh_mod.poly_stream.data:
            for idx in range(len(self.adh_mod.poly_stream.data['xs'])):
                poly_arr = np.fromiter(zip(self.adh_mod.poly_stream.data['xs'][idx],
                                           self.adh_mod.poly_stream.data['ys'][idx],
                                           [0]*len(self.adh_mod.poly_stream.data['ys'][idx])),
                                       dtype=('float,float,float'))

                # if no spacing value is given, use only the existing polygon vertices
                # if self.adh_mod.poly_stream.data['node_spacing'][idx] == self.adh_mod.default_value:
                # if self.adh_mod.poly_stream.data['node_spacing'][idx] == '':
                #     input_polygon.append(xmsmesh.meshing.PolyInput(outside_polygon=poly_arr))

                # redistribute vertices based on spacing from user
                # else:
                # instantiate the redistribution class
                rdp = xmsmesh.meshing.PolyRedistributePts()
                # set the node distance

                node_spacing = float(self.adh_mod.poly_stream.element.dimension_values('node_spacing', expanded=False)[idx])
                # print(node_spacing, type(node_spacing))
                # rdp.set_constant_size_func(float(self.adh_mod.poly_stream.data['node_spacing'][idx]))  # create_constant_size_function
                rdp.set_constant_size_func(node_spacing)
                # run the redistribution function
                outdata = rdp.redistribute(poly_arr)
                # AML CHANGE redistribute with the closed loop and then remove the duplicates
                # and reverse the order because the mesher expects clockwise polygons
                # ensure start and end node are different
                # if len(poly_arr) != 0:  # todo - waiting on fix for moving poly edges that leave a single node poly behind
                #     outdata = check_polygon(outdata)[::-1] # todo pretty sure this is fixed so I'm commenting it out
                # convert the polygon to an 'input polygon'
                input_polygon.append(xmsmesh.meshing.PolyInput(outside_polygon=outdata))

        # add the input polygons as polygons to the mesher class
        self.mesh_io = xmsmesh.meshing.MultiPolyMesherIo(poly_inputs=input_polygon, refine_points=refine_points)

        # Generate Mesh
        succeded, errors = xmsmesh.meshing.mesh_utils.generate_mesh(mesh_io=self.mesh_io)
        if succeded:
            self.status_bar.set_msg('Meshing was successful')
        else:
            self.status_bar.set_msg('Meshing errors found: {}'.format(errors))

        # poly_stream and point_stream are always mercator because they come from bokeh
        proj = Projection(crs_label='Mercator')
        self.adh_mod.mesh = AdhMesh(projection=proj)
        self.adh_mod.mesh.verts, self.adh_mod.mesh.tris = xmsmesh_to_dataframe(self.mesh_io.points, self.mesh_io.cells)
        self.adh_mod.mesh.reproject_points()
        self.adh_mod.mesh.tri_mesh = gv.TriMesh((self.adh_mod.mesh.tris[['v0', 'v1', 'v2']],
                                                 self.adh_mod.mesh.mesh_points)).opts(opts.TriMesh(edge_cmap='yellow', edge_color='yellow'))

    # @param.depends('adh_mod.mesh', 'adh_mod.viewable_points', 'adh_mod.viewable_polys', 'load_data', 'adh_mod.wmts.source', watch=True)
    @param.depends('create', 'load_data', watch=True)
    def view_map(self):
        map = hv.DynamicMap(self.adh_mod.wmts.view) * self.adh_mod.polys * self.adh_mod.points
        if not self.adh_mod.mesh.verts.empty:
            # map *= self.adh_mod.mesh.view_elements(line_color='black')
            map *= self.adh_mod.mesh.view_mesh(line_color='blue')

        return map.opts(width=self.map_width, height=self.map_height)

    @param.depends('create', 'load_data', watch=True)
    def _data_tab(self):
        return pn.Tabs(('Polygons', self.adh_mod.poly_table), ('Points', self.adh_mod.point_table), name='View Data')

    @param.output(adh_mod=AdhModel)
    def output(self):
        return self.adh_mod

    # @param.depends('load_data')
    def panel(self):
        map_panel = pn.Column(self.view_map, pn.panel(self.param, parameters=['create'], show_name=False))
        # display_tab = pn.Column(
        #                         pn.panel(self.adh_mod.wmts.param, parameters=['source'], expand_button=False),
        #                         pn.pane.Markdown('Visibility', style={'font-family':'Arial'}),
        #                         pn.panel(self.adh_mod.param, parameters=['viewable_points'], show_name=False),
        #                         self.adh_mod.param.viewable_polys
        #                         )

        display_tab = pn.Column(
            pn.panel(self.adh_mod.wmts.param, parameters=['source'], expand_button=False, show_name=False)
        )

        # data_tab = pn.Tabs(('Polygons', self.adh_mod.poly_table), ('Points', self.adh_mod.point_table), name='View Data')

        import_tab = pn.Column(pn.panel(self.param, parameters=['point_file', 'poly_file'], show_name=False),
                               pn.panel(self.import_projection, show_name=False),
                               pn.panel(self.param, parameters=['load_data'], show_name=False))

        logo_box = pn.Spacer()

        tool_panel = pn.Column(pn.Tabs(
                                       ('Conceptual Model', self._data_tab),
                                       ('Display', display_tab)
                                       ),
                               logo_box
                               )

        main = pn.Column(pn.Row(map_panel, tool_panel), self.status_bar.panel)

        return main


class InterpolateMesh(param.Parameterized):
    """
    Loaded bathymetry scatterset must be in columnar format with column labels being 'x', 'y', 'z'
    """
    map_width = param.Integer(default=800, precedence=-1)
    map_height = param.Integer(default=600, precedence=-1)

    filepath = param.String(
        default=os.path.join(ROOTDIR, 'tests', 'test_files', 'SanDiego', 'SanDiego_bathy.csv'))
    load_data = param.Action(lambda self: self.param.trigger('load_data'), label='Load', precedence=1)

    scatter = param.DataFrame(default=pd.DataFrame(), precedence=-1)
    scatter_projection = param.ClassSelector(default=Projection(), class_=Projection)

    scatter_toggle = param.Boolean(default=False, doc='Toggle the visibility of the scatterset',
                                   label='View scatter', precedence=24)
    interp_button = param.Action(lambda self: self.param.trigger('interp_button'), label='Interpolate', precedence=25)

    interpolation_option = param.ObjectSelector(default='idw', objects=['linear', 'natural_neighbor', 'idw'],
                                                precedence=1)
    nodal_function = param.ObjectSelector(default='constant', objects=['constant', 'gradient_plane', 'quadratic'],
                                          precedence=1.1)
    truncation = param.Boolean(default=False, precedence=2)

    truncation_range = param.Range(default=(-10, 100), bounds=(None, None), softbounds=(-40, 150), label='',
                                   precedence=-1)

    adh_mod = param.ClassSelector(class_=AdhModel)

    cmap_opts = param.ClassSelector(default=ColormapOpts(), class_=ColormapOpts)
    display_range = param.ClassSelector(default=DisplayRangeOpts(), class_=DisplayRangeOpts)

    def __init__(self, adh_mod, **params):
        super(InterpolateMesh, self).__init__(adh_mod=adh_mod, **params)
        # set defaults for initialized example
        self.display_range.param.color_range.bounds = (10, 90)
        self.display_range.color_range = (10, 90)
        self.cmap_opts.colormap = cc.rainbow
        self.scatter_projection.set_crs(ccrs.GOOGLE_MERCATOR)
        self.adh_mod.wmts.source = gv.tile_sources.EsriImagery

        # print(self.projection.param.UTM_zone_hemi.constant, self.projection.crs_label)
        self.opts = (
                     opts.Curve(height=self.map_height, width=self.map_width, xaxis=None, line_width=1.50, color='red',
                                tools=['hover']),
                     opts.Path(height=self.map_height, width=self.map_width, line_width=3, color='black'),
                     opts.Image(height=self.map_height, width=self.map_width, cmap=self.cmap_opts.param.colormap,
                                clim=self.display_range.param.color_range, colorbar=True,
                                clipping_colors={'NaN': 'transparent', 'min': 'transparent'}, axiswise=True),
                     opts.RGB(height=self.map_height, width=self.map_width),
                     opts.Points(height=self.map_height, width=self.map_width, color_index='z',
                                 cmap=self.cmap_opts.param.colormap, clim=self.display_range.param.color_range,
                                 size=10, tools=['hover'], padding=(0.1, 0.1), colorbar=True),
                     opts.TriMesh(height=self.map_height, width=self.map_width, color_index='z',
                                  cmap=self.cmap_opts.param.colormap, clim=self.display_range.param.color_range,
                                tools=['hover'], padding=(0.1, 0.1), colorbar=True),
                     opts.VLine(color='black'))
        # opts.defaults(*self.opts)

    @param.depends('load_data', watch=True)
    def _load(self):
        self.scatter = pd.read_csv(self.filepath, sep=',', names=['x', 'y', 'z'])
        self.scatter_pts = gv.Points(data=self.scatter, crs=self.scatter_projection.get_crs(),
                                     vdims=['z'], kdims=['x', 'y'])
        self.scatter_toggle = True

    @param.depends('truncation', watch=True)
    def _truncation(self):
        if self.truncation:
            self.param.truncation_range.precedence = 2.4
        else:
            self.param.truncation_range.precedence = -1

    @param.depends('interpolation_option', watch=True)
    def _update_nodal_function(self):
        if self.interpolation_option == 'linear':
            self.param.nodal_function.precedence = -1
        else:
            self.param.nodal_function.precedence = 1.1

    @param.depends('interp_button', watch=True)
    def interpolate(self):

        if self.interpolation_option != 'idw':
            interp_object = xmsinterp.interpolate.InterpLinear(pts=self.scatter.values)
            if self.interpolation_option == 'natural_neighbor':
                interp_object.interp_to_pt((0, 0))  # this will force triangle creation
                interp_object.set_use_natural_neighbor(nodal_func_type=self.nodal_function,
                                                       nd_func_pt_search_opt="nearest_pts")
        else:
            interp_object = xmsinterp.interpolate.InterpIdw(pts=self.scatter.values)
            interp_object.set_nodal_function(nodal_func_type=self.nodal_function)

        if self.truncation:
            interp_object.set_truncation_max_min(self.truncation_range[1], self.truncation_range[0])
        z = interp_object.interp_to_pts(self.adh_mod.mesh.mesh_points.data[['x', 'y']].values)

        self.adh_mod.mesh.mesh_points.data['z'] = z
        self.adh_mod.mesh.tri_mesh = gv.TriMesh((self.adh_mod.mesh.tris[['v0', 'v1', 'v2']], self.adh_mod.mesh.mesh_points)).apply.opts(*self.opts)

        self.adh_mod.mesh.elevation_toggle = True

    # @param.depends('scatter_toggle', watch=True)
    def view_scatter(self):
        # print('view_scatter')
        if self.scatter_toggle and not self.scatter.empty:

            return self.scatter_pts.apply.opts(color_index='z', cmap=self.cmap_opts.param.colormap,
                                               clim=self.display_range.param.color_range, colorbar=True,
                                               marker='o', line_color=None)
        else:
            return Curve([])

    @param.depends('adh_mod.mesh.elements_toggle', 'adh_mod.mesh.elevation_toggle',
                   'interp_button',  'scatter_toggle', watch=True)
    def view_map(self):
        # print('view_map method')

        if self.adh_mod.mesh.elevation_toggle:
            elevation = rasterize(self.adh_mod.mesh.tri_mesh, aggregator=ds.mean('z'), precompute=True).apply.opts(
                opts.Image(cmap=self.cmap_opts.colormap, clim=self.display_range.color_range,
                height=self.map_height, width=self.map_width))
        else:
            elevation = Curve([]).opts(height=self.map_height, width=self.map_width)

        # return self.adh_mod.mesh.view_bathy() * self.adh_mod.mesh.view_elements(line_color='yellow') * base_map * self.view_scatter()

        return elevation * self.adh_mod.mesh.view_elements(line_color='yellow') * hv.DynamicMap(self.adh_mod.wmts.view) * self.view_scatter()

    @param.output(adh_mod=AdhModel)
    def output(self):
        return self.adh_mod

    def panel(self):
        load_tab = pn.Column(pn.panel(self.param, parameters=['filepath'], show_name=False),
                             pn.panel(self.scatter_projection.param, expand_button=False),
                             pn.panel(self.param, parameters=['load_data'], show_name=False))

        map_pane = self.view_map

        interp_pane = pn.Column(pn.panel(self.param, parameters=['interpolation_option', 'nodal_function'],
                                         show_name=False),
                                pn.Row(
                                       pn.panel(self.param, parameters=['truncation', 'truncation_range'], show_name=False),
                                       ),
                                pn.panel(self.param, parameters=['interp_button'], show_name=False))

        display_tab = pn.Column(pn.panel(self.cmap_opts.param, parameters=['colormap'], show_name=False),
                                pn.panel(self.display_range.param, parameters=['color_range'], show_name=False),
                                pn.panel(self.param, parameters=['scatter_toggle'], show_name=False),
                                pn.panel(self.adh_mod.mesh.param, parameters=['elements_toggle', 'elevation_toggle'],
                                         show_name=False),
                                pn.panel(self.adh_mod.wmts.param, parameters=['source'], expand_button=False, show_name=False))

        tool_panel = pn.Tabs(('Load Data', load_tab),
                             ('Display', display_tab),
                             ('Interpolate', interp_pane))

        return pn.Row(map_pane, tool_panel)


def meshing_dashboard():
    """
    Wrapper for AdH Meshing Dashboard

    Defaults to view San Diego simulation from the data folder
    """
    # construct the pages
    stages = [
        ('Create Mesh', CreateMesh),
        ('Interpolate Mesh', InterpolateMesh)
    ]

    # create the pipeline
    pipeline = pn.pipeline.Pipeline(stages, debug=True)

    # modify button width (not exposed)
    pipeline.layout[0][1]._widget_box.width = 100
    pipeline.layout[0][2]._widget_box.width = 100

    # return a display of the pipeline
    return pipeline.layout
