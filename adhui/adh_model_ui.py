import os, glob, logging
import xarray as xr
import param
import numpy as np
import panel as pn
import cartopy.crs as ccrs

from adhmodel.simulation.dat_reader import get_variable_from_file_name
from genesis.util import Projection
from genesis.ui_util.map_display import DisplayRangeOpts, ColormapOpts
from adhmodel.simulation.utils import get_crs
from holoviews.streams import PointerX

import geoviews as gv
from geoviews import tile_sources as gvts
import holoviews as hv
from holoviews.operation.datashader import rasterize
from holoviews import Operation
from holoviews.core.util import basestring
from holoviews import dim, opts
from holoviews.plotting.util import process_cmap

from .simulation_ui import SimulationLocation, DefineHotstart, Attributes, LoadSimulation, BoundaryConditionsUI

from adhmodel.adh_model import AdhModel
from uit.panel_util import PbsScriptStage
from uit import Client, PbsScript, PbsJob

log = logging.getLogger('adh')

ROOTDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class LoadModel(param.Parameterized):
    """
    Parameterized class to load model information

    Formatted for Panel and Pipeline Page
    """
    load_sim_widget = param.ClassSelector(default=LoadSimulation(), class_=LoadSimulation)
    att_widget = param.ClassSelector(default=Attributes(), class_=Attributes)
    projection = param.ClassSelector(default=Projection(), class_=Projection)

    load_data = param.Action(lambda self: self.param.trigger('load_data'), label='Load Data', precedence=0.2)

    label = param.String(default='Basic Input', precedence=-1)

    def __init__(self, **params):
        super(LoadModel, self).__init__(**params)
        self.dat_files = []

        self.model = None

        self.adh_mod = AdhModel()

    # output the adh_viz object
    # @param.output(annot=AdhAnnotator)
    @param.output(adh_mod=AdhModel)
    def output(self):
        return self._load_data()

    @param.depends('load_sim_widget.load_netcdf', 'load_sim_widget.adh_root_filename',
                   'load_sim_widget.adh_directory', watch=True)
    def available_attributes(self):
        # todo this needs a 'no files found' check
        att_list = []
        # enable the projections
        self.projection.set_constant(value=False)

        # if netcdf is selected
        if self.load_sim_widget.load_netcdf:
            filename = os.path.join(self.load_sim_widget.adh_directory,
                                    self.load_sim_widget.adh_root_filename + '.nc')

            try:
                # open the xarray dataset (does not load into memory)
                ncdf = xr.open_dataset(filename)

            except FileNotFoundError:
                print('File Not Found: {}'.format(filename))
            else:
                # look for crs in the file
                if 'crs' in ncdf.attrs.keys():
                    # set the projection in the widget
                    self.projection.set_crs(get_crs(ncdf))
                    # disable the widget
                    self.projection.set_constant(value=True)

                with ncdf:
                    # enable and add all variables in the netcdf file
                    for var in ncdf.data_vars:
                        att_name = var.lower().replace(" ", "_")

                        # if the variable has results dimensions
                        if 'times' in ncdf[var].dims and 'nodes_ids' in ncdf[var].dims:
                            if att_name == 'depth-averaged_velocity':
                                att_list.append('velocity')
                            elif att_name == 'error':
                                att_list.append('error_hydro')
                            else:
                                # add to the list
                                att_list.append(att_name)
                # close the dataset
                ncdf.close()

        # otherwise read from available *.dat files
        else:
            # get the list of filenames  # todo this isn't foolproof e.g. `SanDieg` finds files
            file_names = glob.glob(os.path.join(self.load_sim_widget.adh_directory,
                                                self.load_sim_widget.adh_root_filename + '_*.dat'))

            # convert file suffixes to a list of widgets
            att_list = [self.att_widget.suffix_to_widget[get_variable_from_file_name(x)] for x in file_names]

        # # dictionary for naming inconsistencies # todo add complexity for err_con in future
        # label_to_widget = {'error': 'error_hydro',
        #                    'depth-averaged_velocity': 'velocity'}
        #
        # # loop over the inconsistencies
        # for key in label_to_widget.keys():
        #     # if this is found in the list
        #     if key in att_list:
        #         # remove the key
        #         att_list.remove(key)
        #         # add back in the corrected value
        #         att_list.append(label_to_widget[key])

        # adjust available attribute widgets based on available data
        for p in self.att_widget.param:
            # if this attribute wasn't in the list
            pobj = self.att_widget.param[p]
            if p in att_list:
                # ensure the widget is enabled
                pobj.constant = False
                # set the widget value
                setattr(self.att_widget, p, True)
            elif p != 'name':
                # ensure the widget is enabled
                pobj.constant = False
                # uncheck the attribute
                setattr(self.att_widget, p, False)
                # disable attribute
                pobj.constant = True

    @param.depends('load_data', watch=True)
    def _load_data(self):
        # if file is netcdf
        if self.load_sim_widget.load_netcdf:

            self.adh_mod.from_netcdf(
                path=self.load_sim_widget.adh_directory,
                project_name=self.load_sim_widget.adh_root_filename,
                crs=self.projection.get_crs()
            )
        # request to load data from *dat files:
        else:
            # get a list of requested suffix strings
            slist = self.att_widget.suffix_list(value=True)

            # construct list of filenames
            fnames = []
            [fnames.append(os.path.join(self.load_sim_widget.adh_directory,
                                        self.load_sim_widget.adh_root_filename + '_' + x + '.dat')) for x in slist]
            # read the requested files
            self.adh_mod.from_ascii(
                path=self.load_sim_widget.adh_directory,
                project_name=self.load_sim_widget.adh_root_filename,
                crs=self.projection.get_crs(),
                file_names=fnames
            )

        return self.adh_mod

    # visualize the page
    def panel(self):
        self.available_attributes()  # for the initial load
        return pn.Column(pn.Row(pn.Param(self.load_sim_widget, show_name=False),
                                pn.Param(self.att_widget, show_name=False),
                                pn.Param(self.projection, show_name=False), name=self.label),
                         pn.Pane(self.param, parameters=['load_data'], show_name=False))


class vectorfield_to_paths(Operation):
    arrow_heads = param.Boolean(default=True, doc="""
        Whether or not to draw arrow heads.""")

    color = param.ClassSelector(class_=(basestring, dim, int), default=3)

    magnitude = param.ClassSelector(class_=(basestring, dim, int), default=3, doc="""
        Dimension or dimension value transform that declares the magnitude
        of each vector. Magnitude is expected to be scaled between 0-1,
        by default the magnitudes are rescaled relative to the minimum
        distance between vectors, this can be disabled with the
        rescale_lengths option.""")

    pivot = param.ObjectSelector(default='mid', objects=['mid', 'tip', 'tail'],
                                 doc="""
        The point around which the arrows should pivot valid options
        include 'mid', 'tip' and 'tail'.""")

    scale = param.Number(default=1)

    def _get_lengths(self, element):
        mag_dim = self.magnitude
        if mag_dim:
            if isinstance(mag_dim, dim):
                magnitudes = mag_dim.apply(element, flat=True)
            else:
                magnitudes = element.dimension_values(mag_dim)
        else:
            magnitudes = np.ones(len(element))
        return magnitudes

    def _process(self, element, key=None):
        # Compute segments and arrowheads
        xd, yd = element.kdims
        xs = element.dimension_values(0)
        ys = element.dimension_values(1)

        rads = element.dimension_values(2)

        lens = self._get_lengths(element) / self.p.scale

        # Compute offset depending on pivot option
        xoffsets = np.cos(rads) * lens / 2.
        yoffsets = np.sin(rads) * lens / 2.
        if self.pivot == 'mid':
            nxoff, pxoff = xoffsets, xoffsets
            nyoff, pyoff = yoffsets, yoffsets
        elif self.pivot == 'tip':
            nxoff, pxoff = 0, xoffsets * 2
            nyoff, pyoff = 0, yoffsets * 2
        elif self.pivot == 'tail':
            nxoff, pxoff = xoffsets * 2, 0
            nyoff, pyoff = yoffsets * 2, 0
        x0s, x1s = (xs + nxoff, xs - pxoff)
        y0s, y1s = (ys + nyoff, ys - pyoff)

        if isinstance(self.p.color, dim):
            colors = self.p.color.apply(element, flat=True)
        elif self.p.color is not None:
            colors = element.dimension_values(self.p.color)
        else:
            colors = None

        if self.arrow_heads:
            arrow_len = (lens / 4.)
            xa1s = x0s - np.cos(rads + np.pi / 4) * arrow_len
            ya1s = y0s - np.sin(rads + np.pi / 4) * arrow_len
            xa2s = x0s - np.cos(rads - np.pi / 4) * arrow_len
            ya2s = y0s - np.sin(rads - np.pi / 4) * arrow_len
            xs = np.empty((len(x0s) * 9))
            ys = np.empty((len(x0s) * 9))
            cvals = np.empty((len(x0s) * 9))
            for i, (x0, x1, xa1, xa2, y0, y1, ya1, ya2) in enumerate(zip(x0s, x1s, xa1s, xa2s, y0s, y1s, ya1s, ya2s)):
                slc = slice(i * 9, i * 9 + 9)
                xs[slc] = [x0, x1, np.nan, x0, xa1, np.nan, x0, xa2, np.nan]
                ys[slc] = [y0, y1, np.nan, y0, ya1, np.nan, y0, ya2, np.nan]
                if colors is not None:
                    cvals[slc] = colors[i]
        else:
            xs = np.empty((len(x0s) * 3))
            ys = np.empty((len(x0s) * 3))
            for x0, x1, y0, y1 in enumerate(zip(x0s, x1s, y0s, y1s)):
                slc = slice(i * 3, i * 3 + 3)
                xs[slc] = [x0, x1, np.nan]
                ys[slc] = [y0, y1, np.nan]
                if colors is not None:
                    cvals[slc] = colors[i]

        if colors is None:
            data = [(xs, ys)]
            vdims = []
        else:
            data = [(xs, ys, cvals)]
            vdims = [str(self.p.color)]

        return element.clone(data, vdims=vdims, new_type=hv.Path, datatype=['multitabular'])


class AdhView(param.Parameterized):
    """

    """
    # projection options
    projection = param.ClassSelector(
        default=Projection(name=''),
        class_=Projection
    )
    # generic display options
    cmap_opts = param.ClassSelector(
        default=ColormapOpts(),
        class_=ColormapOpts
    )
    display_range = param.ClassSelector(
        default=DisplayRangeOpts(),
        class_=DisplayRangeOpts
    )

    adh_mod = param.ClassSelector(
        default=AdhModel(),
        class_=AdhModel,
        precedence=-1,
        doc="""AdhModel object containing all the model data"""
    )

    # todo view analysis is currently hidden until it receives more work
    view_analysis = param.Boolean(
        default=False,
        precedence=-1,
    )
    resolution = param.Number(
        default=1000,
        bounds=(10, 2000),
        softbounds=(10, 2000),
        doc="""
            Distance between samples in meters. Used for interpolation
            of the cross-section paths."""
    )

    selected_result = param.ObjectSelector(
    )

    selected_times = param.ObjectSelector()

    bc_ui = param.ClassSelector(
        class_=BoundaryConditionsUI
    )

    def __init__(self, **params):
        super(AdhView, self).__init__(**params)
        # initialize the boundary condition ui
        self.bc_ui = BoundaryConditionsUI(bound_cond=self.adh_mod.simulation.boundary_conditions)

        self.meshes = None

        if len(self.adh_mod.simulation.results.data_vars) != 0:
            # set default values
            self.param.selected_result.objects = self.adh_mod.simulation.results.data_vars
            try:
                self.selected_result = 'Depth'
            except:
                self.selected_result = set(self.adh_mod.simulation.results.data_vars).pop()

            self.param.selected_times.objects = self.adh_mod.mesh.current_sim.results[self.selected_result].times.data
            self.selected_times = set(self.adh_mod.mesh.current_sim.results[self.selected_result].times.data).pop()

        # set default colormap
        self.cmap_opts.colormap = process_cmap('rainbow_r')
        # set default wmts
        self.adh_mod.wmts.source = gvts.tile_sources['EsriImagery']

    # function for dynamic map call
    @param.depends('selected_times')
    def time_mesh_scalar(self):
        # add this time step's data as a vdim under the provided label
        data_points = self.adh_mod.mesh.mesh_points.add_dimension(
            self.selected_result,
            0,
            self.adh_mod.mesh.current_sim.results[self.selected_result].sel(times=self.selected_times).data,
            vdim=True)

        # return a trimesh with this data
        return gv.TriMesh((self.adh_mod.mesh.tris[['v0', 'v1', 'v2']], data_points), label=self.selected_result,
                          crs=ccrs.GOOGLE_MERCATOR)

    @param.depends('selected_times')
    def time_mesh_vector(self):
        vx = self.adh_mod.mesh.current_sim.results[self.selected_result].sel(times=self.selected_times).data[:, 0]
        vy = self.adh_mod.mesh.current_sim.results[self.selected_result].sel(times=self.selected_times).data[:, 1]
        xs = self.adh_mod.mesh.mesh_points.data['x']
        ys = self.adh_mod.mesh.mesh_points.data['y']
        with np.errstate(divide='ignore', invalid='ignore'):
            angle = np.arctan2(vy, vx)
        mag = np.sqrt(vx ** 2 + vy ** 2)
        return gv.VectorField((xs, ys, angle, mag), vdims=['Angle', 'Magnitude'],
                              crs=ccrs.GOOGLE_MERCATOR)

    @param.depends('selected_result')
    def create_animation(self):
        """ Method to create holoviews dynamic map meshes for vector or scalar datasets"""
        # check to make sure the mesh points have been set.
        if self.adh_mod.mesh.mesh_points.data.empty:
            self.adh_mod.mesh.reproject_points()

        if 'BEGSCL' in self.adh_mod.mesh.current_sim.results[self.selected_result].attrs.keys():
            meshes = hv.DynamicMap(self.time_mesh_scalar, label='scalar')
            # meshes = hv.DynamicMap(self.time_mesh_scalar, kdims='times', label='scalar').redim.values(
            #     times=sorted(self.adh_mod.mesh.current_sim.results['Depth'].times.values))

            return meshes

        elif 'BEGVEC' in self.adh_mod.mesh.current_sim.results[self.selected_result].attrs.keys():
            meshes = hv.DynamicMap(self.time_mesh_vector, label='vector')
            return meshes

        else:
            log.error('Data type not recognized. Must be BEGSCL or BEGVEC.')

    @property
    def tabs(self):
        # if the annotator has no mesh
        if self.adh_mod.mesh.verts.empty:
            disp_tab = pn.Column(pn.panel(self.adh_mod.wmts.param, parameters=['source'], expand_button=False, show_name=False))
        # otherwise display all mesh options
        else:
            disp_tab = pn.Column(
                pn.panel(self.adh_mod.wmts.param, parameters=['source'], expand_button=False, show_name=False),
                pn.panel(self.cmap_opts, show_name=False),
                pn.panel(self.display_range, show_name=False),
                pn.panel(self.adh_mod.mesh.param, parameters=['elements_toggle'], show_name=False),
                pn.panel(self.param, parameters=['selected_result'], show_name=False),
                pn.panel(self, parameters=['view_analysis'], show_name=False))

        return [('Display', disp_tab)]

    # what to pass out of this page (for pipeline)
    @param.output()
    def output(self):
        pass

    # how to build this page
    def panel(self):
        return pn.panel(self.run)

    @param.depends('selected_result', 'view_analysis', 'adh_mod.mesh.elements_toggle', watch=True)
    def run(self):
        self.build_map_pane()
        self.build_tool_pane()
        self.build_analysis_pane()

        return pn.Row(self.map_pane, self.analysis_pane, self.tool_pane)

    def build_tool_pane(self, logo=None):
        if logo:
            logo_box = pn.panel(logo, width=300)
        else:
            logo_box = pn.Spacer()
        # self.tool_pane = pn.Column(pn.Tabs(*self.tabs, *self.bc_ui.tabs), logo_box)
        self.tool_pane = pn.Column(pn.Tabs(*self.tabs), logo_box)
        # self.tool_pane = pn.Column(pn.Tabs(*self.tabs), self.bc_ui.panel(), logo_box)

    # @param.depends('annotator.result_label')
    # @param.depends('adh_mod.mesh.elements_toggle', watch=True) # todo I don't know why this won't work
    def build_map_pane(self):
        if self.adh_mod.mesh.verts.empty:
            self.map_pane = self.adh_mod.map_view
            self.analysis_pane = pn.Spacer()
        else:
            # create the meshes for the dynamic map
            meshes = self.create_animation()

            edgepaths_overlay = self.adh_mod.mesh.view_elements()

            # Define dynamic options
            opts = dict(
                clipping_colors={'NaN': 'transparent', 'min': 'transparent'},
                cmap=self.cmap_opts.param.colormap,
                clim=self.display_range.param.color_range,
                height=600, width=600
            )
            if meshes.label == 'scalar':

                # todo THIS IS GOING TO LOSE ALL ANNOTATIONS EVERY TIME THE MAP IS REDRAWN

                rasterized = rasterize(meshes).apply.opts(**opts)
                # Apply the colormap and color range dynamically
                dynamic = (rasterized *
                           hv.DynamicMap(self.adh_mod.wmts.view) *
                           self.adh_mod.polys *
                           self.adh_mod.points *
                           edgepaths_overlay)
            elif meshes.label == 'vector':
                # Apply the colormap and color range dynamically
                paths = vectorfield_to_paths(meshes, color='Magnitude', magnitude='Magnitude', scale=0.005)
                rasterized = rasterize(paths, aggregator='mean', precompute=True).apply.opts(**opts)
                dynamic = (rasterized *
                           hv.DynamicMap(self.adh_mod.wmts.view) *
                           self.adh_mod.polys *
                           self.adh_mod.points *
                           edgepaths_overlay)

            # time = pn.panel(self.adh_mod, parameters=['time'], widgets={'time': pn.widgets.DiscretePlayer}, show_name=False, width=400)
            # time = pn.panel(self.adh_mod.mesh.simulations[self.sim_selector], parameters=['time'], widgets={'time': pn.widgets.DiscretePlayer}, show_name=False, width=400)
            # time = pn.panel(self.adh_mod.mesh.current_sim, parameters=['time'],
            #                 widgets={'time': pn.widgets.DiscretePlayer}, show_name=False, width=400)
            time = pn.panel(self.param, parameters=['selected_times'],
                            widgets={'selected_times': pn.widgets.DiscretePlayer}, show_name=False, width=600)

            hv_panel = pn.panel(dynamic)

            if self.view_analysis:
                # todo going to have to combine the dynamic maps for meshes/vectorization/moving_points
                # self.map_pane = pn.Column(hv_panel[0] * self.annotator.moving_points, pn.Row(pn.Spacer(width=100), time))

                # create the sections of the trimesh
                self.sections = meshes.apply(self.adh_mod._sample,
                                             streams=[self.adh_mod.poly_stream])
                point_x = PointerX(source=self.sections, x=0)
                self.vline = hv.DynamicMap(hv.VLine, streams=[point_x])
                self.moving_points = self.sections.apply(self.adh_mod._pos_indicator, streams=[point_x])
                self.analysis_pane = pn.Column(
                    (self.sections * self.vline).redim.range(Depth=(0, 100), Distance=(0, 20000)).opts(framewise=True))
                self.map_pane = pn.Column(hv_panel[0], pn.Row(pn.Spacer(width=80), time))
            else:
                self.map_pane = pn.Column(hv_panel[0], pn.Row(pn.Spacer(width=80), time))
                self.analysis_pane = pn.Spacer()

    def build_analysis_pane(self):
        # if self.view_analysis:
        #     self.analysis_pane = pn.Column(self.sections * self.vline)
        pass


class AdhViewBasic(AdhView):
    def __init__(self, **params):
        super(AdhViewBasic, self).__init__(**params)

    def build_tool_pane(self, logo=None):

        self.tool_pane = pn.Tabs(*self.tabs, ('Time Series', self.bc_ui.view_time_series))


def results_dashboard():
    """
    Wrapper for AdH Results Viewer Dashboard

    Defaults to view San Diego simulation from the data folder
    """
    # construct the pages
    stages = [
        ('Load Model', LoadModel),
        ('View Results', AdhView)
    ]

    # create the pipeline
    pipeline = pn.pipeline.Pipeline(stages, debug=True)

    # modify button width (not exposed)
    pipeline.layout[0][1]._widget_box.width = 100
    pipeline.layout[0][2]._widget_box.width = 100

    # return a display of the pipeline
    return pipeline.layout


class AdhModelSubmit(PbsScriptStage):
    """Submit a single AdH model simulation to an ERDC HPC (Onyx/Topaz) using UIT+ for authorization and communication
    """
    pbs_job_name = param.String(default='pbs_job_name', precedence=0.1)
    remote_dir_name = param.String(default='remote_dir', precedence=0.3)
    uit_client = param.ClassSelector(Client)
    submit_script_filename = param.String(default='submit.pbs', precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)

        # override predefined nodes bounds
        self.param.nodes.bounds = (1, 100)

    def panel(self):
        pbs_script_options = super().view()
        input_info = pn.Column(self.param.pbs_job_name, self.param.remote_dir_name, name='Job Options')
        return pn.Column(
            pn.panel(pn.layout.Tabs(input_info, pbs_script_options, height=500)),
        )

    @param.output(job=PbsJob)
    def submit(self):
        jobs = list()
        working_dir = self.workdir + '/' + self.remote_dir_name

        job = self.generate_pbs_job(working_dir)
        job.submit(working_dir=working_dir, remote_name=self.submit_script_filename)

        return job

    def generate_pbs_job(self, working_dir, job_num=0):
        """
        It would be nice to have the bounds for the number of nodes, `HPCSubmitScript.nodes`
        dynamically change with the `queue`. It would also require the system so it would
        need to be added at this higher level class.
        """

        script = PbsScript(
            name=f'{self.pbs_job_name}_{job_num}',
            project_id=self.hpc_subproject,
            num_nodes=self.nodes,
            queue=self.queue,
            processes_per_node=44,
            max_time=self.wall_time,
            system=self.uit_client.system,
        )
        # highly oversimplified
        ncpus = {
            'onyx': 44,
            'topaz': 36
        }

        total_processors = self.nodes * ncpus[script.system]

        adh_rootname = 'CTR1_PWOP'
        output_file = f'{adh_rootname}_adh.out'

        if self.uit_client.system == 'onyx':
            executable_path = '$PROJECTS_HOME/AdH_SW/adh_V4.6'
            execute_string = f'aprun -n {total_processors} {executable_path} {adh_rootname} |tee {output_file}'
        elif self.uit_client.system == 'topaz':
            executable_path = '/p/home/apps/unsupported/AdH_SW/adh_v4.6'
            execute_string = f'mpiexec_mpt -n {total_processors} {executable_path} {adh_rootname} > {output_file}'
        else:
            raise RuntimeError('UIT+ is currently only available on topaz and onyx.')

        script.execution_block = f"""
cd $PBS_O_WORKDIR

mkdir -p {working_dir}
cd {working_dir}

{execute_string}

        """

        job = PbsJob(script=script, client=self.uit_client)

        return job
