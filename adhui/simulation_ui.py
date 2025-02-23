import os, logging
import param
import pandas as pd
import numpy as np
from adhmodel import BoundaryConditions
import panel as pn
import bokeh
from bokeh.models.widgets.tables import NumberEditor, StringEditor


log = logging.getLogger('adh')
ROOTDIR = os.path.dirname(os.path.dirname(__file__))
LOGO = os.path.join(ROOTDIR, 'data', 'es_logo.svg')


class SimulationLocation(param.Parameterized):
    adh_root_filename = param.String(default='SanDiego', precedence=1)
    adh_directory = param.String(default=os.path.join(ROOTDIR, 'tests', 'test_files', 'SanDiego'), precedence=2)


class DefineHotstart(param.Parameterized):
    define_depth = param.Boolean(False, precedence=2.1)
    initial_depth = param.Number(10.0, bounds=(0, None), precedence=2.2)
    define_wse = param.Boolean(True, precedence=2.3)
    initial_wse = param.Number(20.0, bounds=(0, None), precedence=2.4)
    define_velocity = param.Boolean(False, precedence=2.5)
    initial_velocity_x = param.Number(0.10, bounds=(0, None), precedence=2.6)
    initial_velocity_y = param.Number(0.10, bounds=(0, None), precedence=2.7)
    initial_velocity_z = param.Number(0.0, bounds=(0, None), precedence=2.8)


class Attributes(param.Parameterized):
    active_layer_distribution     = param.Boolean(default=False, precedence=-1)
    active_layer_thickness        = param.Boolean(default=False, precedence=-1)
    bedload                       = param.Boolean(default=False, precedence=-1)
    bed_layer_distribution        = param.Boolean(default=False, precedence=-1)
    bed_layer_thickness           = param.Boolean(default=False, precedence=-1)
    bed_shear                     = param.Boolean(default=False, precedence=-1)
    clay                          = param.Boolean(default=False, precedence=-1)
    cohesive_bed_properties       = param.Boolean(default=False, precedence=-1)
    generic_constituent           = param.Boolean(default=False, precedence=0.7)
    depth                         = param.Boolean(default=True,  precedence=0.1)
    bed_displacement              = param.Boolean(default=False, precedence=-1)
    error_hydro                   = param.Boolean(default=False, precedence=0.99)
    error_constituent             = param.Boolean(default=False, precedence=-1)
    salinity                      = param.Boolean(default=False, precedence=0.4)
    sediment_mass_residual        = param.Boolean(default=False, precedence=-1)
    sand                          = param.Boolean(default=False, precedence=-1)
    suspended_load                = param.Boolean(default=False, precedence=-1)
    string_flux                   = param.Boolean(default=False, precedence=-1)
    velocity                      = param.Boolean(default=True,  precedence=0.2)
    vorticity                     = param.Boolean(default=False, precedence=0.8)
    water_surface_elevation       = param.Boolean(default=False, precedence=0.3)

    def param_list(self, value=True):
        """Collect the widget values and returns a list of param string names based on
        the requested boolean value.

        Args:
            value: (value) default=True

        Returns:
            plist: (list) param attribute strings
        """
        df = pd.DataFrame(self.get_param_values(), columns=['label', 'value'])
        df = df[df.label != 'name']
        plist = df[df.value == value]['label'].tolist()  # must have `==` instead of `is`

        return plist

    @property
    def widget_to_suffix(self):
        """
        Mapping from widget param objects to the AdH file suffix.

        Returns:
            Dictionary with mapping {param_name: suffix}

        todo fill out and test AdHv4 dictionary - what is below may not be comprehensive
        """
        # # Adhv4
        # widget_to_suffix = {
        #     'active_layer_distribution': 'ald',
        #     'active_layer_thickness': 'alt',
        #     'bedload': 'bedload',
        #     'bed_layer_distribution': 'bld',
        #     'bed_layer_thickness': 'blt',
        #     'bed_shear': 'bsh',
        #     'clay': 'cla',
        #     'cohesive_bed_properties': 'cbp',
        #     'generic_constituent': 'con',
        #     'depth': 'dep',
        #     'bed_displacement': 'bed_dpl',
        #     'error_hydro': 'err_hyd',
        #     'error_constituent': 'err_con',
        #     'salinity': 'sal',
        #     'sediment_mass_residual': 'smr',
        #     'sand': 'snd',
        #     'suspended_load': 'susload',
        #     'string_flux': 'tflx',
        #     'velocity': 'ovl',
        #     'vorticity': 'vor',
        #     'water_surface_elevation': 'wse'
        #     }

        # Adhv5
        widget_to_suffix = {
            'active_layer_distribution': 'ald',
            'active_layer_thickness': 'alt',
            'bedload': 'bedload',
            'bed_layer_distribution': 'bld',
            'bed_layer_thickness': 'blt',
            'bed_shear': 'bsh',
            'clay': 'cla',
            'cohesive_bed_properties': 'cbp',
            'generic_constituent': 'con',
            'depth': 'dep',
            'bed_displacement': 'bed_dpl',
            'error_hydro': 'error',
            'error_constituent': 'error_con',
            'salinity': 'sal',
            'sediment_mass_residual': 'smr',
            'sand': 'snd',
            'suspended_load': 'susload',
            'string_flux': 'tflx',
            'velocity': 'vel',
            'vorticity': 'vor',
            'water_surface_elevation': 'wse'
        }
        # # for adhv5 sediment simulations
        # widget_to_suffix['error_hydro'] = 'error_hyd'

        return widget_to_suffix

    @property
    def suffix_to_widget(self):
        # create an inverse dictionary
        suffix_to_widget = {v: k for k, v in self.widget_to_suffix.items()}
        return suffix_to_widget

    def suffix_list(self, value=True):
        """Collect ther widget values and return a list of filename suffixes based on
        the requested boolean value.

        Args:
            value: default=True

        Returns:
            slist: (list) filename suffix strings

        """
        # get the list of bool params
        plist = self.param_list(value=value)

        # convert the params to suffixes
        slist = []
        [slist.append(self.widget_to_suffix[item]) for item in plist]

        return slist


class LoadSimulation(SimulationLocation):
    load_boundary_condition = param.Boolean(default=False, precedence=-1)

    load_mesh = param.Boolean(default=True, precedence=-1)
    mesh_header_rows = param.Integer(default=2, precedence=-1)

    load_hotstart = param.Boolean(default=True, precedence=-1)

    load_netcdf = param.Boolean(default=True, precedence=5)


def _create_hotstart(adh_model, fem_obj, hotstart_param):
    # ### Create Initial Water Depth ###
    if hotstart_param.define_depth and hotstart_param.define_wse:
        hotstart_param.define_wse = False
        log.warning('Water depth cannot be specified as both depth and wse. Ignoring the wse value')

    if hotstart_param.define_depth:
        # set initial water surface elevation (unused)
        constantWSE = None
        # set inital water depth
        constantWaterDepth = hotstart_param.initial_depth
        # make the ioh hotstart parameter
        adh_model.makeInitialDepth(fem_obj, constantWSE, constantWaterDepth)
    if hotstart_param.define_wse:
        # set initial water surface elevation
        constantWSE = hotstart_param.initial_wse
        # set inital water depth (unused)
        constantWaterDepth = None
        # make the ioh hotstart parameter
        adh_model.makeInitialDepth(fem_obj, constantWSE, constantWaterDepth)

    # ### Create Initial Velocity ###
    if hotstart_param.define_velocity:
        constantVelocity = np.array(
            (hotstart_param.initial_velocity_x, hotstart_param.initial_velocity_y, hotstart_param.initial_velocity_z))
        adh_model.makeInitialVelocity(fem_obj, constantVelocity)

    return adh_model


def dataframe_to_bokeh_table(df):
    def mk_edit(vals):
        if vals.dtype is np.dtype(int):
            return NumberEditor()
        elif vals.dtype is np.dtype(str):
            return StringEditor()
        else:
            return StringEditor()

    source = bokeh.models.ColumnDataSource(df.to_dict(orient='list'))
    columns = [bokeh.models.TableColumn(field=name, title=name.replace('_', ' ').title(), editor=mk_edit(vals))
               for
               name, vals in df.items()]
    table = bokeh.models.DataTable(source=source, columns=columns, editable=True)

    return table


class BoundaryConditionsUI(param.Parameterized):
    bound_cond = param.ClassSelector(default=BoundaryConditions(), class_=BoundaryConditions)

    material_select = param.ObjectSelector(label='Material Number')
    transport_select = param.ObjectSelector(label='Select Constituent')
    time_series_select = param.ObjectSelector(label='Time Series')

    def __init__(self, bc_options=None, **params):

        super(BoundaryConditionsUI, self).__init__(**params)
        self.bc_options = bc_options
        # parse bc visualization options #todo

        # set the available materials and current selected material number
        self.param.material_select.objects = self.bound_cond.material_properties.keys()
        self.material_select = 1
        # set the current selected material
        self.selected_material = self.bound_cond.material_properties[self.material_select]

        # populate the transport objects if any are included in the boundary conditions
        if self.bound_cond.operation_parameters.transport > 0:
            # set the available constituents and the current selected transport
            self.param.transport_select.objects = self.bound_cond.material_properties[
                self.material_select].transport_properties.keys()
            self.transport_select = 1
            # set the current selected transport
            self.selected_constituent = self.bound_cond.material_properties[self.material_select].transport_properties[
                self.transport_select]

        # populate the time series objects if any are included in the boundary conditions
        if self.bound_cond.time_series:
            # set the available materials and current selected material number
            self.param.time_series_select.objects = self.bound_cond.time_series.keys()
            self.time_series_select = 1
            # set the current selected material
            self.selected_time_series = self.bound_cond.time_series[self.time_series_select]

    @param.depends('material_select', watch=True)
    def _material_properties(self):
        """updated the selected material based on the dropdown menu"""
        self.selected_material = self.bound_cond.material_properties[self.material_select]
        return pn.panel(self.selected_material, show_name=False)

    @param.depends('transport_select', 'material_select')
    def _transport_properties(self):
        """update the selected constituent based on the dropdown menus"""
        if self.bound_cond.operation_parameters.transport > 0:
            # set the current selected transport
            self.selected_constituent = self.bound_cond.material_properties[self.material_select].transport_properties[
                self.transport_select]
            return pn.panel(self.selected_material.transport_properties[self.transport_select], show_name=False)
        else:
            return pn.Spacer(width=0)

    @param.depends('time_series_select', watch=True)
    def time_series_table(self):
        """update the selected time series based on the dropdown menus"""
        if self.bound_cond.time_series:
            # set the current selected material
            self.selected_time_series = self.bound_cond.time_series[self.time_series_select]
            # convert dataframe to a bokeh table
            table = dataframe_to_bokeh_table(self.selected_time_series.time_series)

            # from holoviews.plotting.links import DataLink
            # table_link = DataLink(source=self.selected_time_series.time_series, target=table)

            # create base parameter list for time series objects
            parameter_list = ['series_type', 'units', 'output_units']

            # add xy locations for wind and wave series
            if self.selected_time_series.series_type == 'SERIES WIND' or \
                    self.selected_time_series.series_type == 'SERIES WAVE':
                parameter_list.append('x_location')
                parameter_list.append('y_location')

            # construct the time options panel
            time_options = pn.panel(
                self.selected_time_series.param,
                parameters=parameter_list
            )
            return pn.Row(table, time_options)

        else:
            return pn.Spacer(width=0)

    def view_time_series(self):
        time_series_view = pn.Column(
            self.param.time_series_select,
            pn.panel(self.time_series_table)
        )
        time_series_box = pn.WidgetBox('### Time Series', time_series_view)
        return time_series_box

    def view_materials(self):
        materials_view = pn.Column(
            pn.panel(self.param, parameters=['material_select'], show_name=False),
            pn.Row(
                pn.Column('***General Properties***', self._material_properties),
                pn.Column('***Constituent Properties***', self.param.transport_select, self._transport_properties),
                pn.Column('***Wind Properties***', pn.panel(self.selected_material.wind_properties, show_name=False))
            )
        )
        materials_box = pn.WidgetBox('### Materials', materials_view)

        return materials_box

    def view_general(self):
        tab_view = pn.Tabs(
            ('Model Constants', self.bound_cond.model_constants.panel()),
            ('Operation Parameters', self.bound_cond.operation_parameters.panel()),
            ('Interation Parameters', self.bound_cond.iteration_parameters.panel()),
            ('Output Control', self.bound_cond.output_control.panel()),
            ('Time Control', self.bound_cond.time_control.panel()),
            ('Constituent Properties', self.bound_cond.constituent_properties.panel())
        )
        general_box = pn.WidgetBox('### General Boundary Conditions', tab_view, width=900)
        return general_box

    def view_base(self):
        tab_view = pn.Tabs(
            ('Strings', dataframe_to_bokeh_table(self.bound_cond.boundary_strings)),
            ('Solution', dataframe_to_bokeh_table(self.bound_cond.solution_controls)),
            ('Stage Discharge', dataframe_to_bokeh_table(self.bound_cond.stage_discharge_boundary)),
            ('Friction', dataframe_to_bokeh_table(self.bound_cond.friction_controls)),
            ('Breach', dataframe_to_bokeh_table(self.bound_cond.breach_controls)),
            ('Flap Gates', dataframe_to_bokeh_table(self.bound_cond.flap_gates)),
            ('Sluice Gates', dataframe_to_bokeh_table(self.bound_cond.sluice_gates))
        )
        return tab_view

    @param.output()
    def output(self):
        return

    @property
    def tabs(self):
        tab_list = [
            ('Base', self.view_base),
            ('General', self.view_general),
            ('Materials', self.view_materials),
            ('Time Series', self.view_time_series)
        ]
        return tab_list

    def panel(self):
        main_panel = pn.Tabs(
            *self.tabs
        )

        return main_panel

    def panel_basic(self):
        return self.bound_cond.model_constants.panel()

    def string_to_geoviews(self):
        df = self.bound_cond.boundary_strings


