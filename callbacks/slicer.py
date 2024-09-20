from dash import ctx, dash_table, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go

from nnfit.onix.viewer import OnixViewer


import numpy as np


def freq_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def slicer_callbacks(viewer: OnixViewer):
    """ """
    @viewer.app.callback(
        Output("update-metrics", "children"),
        Input("update-metrics-button", "n_clicks"),
        State("fitt-cho-naa-slider", "value"),
        State("nnfit-cho-naa-slider", "value"),
    )
    def update_metrics_info(n_clicks, fitt_threshold, nnfit_threshold):
        """ """
        update_df = viewer.metrics.update_cho_naa_metrics(fitt_threshold, nnfit_threshold)
        update_table = viewer.metrics.get_table(update_df)
        return update_table

    @viewer.app.callback(
        Output(viewer.slicer_z.overlay_data.id, "data"),
        Output(viewer.slicer_y.overlay_data.id, "data"),
        Output(viewer.slicer_x.overlay_data.id, "data"),
        Output("histogram-container", "children"),
        Input("select-volume-1", "value"),
        Input("select-volume-2", "value"),
        Input("select-scaling", "value"),
        Input("select-operation", "value"),
        Input("select-mask", "value"),
        Input("alpha-slider", "value"),
        Input("fitt-cho-naa-slider", "value"),
        Input("nnfit-cho-naa-slider", "value"),
        Input("color-slider", "value"),
        Input("fixed_range_min", "value"),
        Input("fixed_range_max", "value"),
        Input("fixed_range_button", "n_clicks"),
        Input("nawm-avg-button", "n_clicks"),
    )
    def select_overlay(
        vol1, 
        vol2, 
        scaling, 
        operation, 
        mask, 
        alpha_value, 
        fitt_cho_naa_threshold,
        nnfit_cho_naa_threshold,
        color_range,
        fixed_range_min=None,
        fixed_range_max=None,
        fixed_range_button=0,
        nawm_button=0,
    ):
        """ """
        ###trig = ctx.triggered_id
        ###if trig == 'select-volume-1':
        ###    viewer.update_overlay_volume(vol1)
        ###elif trig == 'select-volume-2':
        ###    viewer.update_overlay_volume(vol2, second=True)
        ###elif trig == 'select-scaling':
        ###    viewer.update_overlay_scaling(scaling)
        ###elif trig == 'select-operation':
        ###    viewer.update_overlay_operation(operation)
        ###elif trig in ('alpha-slider', 'color-slider'):
        ###    viewer.update_overlay_colormap(color_range, alpha_value)
        ###else:
        ###    raise Exception("Invalid ctx.triggered_id")
        if (
            fixed_range_button % 2 == 1
            and fixed_range_min is not None 
            and fixed_range_max is not None
        ):
            fixed_range = (fixed_range_min, fixed_range_max)
        else:
            fixed_range = None

        # TODO
        viewer.update_cho_naa_threshold(
            fitt_threshold=fitt_cho_naa_threshold,
            nnfit_threshold=nnfit_cho_naa_threshold,
            use_avg = nawm_button % 2 == 1,
        )

        viewer.update_overlay_volume(vol1)
        viewer.update_overlay_volume(vol2, second=True)
        viewer.update_overlay_scaling(scaling)
        viewer.update_overlay_operation(operation)
        viewer.update_overlay_mask(mask)
        viewer.update_overlay_colormap(color_range, alpha_value, fixed_range)

        z_overlay, y_overlay, x_overlay = viewer.update_overlay()
        hist, xlo, xhi = viewer.overlay.get_hist()

        operate = (
            operation != 'None'
            and vol1 != 'None'
            and vol2 != 'None'
        )
        figure = go.FigureWidget()
        figure.update_layout(
            #title=f"histogram",
            title=(
                f"{vol1 if vol1 != 'None' else ''}"
                f"    {operation if operate else ''}"
                f"    {vol2 if operate else ''}"
            ),
            template="plotly_dark",
            height=400,
        )
        if hist is not None:
            figure.add_trace(
                go.Histogram(
                    x=hist,
                    name="overlay",
                )
            )
            figure.update_layout(
                xaxis=dict(range=[xlo, xhi]),
            )

        return z_overlay, y_overlay, x_overlay, dcc.Graph(figure=figure)

    ###@viewer.app.callback(
    ###    Output('values-container', 'children'),
    ###    Input(viewer.slicer_z.state.id, "data"),
    ###    Input(viewer.slicer_y.state.id, "data"),
    ###    Input(viewer.slicer_x.state.id, "data"),
    ###    Input('select-volume-1', 'value'),
    ###    Input('select-volume-2', 'value'),
    ###)
    ###def values_select(z_state, y_state, x_state, vol1, vol2):
    ###    """
    ###    """
    ###    if z_state == None or y_state == None or x_state == None:
    ###        zz, yy, xx = [i//2 for i in viewer.fitt_ds.t1.image.shape]
    ###    else:
    ###        zz = int(viewer.fitt_ds.t1.image.shape[0]) - z_state["zpos"] - 1
    ###        yy = y_state["zpos"]
    ###        xx = x_state["zpos"]
    ###
    ###    x, y, z = viewer.mri_to_si_coords(xx, yy, zz)
    ###
    ###    values = {}
    ###    values[vol1] = [viewer.overlay.get_volume_1(x, y, z),]
    ###    values[vol2] = [viewer.overlay.get_volume_2(x, y, z),]
    ###    values['overlay'] = [viewer.overlay.get_array(update=False)[z,y,x],]
    ###
    ###    df = pd.DataFrame.from_dict(values).astype(float).round(4)
    ###
    ###    return [dash_table.DataTable(
    ###        data = df.to_dict('records'),
    ###        columns = [{'id': c, 'name': c} for c in df.columns],
    ###        fixed_rows={'headers': True},
    ###        #style_table={
    ###        #    'height': 1000,
    ###        #},
    ###        style_cell={
    ###            'textAlign': 'center',
    ###            'minWidth': 100,
    ###            'maxWidth': 100,
    ###            'width': 100,
    ###        },
    ###        style_header = {
    ###            'backgroundColor': 'rgb(30, 30, 30)',
    ###            'color': 'white',
    ###        },
    ###        style_data = {
    ###            'backgroundColor': 'rgb(50, 50, 50)',
    ###            'color': 'white',
    ###        },
    ###    )]

    @viewer.app.callback(
        Output("spectrum-container", "children"),
        Input(viewer.slicer_z.state.id, "data"),
        Input(viewer.slicer_y.state.id, "data"),
        Input(viewer.slicer_x.state.id, "data"),
        Input("spectrum-component", "value"),
        Input("fitt-spec-button", "n_clicks"),
        Input("fitt-total-button", "n_clicks"),
        Input("fitt-base-button", "n_clicks"),
        Input("nnfit-spec-button", "n_clicks"),
        Input("nnfit-total-button", "n_clicks"),
        Input("nnfit-base-button", "n_clicks"),
        Input("nnfit-phase-button", "n_clicks"),
        Input("nnfit-shift-button", "n_clicks"),
        Input("theme-button", "n_clicks"),
        Input("legend-button", "n_clicks"),
        Input("title-button", "n_clicks"),
        Input("font-input", "value"),
        Input("yaxis-button", "n_clicks"),
    )
    def spectrum_select(
        z_state,
        y_state,
        x_state,
        component="real",
        f_spec=0,
        f_total=0,
        f_base=0,
        n_spec=0,
        n_total=0,
        n_base=0,
        n_phase=0,
        n_freq=0,
        n_theme=0,
        n_legend=0,
        n_title=0,
        font_size=None,
        n_yaxis=0,
    ):
        """ """
        if z_state == None or y_state == None or x_state == None:
            zz, yy, xx = [i // 2 for i in viewer.fitt_ds.t1.image.shape]
        else:
            zz = int(viewer.fitt_ds.t1.image.shape[0]) - z_state["zpos"] - 1
            yy = y_state["zpos"]
            xx = x_state["zpos"]

        frx, fry, frz = viewer.mri_to_si_coords(xx, yy, zz)
        nrx, nry, nrz = viewer.fitt_to_nnfit_coords(frx, fry, frz)
        fx, fy, fz = viewer.fitt_ds.ref_to_spec_coords(frx, fry, frz)
        nx, ny, nz = viewer.nnfit_ds.ref_to_spec_coords(nrx, nry, nrz)

        # Update real spectrum based on x,y,z
        fitt_spec = viewer.fitt_ds.spectra.get(
            fx, fy, fz, component=component, phase=None
        )
        fitt_total = viewer.fitt_ds.fit.get(fx, fy, fz, component=component, phase=None)
        fitt_base = viewer.fitt_ds.baseline.get(
            fx, fy, fz, component=component, phase=None
        )

        if n_phase % 2 == 0:
            phase = viewer.nnfit_ds.nnfit_ds.phase(x=nx, y=ny, z=nz)
            nnfit_spec = viewer.nnfit_ds.spectra.get(
                nx, ny, nz, component=component, phase=phase
            )
            nnfit_total = viewer.nnfit_ds.fit.get(
                nx, ny, nz, component=component, phase=phase
            )
            nnfit_base = viewer.nnfit_ds.baseline.get(
                nx, ny, nz, component=component, phase=phase
            )
        else:
            nnfit_spec = viewer.nnfit_ds.spectra.get(
                nx, ny, nz, component=component, phase=None
            )
            nnfit_total = viewer.nnfit_ds.fit.get(
                nx, ny, nz, component=component, phase=None
            )
            nnfit_base = viewer.nnfit_ds.baseline.get(
                nx, ny, nz, component=component, phase=None
            )

        if n_freq % 2 == 0:
            shift = viewer.nnfit_ds.nnfit_ds.shift(x=nx, y=ny, z=nz, option="points")
            nnfit_spec = freq_shift(nnfit_spec, -shift)
            nnfit_total = freq_shift(nnfit_total, -shift)
            nnfit_base = freq_shift(nnfit_base, -shift)

        # Create figure and add graphics objects to it.
        fw = go.FigureWidget()

        spec_color = "#555555" if n_theme % 2 == 0 else "#999999"
        nnfit_color = "#00aaaa" if n_theme % 2 == 0 else "#636EFA"
        nnfit_base_color = "#007777" if n_theme % 2 == 0 else "#00CC96"
        fitt_color = "#aa00aa" if n_theme % 2 == 0 else "#AB63FA"
        fitt_base_color = "#770077" if n_theme % 2 == 0 else "#FFA15A"
        #spec_color = "#555555" if n_theme % 2 == 0 else "#999999"
        #nnfit_color = "#00aaaa" if n_theme % 2 == 0 else "blue"
        #nnfit_base_color = "#007777" if n_theme % 2 == 0 else "orange"
        #fitt_color = "#aa00aa" if n_theme % 2 == 0 else "red"
        #fitt_base_color = "#770077" if n_theme % 2 == 0 else "green"

        if n_spec % 2 == 0:
            fw.add_trace(
                go.Scatter(
                    x=viewer.nnfit_ds.ppm,
                    y=nnfit_spec,
                    name="Spectrum",
                    #marker=dict(color="#555555"),
                    #marker=dict(color="#999999"),
                    marker=dict(color=spec_color),
                )
            )
        if n_total % 2 == 0:
            fw.add_trace(
                go.Scatter(
                    x=viewer.nnfit_ds.ppm,
                    y=nnfit_total,
                    name="NNFit",
                    #marker=dict(color="#00aaaa"),
                    #marker=dict(color="blue"),
                    marker=dict(color=nnfit_color),
                )
            )
        if n_base % 2 == 0:
            fw.add_trace(
                go.Scatter(
                    x=viewer.nnfit_ds.ppm,
                    y=nnfit_base,
                    name="NNFit Baseline",
                    #marker=dict(color="#007777"),
                    #marker=dict(color="green"),
                    marker=dict(color=nnfit_base_color),
                )
            )

        if f_spec % 2 == 0:
            fw.add_trace(
                go.Scatter(
                    x=viewer.nnfit_ds.ppm,
                    y=fitt_spec,
                    name="Spectrum",
                    #marker=dict(color="#555555"),
                    #marker=dict(color="#999999"),
                    marker=dict(color=spec_color),
                )
            )
        if f_total % 2 == 0:
            fw.add_trace(
                go.Scatter(
                    x=viewer.nnfit_ds.ppm,
                    y=fitt_total,
                    name="FITT",
                    #marker=dict(color="#aa00aa"),
                    #marker=dict(color="red"),
                    marker=dict(color=fitt_color),
                )
            )
        if f_base % 2 == 0:
            fw.add_trace(
                go.Scatter(
                    x=viewer.nnfit_ds.ppm,
                    y=fitt_base,
                    name="FITT Baseline",
                    #marker=dict(color="#770077"),
                    #marker=dict(color="orange"),
                    marker=dict(color=fitt_base_color),
                )
            )

        # UI elements
        fw.update_layout(
            title=(
                f"T1: {xx}, {yy}, {zz}"
                +  f"        SI: {fx}, {fy}, {fz}"
                #f"T1: {xx}, {yy}, {zz}"
                #+ f"        FITT: {fx}, {fy}, {fz}"
                #+ f"        NNFit: {nx}, {ny}, {nz}"
                if n_title % 2 == 0
                else None
            ),
            xaxis={"autorange": "reversed", "title": "ppm"},
            height=400,
            showlegend = True if n_legend % 2 == 0 else False,
            template = "plotly_dark" if n_theme % 2 == 0 else "plotly_white",
            # increase text size
        )

        fw.update_yaxes(visible=True if n_yaxis % 2 == 0 else False)

        if font_size is not None:
            fw.update_layout(font=dict(size=font_size))

        return dcc.Graph(figure=fw)
