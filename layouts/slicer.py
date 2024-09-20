from dash import html, dcc
import dash_bootstrap_components as dbc

from nnfit.onix.viewer import OnixViewer


def slicer_layout(viewer: OnixViewer):
    """ """
    return html.Div(
        style={"maxWidth": "2560px", "margin": "auto"},
        children=[
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "20% 80%",
                },
                children=[
                    html.Div(
                        children=[
                            html.Br(),
                            dcc.Dropdown(
                                options=[
                                    {"label": x, "value": x}
                                    for x in viewer.overlay_list
                                ],
                                maxHeight=400,
                                value="None",
                                id="select-volume-1",
                            ),
                            html.Br(),
                            dcc.Dropdown(
                                options=[
                                    {"label": x, "value": x}
                                    for x in viewer.overlay_list
                                ],
                                maxHeight=400,
                                value="None",
                                id="select-volume-2",
                            ),
                            html.Br(),
                            dcc.Dropdown(
                                options=[
                                    {"label": x, "value": x}
                                    for x in viewer.overlay.scaling_list
                                ],
                                maxHeight=400,
                                value="None",
                                id="select-scaling",
                            ),
                            html.Br(),
                            dcc.Dropdown(
                                options=[
                                    {"label": x, "value": x}
                                    for x in viewer.overlay.operation_list
                                ],
                                maxHeight=400,
                                value="None",
                                id="select-operation",
                            ),
                            html.Br(),
                            dcc.Dropdown(
                                options=[
                                    {"label": x, "value": x}
                                    for x in viewer.overlay.mask_list
                                ],
                                maxHeight=400,
                                value="Brain Mask",
                                id="select-mask",
                            ),
                            html.Br(),
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        id="spectrum-component",
                                        class_name="btn-group",
                                        input_class_name="btn-check",
                                        label_class_name="btn btn-outline-primary",
                                        label_checked_class_name="active",
                                        options=[
                                            {
                                                "label": "Real", 
                                                "value": "real"
                                            },
                                            {
                                                "label": "Imaginary",
                                                "value": "imaginary",
                                            },
                                            {
                                                "label": "Magnitude",
                                                "value": "magnitude",
                                            },
                                        ],
                                        value="real",
                                    ),
                                ],
                                className="radio-group",
                            ),
                            html.Br(),
                            dbc.ButtonGroup(
                                children=[
                                    dbc.Button(
                                        "Spectra",
                                        id="fitt-spec-button",
                                        style=dict(backgroundColor="#990099"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Fit",
                                        id="fitt-total-button",
                                        style=dict(backgroundColor="#990099"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Base",
                                        id="fitt-base-button",
                                        style=dict(backgroundColor="#990099"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Phase",
                                        id="fitt-phase-button",
                                        style=dict(backgroundColor="#990099"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Shift",
                                        id="fitt-shift-button",
                                        style=dict(backgroundColor="#990099"),
                                        n_clicks=0,
                                    ),
                                ],
                                vertical=False,
                            ),
                            html.Br(),
                            html.Br(),
                            dbc.ButtonGroup(
                                children=[
                                    dbc.Button(
                                        "Spectra",
                                        id="nnfit-spec-button",
                                        style=dict(backgroundColor="#009999"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Fit",
                                        id="nnfit-total-button",
                                        style=dict(backgroundColor="#009999"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Base",
                                        id="nnfit-base-button",
                                        style=dict(backgroundColor="#009999"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Phase",
                                        id="nnfit-phase-button",
                                        style=dict(backgroundColor="#009999"),
                                        n_clicks=0,
                                    ),
                                    dbc.Button(
                                        "Shift",
                                        id="nnfit-shift-button",
                                        style=dict(backgroundColor="#009999"),
                                        n_clicks=0,
                                    ),
                                ],
                                vertical=False,
                            ),
                            html.Br(),
                            html.Br(),
                            html.Div(
                                [
                                    dcc.Input(id="fixed_range_min", type="number"),
                                    html.Br(),
                                    dcc.Input(id="fixed_range_max", type="number"),
                                    html.Br(),
                                    html.Br(),
                                    dbc.Button("Fixed Range", id="fixed_range_button", n_clicks=0),
                                ],
                            ),
                            html.Br(),
                            html.Div(
                                [
                                    dbc.Button("Theme", id="theme-button", n_clicks=0),
                                    dbc.Button("Legend", id="legend-button", n_clicks=0),
                                    dbc.Button("Title", id="title-button", n_clicks=0),
                                    dbc.Button("Y-axis", id="yaxis-button", n_clicks=0),
                                    dcc.Input(id="font-input", type="number"),
                                ],
                            ),
                            dcc.Slider(
                                id="alpha-slider",
                                min=0,
                                max=255,
                                marks=None,
                                step=1,
                                value=128,
                                updatemode="mouseup",
                                vertical=False,
                            ),
                            dcc.Slider(
                                id="fitt-cho-naa-slider",
                                min=1.0,
                                max=5.0,
                                marks={
                                    1.0: '1.0',
                                    1.5: '1.5',
                                    2.0: '2.0',
                                    2.5: '2.5',
                                    3.0: '3.0',
                                    3.5: '3.5',
                                    4.0: '4.0',
                                    4.5: '4.5',
                                    5.0: '5.0',
                                },
                                step=0.1,
                                value=2.0,
                                updatemode="mouseup",
                                vertical=False,
                            ),
                            dcc.Slider(
                                id="nnfit-cho-naa-slider",
                                min=1.0,
                                max=5.0,
                                marks={
                                    1.0: '1.0',
                                    1.5: '1.5',
                                    2.0: '2.0',
                                    2.5: '2.5',
                                    3.0: '3.0',
                                    3.5: '3.5',
                                    4.0: '4.0',
                                    4.5: '4.5',
                                    5.0: '5.0',
                                },
                                step=0.1,
                                value=2.0,
                                updatemode="mouseup",
                                vertical=False,
                            ),
                            dbc.Button("same nawm avg", id="nawm-avg-button", n_clicks=0),
                        ],
                    ),
                    html.Div(
                        id="visual-container",
                        children=[
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "33% 33% 33%",
                                },
                                children=[
                                    html.Div(
                                        id="y-div",
                                        children=[
                                            viewer.slicer_y.graph,
                                            viewer.slicer_y.slider,
                                            *viewer.slicer_y.stores,
                                        ],
                                    ),
                                    html.Div(
                                        id="z-div",
                                        children=[
                                            viewer.slicer_z.graph,
                                            viewer.slicer_z.slider,
                                            *viewer.slicer_z.stores,
                                        ],
                                    ),
                                    html.Div(
                                        id="x-div",
                                        children=[
                                            viewer.slicer_x.graph,
                                            viewer.slicer_x.slider,
                                            *viewer.slicer_x.stores,
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "50% 50%",
                                },
                                children=[
                                    html.Div(
                                        id="spectrum-container",
                                    ),
                                    html.Div(
                                        id="histogram-container",
                                    ),
                                ],
                            ),
                            #html.Br(),
                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "50% 50%",
                                },
                                children=[
                                    html.Div(
                                        [
                                            ###dcc.Slider(
                                            ###    id="alpha-slider-2",
                                            ###    min=0,
                                            ###    max=255,
                                            ###    marks=None,
                                            ###    step=1,
                                            ###    value=128,
                                            ###    updatemode="mouseup",
                                            ###    vertical=False,
                                            ###),
                                        ],
                                    ),
                                    html.Div(
                                        [
                                            dcc.RangeSlider(
                                                id="color-slider",
                                                min=0,
                                                max=viewer.overlay._cmax,
                                                marks=None,
                                                step=1,
                                                value=[0, viewer.overlay._cmax],
                                                updatemode="mouseup",
                                                vertical=False,
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        id = "update-metrics",
                                    ),
                                    dbc.Button(
                                        "Update",
                                        color="primary",
                                        n_clicks=0,
                                        id="update-metrics-button",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
