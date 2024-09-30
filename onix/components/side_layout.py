from dash import html, dcc
import dash_bootstrap_components as dbc

from onix.viewer import Viewer


def map_1_dropdown(viewer: Viewer):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "20% 80%",
            "justifyItems": "center",
        },
        children=[
            "Map 1",
            dcc.Dropdown(
                options = ([
                    {
                        "label": x, 
                        "value": x,
                        "disabled": (
                            True if x in ('FITT', 'NNFit', 'Masks') else False
                        ),
                    }
                    for x in viewer.overlay_list
                ]),
                maxHeight=400,
                value="None",
                id="select-volume-1",
                style={
                    "width": "95%",
                },
            ),
        ],
    )


def map_2_dropdown(viewer: Viewer):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "20% 80%",
            "justifyItems": "center",
        },
        children=[
            "Map 2",
            dcc.Dropdown(
                options = ([
                    {
                        "label": x, 
                        "value": x,
                        "disabled": (
                            True if x in ('FITT', 'NNFit', 'Masks') else False
                        ),
                    }
                    for x in viewer.overlay_list
                ]),
                maxHeight=400,
                value="None",
                id="select-volume-2",
                style=dict(width="95%"),
            ),
        ],
    )


def operation_dropdown(viewer: Viewer):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "20% 80%",
            "justifyItems": "center",
        },
        children=[
            "Operation",
            dcc.Dropdown(
                options=[
                    {"label": x, "value": x}
                    for x in viewer.overlay.operation_list
                ],
                maxHeight=400,
                value="None",
                id="select-operation",
                style=dict(width="95%"),
            ),
        ],
    )


def mask_dropdown(viewer: Viewer):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "20% 80%",
            "justifyItems": "center",
        },
        children=[
            "Mask",
            dcc.Dropdown(
                options=[
                    {"label": x, "value": x}
                    for x in viewer.overlay.mask_list
                    if x in ("Brain Mask", "QMap", "nnqmap")
                ],
                maxHeight=400,
                value="Brain Mask",
                id="select-mask",
                style=dict(width="95%"),
            ),
        ],
    )


def alpha_slider(viewer: Viewer):
    html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "20% 80%",
        },
        children=[
            html.Div(
                "Opacity", 
                style={
                    "textAlign": "center",
                },
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
        ],
    )


def spectrum_component_buttons(viewer: Viewer):
    return html.Div(
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
                        "label": "Imag",
                        "value": "imaginary",
                    },
                    {
                        "label": "Norm",
                        "value": "magnitude",
                    },
                ],
                value="real",
            ),
        ],
        className="radio-group",
    )


def fixed_range_inputs(viewer: Viewer):
    return html.Div(
        [
            html.Div(),
            dbc.Input(
                id="fixed_range_min", 
                type="number", 
                style={
                    "border-color": "#999999",
                    "width": "90%",
                },
            ),
            dbc.Input(
                id="fixed_range_max", 
                type="number", 
                style={
                    "border-color": "#999999",
                    "width": "90%",
                },
            ),
            dbc.Button(
                "Range", 
                id="fixed_range_button", 
                n_clicks=0,
            ),
            html.Div(),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "5% 30% 30% 30% 5%",
            "justifyItems": "center",
        },
    )


def fit_1_buttons(viewer: Viewer):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "5% 15% 75% 5%",
            "justifyItems": "center",
        },
        children=[
            html.Div(),
            html.B(
                "FITT",
                style={
                    "color": "#FF00FF",
                    "fontSize": "20px",
                },
            ),
            dbc.ButtonGroup(
                children=[
                    dbc.Button(
                        "Spectra",
                        id="fitt-spec-button",
                        style=dict(backgroundColor="#660066"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Fit",
                        id="fitt-total-button",
                        style=dict(backgroundColor="#660066"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Base",
                        id="fitt-base-button",
                        style=dict(backgroundColor="#660066"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Phase",
                        id="fitt-phase-button",
                        style=dict(backgroundColor="#660066"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Shift",
                        id="fitt-shift-button",
                        style=dict(backgroundColor="#660066"),
                        n_clicks=0,
                    ),
                ],
                vertical=False,
            ),
            html.Div(),
        ],
    )


def fit_2_buttons(viewer: Viewer):
    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "5% 15% 75% 5%",
            "justifyItems": "center",
        },
        children=[
            html.Div(),
            html.B(
                "NNFit",
                style={
                    "color": "#00FFFF",
                    "fontSize": "20px",
                    "textAlign": "center",
                },
            ),
            dbc.ButtonGroup(
                children=[
                    dbc.Button(
                        "Spectra",
                        id="nnfit-spec-button",
                        style=dict(backgroundColor="#006666"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Fit",
                        id="nnfit-total-button",
                        style=dict(backgroundColor="#006666"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Base",
                        id="nnfit-base-button",
                        style=dict(backgroundColor="#006666"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Phase",
                        id="nnfit-phase-button",
                        style=dict(backgroundColor="#006666"),
                        n_clicks=0,
                    ),
                    dbc.Button(
                        "Shift",
                        id="nnfit-shift-button",
                        style=dict(backgroundColor="#006666"),
                        n_clicks=0,
                    ),
                ],
                vertical=False,
            ),
            html.Div(),
        ],
    )


def mask_sliders(viewer: Viewer):
    html.Div(
        html.B(
            "Cho/NAA [nawm]",
            style={
                "fontSize": "20px",
            },
        ),
        style={
            "display": "grid",
            "justifyItems": "center",
        },
    ),
    html.Br(),
    html.Div(
        [
            html.B(
                "FITT", 
                style={
                    "textAlign": "center",
                    "color": "#FF00FF",
                    "fontSize": "20px",
                },
            ),
            dcc.Slider(
                id="fitt-cho-naa-slider",
                min=1.0,
                max=5.0,
                marks={
                    float(x): str(x) for x in range(1, 6)
                },
                step=0.1,
                value=2.0,
                updatemode="mouseup",
                vertical=False,
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "20% 80%",
        },
    ),
    html.Div(
        [
            html.B(
                "NNFit", 
                style={
                    "textAlign": "center",
                    "color": "#00FFFF",
                    "fontSize": "20px",
                },
            ),
            dcc.Slider(
                id="nnfit-cho-naa-slider",
                min=1.0,
                max=5.0,
                marks={
                    float(x): str(x) for x in range(1, 6)
                },
                step=0.1,
                value=2.0,
                updatemode="mouseup",
                vertical=False,
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "20% 80%",
        },
    ),


def side_layout(viewer: Viewer):
    return html.Div(
        children=[
            html.Br(),
            map_1_dropdown(viewer),
            html.Br(),
            map_2_dropdown(viewer),
            html.Br(),
            operation_dropdown(viewer),
            html.Br(),
            mask_dropdown(viewer),
            html.Br(),
            alpha_slider(viewer),
            html.Div(
                [
                    spectrum_component_buttons(viewer),
                    fixed_range_inputs(viewer),
                ],
                style={
                    # Center element
                    "display": "grid",
                    "gridTemplateColumns": "40% 60%",
                    "justifyItems": "center",
                },
            ),
            html.Br(),
            fit_1_buttons(viewer),
            html.Br(),
            fit_2_buttons(viewer),
            html.Br(),
            mask_sliders(viewer),
        ],
    )
