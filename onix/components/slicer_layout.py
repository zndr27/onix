from dash import html, dcc
from dash_bootstrap_components import dbc

from onix.viewer import Viewer
from onix.components.side_layout import side_layout


def slicer_layout(viewer: Viewer):
    """ """
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
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "50% 50%",
                },
                children=[
                    html.Div(),
                    html.Div(
                        dcc.RangeSlider(
                            id="color-slider",
                            min=0,
                            max=viewer.overlay._cmax,
                            marks=None,
                            step=1,
                            value=[0, viewer.overlay._cmax],
                            updatemode="mouseup",
                            vertical=False,
                        )
                    ),
                ],
            ),
        ],
    )


def main_layout(viewer: Viewer):
    return html.Div(
        style={
            "display": "grid", 
            "gridTemplateColumns": "25% 75%",
        },
        children=[
            side_layout(viewer),
            slicer_layout(viewer),
        ],
    )
