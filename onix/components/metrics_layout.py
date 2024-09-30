from dash import html, dcc

from onix.viewer import Viewer


def metrics_layout(viewer: Viewer):
    """ """
    return html.Div(
        children=[
            html.Br(),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "20% 60% 20%",
                },
                children=[
                    html.Div(),
                    html.Div(
                        [
                            html.H1("Map Metrics: MRSI Resolution"),
                            viewer.metrics.map_table,
                            html.Br(),
                            html.H1("Map Metrics: MRI Resolution"),
                            viewer.metrics.map_table_t1,
                            html.Br(),
                            html.H1("Ratio Mask Metrics"),
                            viewer.metrics.ratio_table,
                        ]
                    ),
                    html.Div(),
                ],
            ),
        ]
    )
