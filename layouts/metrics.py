from dash import html, dcc

from nnfit.onix.viewer import OnixViewer


def metrics_layout(viewer: OnixViewer):
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
                            viewer.metrics.map_table,
                            html.Br(),
                            viewer.metrics.map_table_t1,
                            html.Br(),
                            viewer.metrics.ratio_table,
                        ]
                    ),
                    html.Div(),
                ],
            ),
        ]
    )
