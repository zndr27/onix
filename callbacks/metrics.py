from dash import dcc, html
from dash.dependencies import Input, Output

from nnfit.onix.viewer import OnixViewer


def metrics_callbacks(viewer: OnixViewer):
    """ """
    ###@viewer.app.callback(
    ###    Output('metrics-container', 'children'),
    ###    Input('metrics-options', 'value'),
    ###)
    ###def metrics_select(option):
    ###    #if option == 'map':
    ###    #    return viewer.metrics.map_table
    ###    #elif option == 'ratio':
    ###    #    return viewer.metrics.ratio_table
    ###    #else:
    ###    #    raise Exception("Invalid option for metrics")
    ###    return [
    ###        viewer.metrics.map_table,
    ###        html.Div(),
    ###        viewer.metrics.ratio_table,
    ###    ]
    pass
