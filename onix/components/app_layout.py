import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from onix.viewer import Viewer


def app_layout(viewer: Viewer):
    """ """
    nav_item = dbc.NavItem(dbc.NavLink("Browser", href="#"))

    dropdown = dbc.DropdownMenu(
        children=[
            dbc.DropdownMenuItem(dcc.Link(page["name"], href=page["relative_path"]))
            for page in dash.page_registry.values()
        ],
        nav=True,
        in_navbar=True,
        label="Menu",
    )

    return dbc.Container(
        [
            dbc.Navbar(
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Img(src=viewer._PLOTLY_LOGO, height="30px"),
                                ),
                                dbc.Col(
                                    dbc.NavbarBrand(f"ONIX", className="ms-1"),
                                ),
                                dbc.Col(),
                                dbc.Col(
                                    html.H5(
                                        [
                                            dbc.Badge(
                                                f"Subject: {viewer.subject_id}",
                                                color="dark",
                                                className="me-1",
                                            ),
                                        ]
                                    ),
                                ),
                                dbc.Col(
                                    html.H5(
                                        [
                                            dbc.Badge(
                                                f"Date: {viewer.study_date}",
                                                color="dark",
                                                className="me-1",
                                            ),
                                        ]
                                    ),
                                ),
                            ],
                        ),
                        dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                        dbc.Collapse(
                            dbc.Nav(
                                [nav_item, dropdown],
                                class_name="ms-auto",
                                navbar=True,
                            ),
                            id="navbar-collapse",
                            navbar=True,
                        ),
                    ],
                ),
                color="primary",
                dark=True,
                style={"height": "60px"},
            ),
            dash.page_container,
        ],
        fluid=True,
        class_name="dbc",
    )
