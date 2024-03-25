import flask
import dash
import dash_mantine_components as dmc
import re
import logging
import os
import src.Config as Config


import src.backend.ArticleDatabase as ArticleDatabase

MODAL_OPENED = False


def layout():
    # Empty layout
    layout = dmc.Container(
        children=[dash.dcc.Location(id="redirect", refresh=True)],
        id="layout",
        size="80%",
    )

    # ---------------------------------- HEADER ---------------------------------- #

    # Title of the app (underlined and clickable)
    title = dash.dcc.Link(
        href="/",  # Redirect to the root of the site
        children=dmc.Title(
            children="Lingorank Demo",
            order=1,
            align="center",
        ),
        style={"textDecoration": "none", "color": "black"},
    )

    # Add url button
    add_url_button = dmc.Button(
        children="Add URL",
        color="blue",
        id="add-url-button",
    )

    # Top bar
    top_bar = dmc.Group(
        children=[title, add_url_button],
        align="center",
        position="apart",
    )

    # Url input
    text_input = dmc.TextInput(
        placeholder="Enter URL",
        id="url-input",
        style={"width": "100%"},
    )

    # Submit url button
    submit_url_button = dmc.Button(
        children="Submit",
        color="blue",
        id="submit-url-button",
    )

    # Creat stack for modal
    modal_stack = dmc.Stack(
        children=[text_input, submit_url_button],
        spacing="xl",
        align="flex-end",
    )

    # Create Modal
    ## TODO: Check if an url is valid
    modal = dmc.Modal(
        children=[modal_stack],
        title="Add URL",
        id="add-url-modal",
        centered=True,
    )

    # Creat Header
    header = dmc.Header(children=[top_bar, modal], height=50, withBorder=True)

    # Create content container
    content = dash.page_container

    # Add components to the layout
    layout.children.extend([header, content])

    # Return the layout
    return layout


# ---------------------------------------------------------------------------- #
#                                   CALLBACKS                                  #
# ---------------------------------------------------------------------------- #


@dash.callback(
    dash.dependencies.Output("add-url-modal", "opened"),
    dash.dependencies.Input("add-url-button", "n_clicks"),
    dash.dependencies.Input("submit-url-button", "n_clicks"),
    dash.dependencies.State("add-url-modal", "opened"),
    prevent_initial_call=True,
)
def callback_add_url_modal(n_click: int, _, opened: bool):
    """Callback for the add url modal.

    Args:
        n_click (int): The number of clicks on the button
        opened (bool): Whether the modal is opened.
        url (str): The url to add.

    Returns:
        Any: Whether to open the modal or not.
    """
    # Get context
    ctx = dash.callback_context
    # If the add url button was clicked
    if ctx.triggered_id == "add-url-button":
        # Open the modal
        return not opened

    # If the submit url button was clicked
    if ctx.triggered_id == "submit-url-button":
        # Close the modal
        return not opened


@dash.callback(
    dash.dependencies.Output("redirect", "href"),
    dash.dependencies.Input("submit-url-button", "n_clicks"),
    dash.dependencies.State("url-input", "value"),
    prevent_initial_call=True,
)
def callback_update_article_list(n_clicks: int, url: str):
    # Add the article
    ArticleDatabase.add_article_from_url(url)

    # Redirect to the root of the site
    return "/"


# ---------------------------------------------------------------------------- #
#                                INITIALIZATION                                #
# ---------------------------------------------------------------------------- #
# Start the Flask server
flask_app = flask.Flask(__name__)

# Start background task manager
background_callback_manager = None
if "REDIS_URL" in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery

    celery_app = Celery(
        __name__,
        broker=os.environ["REDIS_URL"],
        backend=os.environ["REDIS_URL"],
    )
    background_callback_manager = dash.CeleryManager(celery_app)
else:
    # Diskcache for non-production apps when developing locally
    import diskcache

    cache = diskcache.Cache("./.cache")
    background_callback_manager = dash.DiskcacheManager(cache)

# Create the Dash app
dash_app = dash.Dash(
    __name__,
    server=flask_app,
    url_base_pathname="/",
    use_pages=True,
    background_callback_manager=background_callback_manager,
)

# Set the layout
dash_app.layout = layout()

# Load subpages
import src.frontend.pages.ArticleList as ArticleList


def run():
    dash_app.run_server(debug=Config.debug, port=Config.port)
