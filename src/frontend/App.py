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

    # ---------------------------- ADD CONTENT BUTTONS --------------------------- #
    # Add url button
    add_url_button = dmc.Button(
        children="Add URL",
        color="blue",
        id="add-url-button",
    )
    # Add text button
    add_text_button = dmc.Button(
        children="Add TEXT",
        color="green",
        id="add-text-button",  # Corrected ID to ensure uniqueness
    )
    add_group = dmc.Group(
        [
            add_url_button,
            add_text_button,
        ],
        align="center",
        position="right",
        spacing="md",
    )
    # Top bar
    top_bar = dmc.Group(
        children=[title, add_group],
        align="center",
        position="apart",
    )

    # --------------------------------- URL MODAL -------------------------------- #
    ## Url input
    url_input = dmc.TextInput(
        placeholder="Enter URL",
        id="url-input",
        style={"width": "100%"},
    )
    ## Submit url button
    submit_url_button = dmc.Button(
        children="Submit",
        color="blue",
        id="submit-url-button",
    )
    ## Creat stack for modal
    url_modal_stack = dmc.Stack(
        children=[url_input, submit_url_button],
        spacing="xl",
        align="flex-end",
    )
    ## Create Modal
    ## TODO: Check if an url is valid live
    url_modal = dmc.Modal(
        children=[url_modal_stack],
        title="Add URL",
        id="add-url-modal",
        centered=True,
    )

    # -------------------------------- TEXT MODAL -------------------------------- #
    text_input = dmc.Textarea(
        label="Please input you text",
        placeholder="Input Text",
        w="80%",
        autosize=True,
        minRows=2,
        id="text-input",
    )
    ## Submit text button
    submit_text_button = dmc.Button(
        children="Submit",
        color="blue",
        id="submit-text-button",
    )
    # TODO Center it
    text_modal_stack = dmc.Stack(
        children=[text_input, submit_text_button],
        spacing="xl",
        align="flex-end",
    )
    ## Create modal
    text_modal = dmc.Modal(
        children=[text_modal_stack],
        title="Add Text",
        id="add-text-modal",
        centered=True,
    )

    # Creat Header
    header = dmc.Header(
        children=[top_bar, url_modal, text_modal], height=50, withBorder=True
    )

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

    return opened


@dash.callback(
    dash.dependencies.Output("add-text-modal", "opened"),
    dash.dependencies.Input("add-text-button", "n_clicks"),
    dash.dependencies.Input("submit-text-button", "n_clicks"),
    dash.dependencies.State("add-text-modal", "opened"),
    prevent_initial_call=True,
)
def callback_add_text_modal(n_click: int, _, opened: bool):
    """Callback for the add text modal.

    Args:
        n_click (int): The number of clicks on the button
        opened (bool): Whether the modal is opened.
        text (str): The text to add.

    Returns:
        Any: Whether to open the modal or not.
    """
    # Get context
    ctx = dash.callback_context
    # If the add text button was clicked
    if ctx.triggered_id == "add-text-button":
        # Open the modal
        return not opened

    # If the submit text button was clicked
    if ctx.triggered_id == "submit-text-button":
        # Close the modal
        return not opened

    return opened


@dash.callback(
    dash.dependencies.Output("redirect", "href"),
    dash.dependencies.Input("submit-url-button", "n_clicks"),
    dash.dependencies.Input("submit-text-button", "n_clicks"),
    dash.dependencies.State("url-input", "value"),
    dash.dependencies.State("text-input", "value"),
    prevent_initial_call=True,
)
def callback_update_article_list(
    n_clicks_url: int, n_clicks_text: int, url: str, text: str
):
    # Add the article from url
    if n_clicks_url:
        ArticleDatabase.add_article_from_url(url)
    elif n_clicks_text:
        ArticleDatabase.add_article_from_text(text)

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
