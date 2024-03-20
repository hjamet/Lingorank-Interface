from typing import Any
import flask
import dash
import dash_mantine_components as dmc
import re
import logging
import os


from src.backend.ArticleDatabase import ArticleDatabase
from src.frontend.ArticlePage import ArticlePage
from src.frontend.ArticleList import ArticleList


# TODO : Transform class into page -> register the page in the app
class App:
    def __init__(self):
        # Start the Flask server
        self.flask_app = flask.Flask(__name__)

        # Start background task manager
        if "REDIS_URL" in os.environ:
            # Use Redis & Celery if REDIS_URL set as an env variable
            from celery import Celery

            celery_app = Celery(
                __name__,
                broker=os.environ["REDIS_URL"],
                backend=os.environ["REDIS_URL"],
            )
            self.background_callback_manager = dash.CeleryManager(celery_app)
        else:
            # Diskcache for non-production apps when developing locally
            import diskcache

            cache = diskcache.Cache("./.cache")
            self.background_callback_manager = dash.DiskcacheManager(cache)

        # Create the Dash app
        self.dash_app = dash.Dash(
            __name__,
            server=self.flask_app,
            url_base_pathname="/",
            background_callback_manager=self.background_callback_manager,
        )

        # Set the layout
        self.dash_app.layout = self.__get_layout()

        # Add callbacks
        ## Url Modal
        self.dash_app.callback(
            dash.dependencies.Output("add-url-modal", "opened"),
            dash.dependencies.Input("add-url-button", "n_clicks"),
            dash.dependencies.Input("submit-url-button", "n_clicks"),
            dash.dependencies.State("add-url-modal", "opened"),
            dash.dependencies.State("url-input", "value"),
            prevent_initial_call=True,
            background=True,
            running=[
                (
                    dash.dependencies.Output("article-loading-div", "style"),
                    {"display": "block"},
                    {"display": "none"},
                )
            ],
        )(self.__callback_add_url_modal)
        ## Content
        self.dash_app.callback(
            dash.dependencies.Output("content", "children"),
            dash.dependencies.Input("url", "pathname"),
            prevent_initial_call=True,
        )(self.__callback_update_content)

        # Load the article database
        self.article_database = ArticleDatabase()

        # Load subpages
        self.article_page = ArticlePage()
        self.article_list = ArticleList(self)

    def __call__(self, debug: bool = False, port: int = 5000):
        """Run the server

        Args:
            debug (bool, optional): Whether to run the server in debug mode. Defaults to False.
            port (int, optional): The port to run the server on. Defaults to 5000.
        """
        # Run the server
        self.dash_app.run_server(debug=debug, port=port)

    def __get_layout(self):
        """An internal method to get the layout of the app.

        Returns:
            str: The layout of the app.
        """
        # Empty layout
        layout = dmc.Container(
            children=[
                dash.dcc.Location(id="url", refresh=False),
            ],
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
        content = dmc.Container(
            children=[],
            id="content",
            fluid=True,
            style={
                "height": "100%",
                "overflowY": "auto",
                "padding": "20px",
                "boxSizing": "border-box",
            },
        )

        # Add components to the layout
        layout.children.extend([header, content])

        # Return the layout
        return layout

    # ---------------------------------------------------------------------------- #
    #                                   CALLBACKS                                  #
    # ---------------------------------------------------------------------------- #

    def __callback_add_url_modal(self, n_click: int, _, opened: bool, url: str):
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
            return not opened

        # If the submit url button was clicked
        if ctx.triggered_id == "submit-url-button":
            # Add the url to the database
            self.article_database.add_article_from_url(url)

            # Close the modal
            return False

    def __callback_update_content(self, path: str):
        """Callback to update the content of the app.

        Args:
            path (str): The path to the database.
        """
        # Capture the article id
        article_id = re.findall(r"/article/(\d+)", path)
        if article_id:
            return self.article_page.get_layout(
                article_id=int(article_id[0]),
            )
        elif path == "/":
            return self.article_list.get_layout()
        else:
            logging.warning(f"The path {path} is not valid.")
