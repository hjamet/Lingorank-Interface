from typing import Any
import flask
import dash
import dash_mantine_components as dmc

from src.backend.ArticleDatabase import ArticleDatabase


class App:
    def __init__(self):
        # Start the Flask server
        self.flask_app = flask.Flask(__name__)

        # Create the Dash app
        self.dash_app = dash.Dash(
            __name__,
            server=self.flask_app,
            url_base_pathname="/",
        )

        # Set the layout
        self.dash_app.layout = self.__get_layout()

        # Add callbacks
        ## Add url modal
        self.dash_app.callback(
            dash.dependencies.Output("add-url-modal", "opened"),
            dash.dependencies.Input("add-url-button", "n_clicks"),
            dash.dependencies.Input("submit-url-button", "n_clicks"),
            dash.dependencies.State("add-url-modal", "opened"),
            dash.dependencies.State("url-input", "value"),
            prevent_initial_call=True,
        )(self.__callback_add_url_modal)

        # Load the article database
        self.article_database = ArticleDatabase()

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
        layout = dmc.Container(children=[], id="layout", size="80%")

        # Title of the app (underlined)
        title = dmc.Title(
            children="Lingorank Demo",
            order=1,
            align="center",
            style={"textDecoration": "underline"},
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

        # Add components to the layout
        layout.children.extend([top_bar, modal])

        # Return the layout
        return layout

    # ---------------------------------------------------------------------------- #
    #                                   CALLBACKS                                  #
    # ---------------------------------------------------------------------------- #

    def __callback_add_url_modal(
        self, n_clicks_add: int, n_clicks_submit: int, opened: bool, url: str
    ):
        """Callback for the add url modal.

        Args:
            n_clicks (int): The number of clicks on the add url button.
            n_clicks_submit (int): The number of clicks on the submit url button.
            opened (bool): Whether the modal is opened.
            url (str): The url to add.

        Returns:
            Any: Whether to open the modal or not.
        """
        # If the add url button was clicked
        if n_clicks_add:
            return not opened

        # If the submit url button was clicked
        if n_clicks_submit:
            # Add the url to the database
            self.article_database.add_article_from_url(url)

            # Close the modal
            return False
