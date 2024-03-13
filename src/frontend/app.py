from typing import Any
import flask
import dash
import dash_mantine_components as dmc


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
        self.dash_app.callback(
            dash.dependencies.Output("add-url-modal", "opened"),
            dash.dependencies.Input("add-url-button", "n_clicks"),
            dash.dependencies.State("add-url-modal", "opened"),
            prevent_initial_call=True,
        )(self.__callback_add_url)

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
        layout = dmc.Container(children=[], id="layout", size="100%")

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
        )

        # Create Modal
        modal = dmc.Modal(
            children=[text_input],
            title="Add URL",
            id="add-url-modal",
            centered=True,
        )

        # Add components to the layout
        layout.children.extend([top_bar, modal])

        # Return the layout
        return layout

    def __callback_add_url(self, n_clicks: int, opened: bool) -> bool:
        """An internal method to handle the add url button click.

        Args:
            n_clicks (int): The number of times the button has been clicked.
            opened (bool): Whether the modal is open.

        Returns:
            bool: Whether the modal should be open.
        """
        # Return the opposite of the current state
        return not opened
