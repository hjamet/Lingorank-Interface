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
            children=[],
        )

        # Title
        title = dmc.Text(children="Lingorank Demo", size="xl", weight=700)

        # Add url button
        add_url_button = dmc.Button(
            children="Add URL",
            color="blue",
        )

        # Top bar
        top_bar = dmc.SimpleGrid(
            cols="1",
            spacing="lg",
            children=[title, add_url_button],
        )

        # Add the top bar to the layout
        layout.children.append(top_bar)

        # Return the layout
        return layout
