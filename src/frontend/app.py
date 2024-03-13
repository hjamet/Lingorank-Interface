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
            dash.dependencies.Output("layout", "children"),
            dash.dependencies.Input("add-url-button", "n_clicks"),
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

        # Add the top bar to the layout
        layout.children.append(top_bar)

        # Return the layout
        return layout

    def __callback_add_url(self, n_clicks: int) -> Any:
        """An internal method to add a URL to the layout.

        Args:
            n_clicks (int): The number of times the button has been clicked.

        Returns:
            Any: The new layout.
        """
        # If the button has been clicked
        if n_clicks:
            # Add a new URL input
            url_input = dmc.TextInput(
                label="URL",
                id={"type": "url-input", "index": n_clicks},
            )

            # Add the URL input to the layout
            self.dash_app.layout.children.append(url_input)

        # Return the new layout
        return self.dash_app.layout
