from typing import Any
import flask
import dash


class App:
    def __init__(self):
        # Start the Flask server
        self.flask_app = flask.Flask(__name__)

        # Create the Dash app
        self.dash_app = dash.Dash(
            __name__, server=self.flask_app, url_base_pathname="/"
        )

    def __call__(self, debug: bool = False, port: int = 5000):
        """Run the server"""
        self.dash_app.run_server(debug=debug, port=port)
