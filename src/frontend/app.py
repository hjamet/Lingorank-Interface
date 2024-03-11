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
