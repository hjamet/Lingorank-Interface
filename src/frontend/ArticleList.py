import dash
import dash_mantine_components as dmc
import logging

from src.backend.ArticleDatabase import ArticleDatabase


class ArticleList:

    def __init__(self):
        """A class listing the items in the database in the form of a grid of cards."""
        self.article_database = ArticleDatabase()

    def get_layout(self):
        """Get the layout of the article list.

        Returns:
            dmc.Container: The layout of the article list.
        """
        hello_world = dmc.Text(children="Hello, world!")

        # Container
        layout = dmc.Container(
            children=[hello_world],
            id="article-layout",
            size="80%",
        )

        return layout
