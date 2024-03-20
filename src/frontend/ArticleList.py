import logging
import re

import dash
import dash_mantine_components as dmc

from src.backend.ArticleDatabase import ArticleDatabase


class ArticleList:

    def __init__(self, dash_app: dash.Dash):
        """A class listing the items in the database in the form of a grid of cards.

        Args:
            dash_app (dash.Dash): The dash app.
        """
        self.dash_app = dash_app

        self.article_database = ArticleDatabase()

    def get_layout(self):
        """Get the layout of the article list.

        Returns:
            dmc.Container: The layout of the article list.
        """
        hello_world = dmc.Text(children="Hello, world!")

        # Generate the cards
        cards = []
        for article_id in range(len(self.article_database)):
            article = self.article_database.get_article(article_id)
            cards.append(self.__create_card(article))

        # Grid
        grid = dmc.SimpleGrid(
            children=cards,
            cols=3,
            spacing="xl",
        )

        # Container
        layout = dmc.Container(
            children=[hello_world, grid],
            id="article-layout",
            size="80%",
        )

        return layout

    # ---------------------------------------------------------------------------- #
    #                                PRIVATE METHODS                               #
    # ---------------------------------------------------------------------------- #

    # --------------------------------- CALLBACKS -------------------------------- #

    # ---------------------------------- HELPERS --------------------------------- #

    def __create_card(self, article: dict):
        """Create a card for an article.

        Args:
            article (dict): The article.

        Returns:
            dmc.Card: The card.
        """
        # Image
        image = dmc.CardSection(
            dmc.Image(
                src=article["image"],
                height=160,
            )
        )

        # Title & Difficulty
        ## TODO: Link difficulty
        title_and_difficulty = dmc.Group(
            [
                dmc.Text(article["title"], weight=500),
                dmc.Badge("C1", color="orange", variant="light"),
            ],
            position="apart",
            mt="md",
            mb="xs",
        )

        # Description
        description = dmc.Text(article["description"], size="sm", color="dimmed")

        # Button
        read_button = dmc.Anchor(
            dmc.Button(
                "Read now",
                variant="light",
                color="blue",
                gradient={"from": "indigo", "to": "cyan"},
                fullWidth=True,
                mt="md",
                radius="md",
                id=f"read-button-{article['article_id']}",
            ),
            href=f"/article/{article['article_id']}",
        )

        # Card
        card = dmc.Card(
            children=[
                image,
                title_and_difficulty,
                description,
                read_button,
            ]
        )

        return card
