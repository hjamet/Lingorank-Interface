import logging
import re

import dash
import dash_mantine_components as dmc

from src.backend.ArticleDatabase import ArticleDatabase


class ArticleList:

    def __init__(self, app):
        """A class listing the items in the database in the form of a grid of cards.

        Args:
            app (App): The class containing all important components of the app.
        """
        self.dash_app = app.dash_app
        self.article_database = app.article_database
        self.background_callback_manager = app.background_callback_manager

    def get_layout(self):
        """Get the layout of the article list.

        Returns:
            dmc.Container: The layout of the article list.
        """
        # Generate the cards
        cards = []
        for article_id in range(len(self.article_database)):
            article = self.article_database.get_article(article_id)
            cards.append(self.__create_card(article))

        # Loading skeleton
        loading_skeleton = dash.html.Div(
            id="article-loading-div",
            children=dmc.Stack(
                [
                    dmc.Skeleton(
                        width="100%",
                        height=160,
                        radius="md",
                    ),
                    dmc.Skeleton(
                        width="100%",
                        height=8,
                        radius="md",
                    ),
                    dmc.Skeleton(
                        width="100%",
                        height=8,
                        radius="md",
                    ),
                    dmc.Skeleton(
                        width="75%",
                        height=8,
                        radius="md",
                    ),
                ],
                spacing="xl",
            ),
            style={"display": "none"},
        )
        cards.append(loading_skeleton)

        # Grid
        grid = dmc.SimpleGrid(
            children=cards,
            cols=3,
            spacing="xl",
        )

        # Container
        layout = dmc.Container(
            children=[grid],
            id="article-layout",
            size="80%",
        )

        return layout

    # ---------------------------------------------------------------------------- #
    #                                PRIVATE METHODS                               #
    # ---------------------------------------------------------------------------- #

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
