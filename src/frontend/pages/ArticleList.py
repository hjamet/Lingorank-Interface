import dash
import dash_mantine_components as dmc

import src.backend.ArticleDatabase as ArticleDatabase
from src.frontend.helpers import pages as pages


def layout():
    # Generate the cards
    cards = []
    for article_id in range(ArticleDatabase.get_article_database_length()):
        article = ArticleDatabase.get_article(article_id)
        cards.append(pages.create_card(article))

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
#                                 REGISTER PAGE                                #
# ---------------------------------------------------------------------------- #

dash.register_page(__name__, path_template="/")
