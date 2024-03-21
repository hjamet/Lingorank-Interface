import dash
import dash_mantine_components as dmc

import src.backend.ArticleDatabase as ArticleDatabase


def layout():
    # Generate the cards
    cards = []
    for article_id in range(ArticleDatabase.get_article_database_length()):
        article = ArticleDatabase.get_article(article_id)
        cards.append(__create_card(article))

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


def __create_card(article: dict):
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


# ---------------------------------------------------------------------------- #
#                                 REGISTER PAGE                                #
# ---------------------------------------------------------------------------- #

dash.register_page(__name__, path_template="/")
