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

    # Get estimated difficulty
    difficulty = max(["A1", "A2", "B1", "B2", "C1", "C2"], key=lambda x: article[x])
    difficulty_color = {
        "A1": "lime",
        "A2": "blue",
        "B1": "violet",
        "B2": "yellow",
        "C1": "orange",
        "C2": "red",
    }[difficulty]

    # Estimated reading time
    reading_time = len(article["text"]) // 1000
    if reading_time < 1:
        reading_time_color = "lime"
    elif reading_time < 3:
        reading_time_color = "blue"
    elif reading_time < 5:
        reading_time_color = "violet"
    elif reading_time < 7:
        reading_time_color = "yellow"
    elif reading_time < 10:
        reading_time_color = "orange"
    else:
        reading_time_color = "red"

    # Title & Difficulty
    title_difficulty_readtime = dmc.Group(
        [
            dmc.Text(article["title"], weight=500),
            dmc.Stack(
                [
                    dmc.Badge(difficulty, color=difficulty_color, variant="filled"),
                    dmc.Badge(
                        f"~{reading_time}m", color=reading_time_color, variant="dot"
                    ),
                ],
                align="center",
                spacing="xs",
            ),
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
            title_difficulty_readtime,
            description,
            read_button,
        ]
    )

    return card


# ---------------------------------------------------------------------------- #
#                                 REGISTER PAGE                                #
# ---------------------------------------------------------------------------- #

dash.register_page(__name__, path_template="/")
