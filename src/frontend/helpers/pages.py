import dash_mantine_components as dmc
import src.Config as Config


def create_card(article: dict):
    """Create a nice card describing an article.

    Args:
        article (dict): The article to describe. (title, description, image, A1, A2, B1, B2, C1, C2, text)

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

    # Get estimated difficulty
    difficulty = max(["A1", "A2", "B1", "B2", "C1", "C2"], key=lambda x: article[x])
    difficulty_color = Config.difficulty_colors[difficulty]

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
                    dmc.Badge(
                        difficulty, color=difficulty_color, variant="filled", size="lg"
                    ),
                    dmc.Badge(
                        f"~{reading_time} minutes",
                        color=reading_time_color,
                        variant="dot",
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
        ],
        shadow="md",
    )

    return card
