import dash
import dash_mantine_components as dmc
import logging

import src.backend.ArticleDatabase as ArticleDatabase


def layout(article_id):
    # Check if article_id is an integer
    try:
        article_id = int(article_id)
    except:
        article_id = -1

    # Get the article
    article = ArticleDatabase.get_article(article_id)
    if article is None:
        logging.error(f"The article with id {article_id} does not exist.")
        return dmc.Text(children="The article does not exist.")

    # Title
    title = dmc.Title(children=article["title"], order=1)

    # Text
    text = dash.dcc.Markdown(
        children=article["text"],
        id="article-text",
    )

    # Container
    layout = dmc.Container(
        children=[title, text],
        id="article-layout",
        size="80%",
    )

    return layout


# ---------------------------------------------------------------------------- #
#                                 REGISTER PAGE                                #
# ---------------------------------------------------------------------------- #

dash.register_page(__name__, path_template="/article/<article_id>")
