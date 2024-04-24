import dash
import dash_mantine_components as dmc
import logging
import plotly.graph_objects as go

import src.backend.ArticleDatabase as ArticleDatabase
import src.Config as Config


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

    # TODO: Add Difficulty Graph
    # Spider graph
    ## Get data
    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    values = [article[label] for label in labels]
    ## Create figure
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            marker=dict(
                color=Config.difficulty_colors[max(labels, key=lambda x: article[x])]
            ),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
    )
    ## Graph
    graph = dash.dcc.Graph(
        figure=fig,
        id="article-difficulty-graph",
    )

    # Title
    title = dmc.Title(children=article["title"], order=1)

    # Text
    text = dash.dcc.Markdown(
        children=article["text"],
        id="article-text",
    )

    # Container
    layout = dmc.Container(
        children=[graph, title, text],
        id="article-layout",
        size="80%",
    )

    return layout


# ---------------------------------------------------------------------------- #
#                                 REGISTER PAGE                                #
# ---------------------------------------------------------------------------- #

dash.register_page(__name__, path_template="/article/<article_id>")
