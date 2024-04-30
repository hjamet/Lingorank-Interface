import logging

import dash
import dash_mantine_components as dmc
import plotly.graph_objects as go

import src.backend.ArticleDatabase as ArticleDatabase
import src.Config as Config
from src.frontend.helpers import pages as pages


def layout(article_id):
    # TODO Add Stepper to navigate between simplifications

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

    # Spider graph
    # TODO Add video
    # TODO : Add search
    ## Get data
    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    article_label = max(article, key=lambda x: article[x] if x in labels else -1)
    values = [round(article[label] * 100, 0) for label in labels]
    ## Create figure
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            marker=dict(color=Config.difficulty_colors[article_label]),
            hovertemplate="Ce texte a %{r}% d'être de niveau %{theta}.<extra></extra>",
            showlegend=False,
            name=article_label,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=False,
    )
    ## Graph
    graph = dash.dcc.Graph(
        figure=fig, id="article-difficulty-graph", config={"displayModeBar": False}
    )

    # Article Card
    article_card = pages.create_card(article)
    article_card.w = "40%"

    # Article Card & Graph
    article_card_graph = dmc.Group(
        [
            graph,
            article_card,
        ],
        position="center",
        spacing="xl",
        style={"marginTop": "5rem", "marginBottom": "5rem"},
        id="article-card-graph",
    )

    # Accordion
    accordion = dmc.Accordion(
        children=[__create_accordion(article, value=f"{article_id}:{article_label}:0")],
        id="article-accordion",
        variant="sepated",
        value=f"{article_id}:{article_label}:0",
    )

    # Simplify button
    simplify_map = {
        "A1": "A1",
        "A2": "A1",
        "B1": "A2",
        "B2": "A2",
        "C1": "B1",
        "C2": "B1",
    }
    simplify_button = dmc.Center(
        dmc.Menu(
            [
                dmc.MenuTarget(
                    dmc.Container(
                        dmc.Button(
                            children=f"Simplify to {simplify_map[article_label]}",
                            variant="gradient",
                            gradient={
                                "from": Config.difficulty_colors[article_label],
                                "to": Config.difficulty_colors[
                                    simplify_map[article_label]
                                ],
                            },
                            size="xl",
                            id="simplify-button",
                        ),
                        id="simplify-button-container",
                    )
                ),
                dmc.MenuDropdown(
                    [
                        dmc.MenuItem("test"),
                        dmc.MenuItem("test"),
                    ]
                ),
            ],
            trigger="hover",
        ),
        style={"marginTop": "5rem", "marginBottom": "5rem"},
    )

    # Container
    layout = dmc.Container(
        children=[article_card_graph, accordion, simplify_button],
        id="article-layout",
        size="80%",
    )

    return layout


# ---------------------------------------------------------------------------- #
#                                    PRIVATE                                   #
# ---------------------------------------------------------------------------- #


def __create_accordion(article: dict, value: str):
    # TODO Improve visual of title with badges
    # Get infos from accordion value
    article_id, article_label, simplification_id = value.split(":")

    if simplification_id == "0":
        accordion_title = f"Original text : {article_label}"
    else:
        accordion_title = f"Simplification n°{simplification_id} : {article_label}"

    # Title
    title = dmc.Title(children=article["title"], order=1)

    # Text
    text = dash.dcc.Markdown(
        children=article["text"],
        id="article-text",
    )

    # Accordion
    accordion_item = dmc.AccordionItem(
        [
            dmc.AccordionControl(children=accordion_title),
            dmc.AccordionPanel(
                children=[
                    title,
                    text,
                ]
            ),
        ],
        value=value,
        style={"backgroundColor": "#f0f0f0"},
    )

    return accordion_item


# ---------------------------------------------------------------------------- #
#                                   CALLBACK                                   #
# ---------------------------------------------------------------------------- #


@dash.callback(
    dash.dependencies.Output("article-card-graph", "children"),
    dash.dependencies.Output("simplify-button-container", "children"),
    dash.dependencies.Input("article-accordion", "value"),
    dash.dependencies.State("article-card-graph", "children"),
)
def call_update_graph(accordion_value: str, article_card_graph_children):
    if accordion_value is None:
        article_card_graph_children = (
            [article_card_graph_children[1]]
            if len(article_card_graph_children) == 2
            else article_card_graph_children
        )
        return article_card_graph_children, []

    # Get infos from accordion value
    article_id, article_label, simplification_id = accordion_value.split(":")

    # Load article
    article = ArticleDatabase.get_article(int(article_id))

    # Simplify button
    simplify_map = {
        "A1": "A1",
        "A2": "A1",
        "B1": "A2",
        "B2": "A2",
        "C1": "B1",
        "C2": "B1",
    }
    simplify_button = dmc.Button(
        children=f"Simplify to {simplify_map[article_label]}",
        variant="gradient",
        gradient={
            "from": Config.difficulty_colors[article_label],
            "to": Config.difficulty_colors[simplify_map[article_label]],
        },
        size="xl",
        id="simplify-button",
    )

    # Spider graph
    # TODO: Add new graph on simplified text
    # TODO Add video
    # TODO : Add search
    ## Get data
    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    values = [round(article[label] * 100, 0) for label in labels]
    ## Create figure
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            marker=dict(color=Config.difficulty_colors[article_label]),
            hovertemplate="Ce texte a %{r}% d'être de niveau %{theta}.<extra></extra>",
            showlegend=False,
            name=article_label,
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=False,
    )
    ## Graph
    graph = dash.dcc.Graph(
        figure=fig, id="article-difficulty-graph", config={"displayModeBar": False}
    )
    ## Update children
    if len(article_card_graph_children) == 2:
        article_card_graph_children[0] = graph
    else:
        article_card_graph_children = [graph, article_card_graph_children[0]]

    return article_card_graph_children, simplify_button


@dash.callback(
    dash.dependencies.Output("article-accordion", "children"),
    dash.dependencies.Output("article-accordion", "value"),
    dash.dependencies.Input("simplify-button", "n_clicks"),
    dash.dependencies.State("article-accordion", "value"),
    dash.dependencies.State("article-accordion", "children"),
)
def call_simplify_text(n_clicks, accordion_value, accordion_children):
    if n_clicks is None:
        return dash.no_update, dash.no_update

    # Get infos from accordion value
    if accordion_value is None:
        return dash.no_update, dash.no_update
    article_id, article_label, simplification_id = accordion_value.split(":")

    # Get current position in accordion
    current_position = int(simplification_id)
    ## Return next already loaded simplification
    if current_position + 1 < len(accordion_children):
        return (
            accordion_children,
            accordion_children[current_position + 1]["props"]["value"],
        )
    ## Create new simplification and return it
    else:
        ## Create new simplification
        simplification = ArticleDatabase.get_simplification(
            article_id=int(article_id), simplification_id=current_position
        )
        ## Update accordion
        simplification_label = max(
            simplification,
            key=lambda x: (
                simplification[x] if x in ["A1", "A2", "B1", "B2", "C1", "C2"] else -1
            ),
        )
        accordion_children.append(
            __create_accordion(
                simplification,
                value=f"{article_id}:{simplification_label}:{current_position+1}",
            )
        )
        return (
            accordion_children,
            f"{article_id}:{simplification_label}:{current_position+1}",
        )


# ---------------------------------------------------------------------------- #
#                                 REGISTER PAGE                                #
# ---------------------------------------------------------------------------- #

dash.register_page(__name__, path_template="/article/<article_id>")
