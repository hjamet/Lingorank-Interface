import dash
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash_iconify import DashIconify

import src.backend.ArticleDatabase as ArticleDatabase
import src.backend.Models as Models
import src.Config as Config
from src.Exceptions import OpenAIKeyNotSet
from src.frontend.helpers import pages as pages
from src.Logger import logger


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
        logger.error(f"The article with id {article_id} does not exist.")
        return dmc.Text(children="The article does not exist.")

    # OpenAI Key Modal
    ## OpenAI Key Input
    openai_key_input = dmc.TextInput(
        placeholder="Enter OpenAI API Key",
        id="openai-key-input",
        style={"width": "100%"},
    )
    ## Submit OpenAI Key Button
    submit_openai_key_button = dmc.Button(
        children="Submit",
        color="blue",
        id="submit-openai-key-button",
    )
    ## Create Stack for Modal
    openai_key_modal_stack = dmc.Stack(
        children=[openai_key_input, submit_openai_key_button],
        spacing="xl",
        align="flex-end",
    )
    ## Create Modal
    # TODO : Improve key visual (the key seems to be incorrect etc.)
    # TODO : Live check if the key is correct
    openai_key_modal = dmc.Modal(
        children=[openai_key_modal_stack],
        title="Please provide your OpenAI API key",
        id="openai-key-modal",
        centered=True,
        zIndex=10000,
    )

    # Spider graph
    # TODO Add video
    # TODO : Add search
    # TODO : Bug with incorrect major label
    ## Get data
    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    article_label = max(article, key=lambda x: article[x] if x in labels else -1)

    # Article Card
    article_card = pages.create_card(article)
    article_card.w = "80%"

    # Article Card & Graph
    article_card_graph = dmc.SimpleGrid(
        [
            dmc.Container(style={"width": "100%"}),
            article_card,
        ],
        cols=2,
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
    simplify_menu = dmc.Center(
        dmc.Menu(
            [
                dmc.MenuTarget(
                    dmc.Container(
                        simplify_button,
                        id="simplify-button-container",
                    )
                ),
                dmc.MenuDropdown(
                    [
                        dmc.MenuItem(key, id="model-button")
                        for key, value in Models.list_available_models().items()
                    ]
                ),
            ],
            trigger="hover",
            transition={"transition": "rotate-right", "duration": 150},
            position="right",
        ),
        style={"marginTop": "3rem", "marginBottom": "3rem"},
    )

    # Simplification Skeleton & Progress Bar
    simplification_progress_bar = dmc.Progress(
        radius="xl",
        size="xl",
        animate=True,
        color="lime",
        id="simplification-progress-bar",
    )
    small_space = dmc.Space(h="xl")
    simplification_skeleton = dmc.Stack(
        children=[
            dmc.Skeleton(h=50, mb="xl"),
            dmc.Skeleton(h=8, radius="xl"),
            dmc.Skeleton(h=8, my=6),
            dmc.Skeleton(h=8, w="85%", radius="xl"),
            dmc.Skeleton(h=50, mb="xl"),
            dmc.Skeleton(h=8, radius="xl"),
            dmc.Skeleton(h=8, my=6),
            dmc.Skeleton(h=8, w="65%", radius="xl"),
        ],
    )
    simplification_in_progress = dmc.Stack(
        children=[simplification_progress_bar, small_space, simplification_skeleton],
        align="center",
        style={"display": "none"},
        id="simplification-in-progress",
    )

    # Container
    layout = dmc.Container(
        children=[
            article_card_graph,
            accordion,
            simplification_in_progress,
            simplify_menu,
            openai_key_modal,
        ],
        id="article-layout",
        size="80%",
    )

    return layout


# ---------------------------------------------------------------------------- #
#                                    PRIVATE                                   #
# ---------------------------------------------------------------------------- #


def __create_accordion(article: dict, value: str):
    # Get infos from accordion value
    article_id, article_label, simplification_id = value.split(":")

    accordion_title_badge = dmc.Badge(
        article_label, color=Config.difficulty_colors[article_label], variant="filled"
    )
    if simplification_id == "0":
        accordion_title_icon = dmc.ThemeIcon(
            size="lg",
            color=Config.difficulty_colors[article_label],
            variant="filled",
            children=DashIconify(icon="carbon:document", width=25),
        )
        accordion_title_string = dash.html.B(f"Original text")
    else:
        # TODO Add mistral icon
        accordion_title_icon = dmc.ThemeIcon(
            size="lg",
            color=Config.difficulty_colors[article_label],
            variant="filled",
            children=DashIconify(icon="ri:openai-fill", width=25),
        )
        accordion_title_string = dash.html.B(f"Simplification n°{simplification_id}")
    accordion_title = dmc.Group(
        [
            accordion_title_icon,
            accordion_title_string,
            accordion_title_badge,
        ],
        position="space-around",
        spacing="lg",
    )

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
    if simplification_id == "0":
        article = ArticleDatabase.get_article(int(article_id))
    else:
        article = ArticleDatabase.get_simplification(
            article_id=int(article_id), simplification_id=int(simplification_id) - 1
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


# TODO https://dash.plotly.com/background-callbacks#example-4:-progress-bar
@dash.callback(
    dash.dependencies.Output("article-accordion", "children"),
    dash.dependencies.Output("article-accordion", "value"),
    dash.dependencies.Output("openai-key-modal", "opened"),
    dash.dependencies.Input("simplify-button", "n_clicks"),
    dash.dependencies.Input("model-button", "children"),
    dash.dependencies.Input("model-button", "n_clicks"),
    dash.dependencies.Input("submit-openai-key-button", "n_clicks"),
    dash.dependencies.State("article-accordion", "value"),
    dash.dependencies.State("article-accordion", "children"),
    dash.dependencies.State("openai-key-input", "value"),
    prevent_initial_call=True,
    background=True,
    running=[
        (dash.dependencies.Output("simplify-button", "disabled"), True, False),
        (
            dash.dependencies.Output("simplification-in-progress", "style"),
            {"display": "block"},
            {"display": "none"},
        ),
    ],
    progress=[
        dash.dependencies.Output("simplification-progress-bar", "value"),
    ],
)
def call_simplify_text(
    set_progress,
    simplify_button_n_clicks,
    model_button_children,
    model_button_n_clicks,
    submit_openai_key_n_clicks,
    accordion_value,
    accordion_children,
    openai_key,
):
    if simplify_button_n_clicks is not None or model_button_n_clicks is not None:
        # Get infos from accordion value
        if accordion_value is None or (
            simplify_button_n_clicks is None and model_button_n_clicks is None
        ):
            return dash.no_update, dash.no_update, False
        article_id, article_label, simplification_id = accordion_value.split(":")

        # Get current position in accordion
        current_position = int(simplification_id)
        # Return next already loaded simplification
        if current_position + 1 < len(accordion_children):
            return (
                accordion_children,
                accordion_children[current_position + 1]["props"]["value"],
                False,
            )
        # Create new simplification and return it
        else:
            # Determine the model to use
            if simplify_button_n_clicks:
                model_name = model_button_children
            else:
                model_name = None

            # Create new simplification
            try:
                simplification = ArticleDatabase.get_simplification(
                    article_id=int(article_id),
                    simplification_id=current_position,
                    model_to_use=model_name,
                    set_progress=set_progress,
                )
            except OpenAIKeyNotSet as e:
                logger.warning(e)
                return (
                    accordion_children,
                    accordion_children[current_position]["props"]["value"],
                    True,
                )
            # Update accordion
            simplification_label = max(
                simplification,
                key=lambda x: (
                    simplification[x]
                    if x in ["A1", "A2", "B1", "B2", "C1", "C2"]
                    else -1
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
                False,
            )
    elif submit_openai_key_n_clicks is not None:
        if openai_key is not None:
            Models.connect_to_openai(openai_key)
        return dash.no_update, dash.no_update, False
    else:
        return dash.no_update, dash.no_update, dash.no_update


# ---------------------------------------------------------------------------- #
#                                 REGISTER PAGE                                #
# ---------------------------------------------------------------------------- #

dash.register_page(__name__, path_template="/article/<article_id>")
