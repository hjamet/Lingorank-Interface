import dash
import dash_mantine_components as dmc
import logging

from src.backend.ArticleDatabase import ArticleDatabase


class ArticlePage:

    def __init__(self):
        """A class to display an article."""
        self.article_database = ArticleDatabase()

    def get_layout(self, article_id: int) -> str:
        """Get the layout of the article page.

        Args:
            article (dict): The article to display.

        Returns:
            str: The layout of the article page.
        """
        # Get the article
        article = self.article_database.get_article(article_id)
        if article is None:
            logging.error(f"The article with id {article_id} does not exist.")
            return dmc.Text(children="The article does not exist.")

        # Title
        title = dmc.Title(children=article["title"], order=1)

        # Image
        image = dmc.Center(
            dmc.Image(
                src=article["image"], alt=article["title"], width="50%", height="50%"
            )
        )

        # Text
        text = dash.dcc.Markdown(
            children=article["text"],
            id="article-text",
        )

        # Container
        layout = dmc.Container(
            children=[title, image, text],
            id="article-layout",
            size="80%",
        )

        return layout
