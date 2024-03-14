import dash
import dash_mantine_components as dmc


class ArticlePage:

    def __init__(self):
        """A class to display an article."""

    def get_layout(self, article: dict) -> str:
        """Get the layout of the article page.

        Args:
            article (dict): The article to display.

        Returns:
            str: The layout of the article page.
        """
        # Title
        title = dmc.Title(children=article["title"], order=1)

        # Image
        image = dmc.Image(src=article["image"], alt=article["title"], order=2)

        # Description
        description = dmc.Text(children=article["description"], order=3)

        # Text
        text = dmc.Text(children=article["text"], order=4)

        # Create the layout
        layout = dmc.Container(
            children=[title, image, description, text], id="article-layout", size="80%"
        )

        return layout
