import os
import pandas as pd
import git


class ArticleDatabase:
    def __init__(self):
        """Une classe pour gérer la base de données des articles.
        Pour le moment, les articles sont sauvegardés dans un fichier csv.
        """
        # Find pwd
        pwd = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir

        # Path to the database
        self.path = os.path.join(pwd, "data/articles.csv")

        # Create the database if it doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                pd.DataFrame(
                    columns=["url", "title", "image", "description", "text"]
                ).to_csv(f, index=False)

    # ---------------------------------------------------------------------------- #
    #                                PUBLIC METHODS                                #
    # ---------------------------------------------------------------------------- #

    def add_article_from_url(self, url: str):
        raise NotImplementedError

    # ---------------------------------------------------------------------------- #
    #                                PRIVATE METHODS                               #
    # ---------------------------------------------------------------------------- #

    def __add_article(
        self, url: str, title: str, image: str, description: str, text: str
    ):
        """Add an article to the database.

        Args:
            url (str): The url of the article.
            title (str): The title of the article.
            image (str): The image of the article.
            description (str): The description of the article.
            text (str): The text of the article.
        """
        # Load the database
        df = pd.read_csv(self.path)

        # Add the article
        df = df.append(
            {
                "url": url,
                "title": title,
                "image": image,
                "description": description,
                "text": text,
            },
            ignore_index=True,
        )

        # Save the database
        with open(self.path, "w") as f:
            df.to_csv(f, index=False)
