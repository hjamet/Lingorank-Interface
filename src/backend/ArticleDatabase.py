import os
import pandas as pd
import git
import requests
from bs4 import BeautifulSoup
import bs4
import markdownify as md
import logging


class ArticleDatabase:
    def __init__(self):
        """Une classe pour gérer la base de données des articles.
        Pour le moment, les articles sont sauvegardés dans un fichier json.
        """
        # Find pwd
        pwd = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir

        # Path to the database
        self.path = os.path.join(pwd, "data/articles.json")

        # Create the database if it doesn't exist
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as f:
                pd.DataFrame(
                    columns=["url", "title", "image", "description", "text"]
                ).to_json(f, orient="records")

    # ---------------------------------------------------------------------------- #
    #                                PUBLIC METHODS                                #
    # ---------------------------------------------------------------------------- #

    def add_article_from_url(self, url: str):
        """This method will read an article from a url and add it to the database.
        The method will try to extract the title, image, description and text of the article.

        Args:
            url (str): The url of the article.
        """
        # Check if url not already in database
        df = pd.read_json(self.path)
        if not df.empty and url in df["url"].values:
            logging.warning(f"The url {url} is already in the database.")
            return

        # Get the page
        try:
            page = requests.get(url)
        except:
            logging.error(f"Could not get the page at {url}")

        # Parse the page
        soup = BeautifulSoup(page.text, "html.parser")

        # Get the title
        try:
            title = soup.title.string
        except:
            logging.warning(f"Could not get the title at {url}")
            # Default title (the domain name)
            title = url.split("//")[0]

        # Get the image
        try:
            image = soup.find("img")["src"]
            image = image if image.startswith("http") else url.split("/")[0] + image
        except:
            logging.warning(f"Could not get the image at {url}")
            # Default image
            image = "https://static.thenounproject.com/png/2684410-200.png"

        # Get the description
        try:
            description = soup.find("meta", attrs={"name": "description"})["content"]
        except:
            logging.warning(f"Could not get the description at {url}")
            # Default description
            description = "No description"

        # Get the text
        # Remove meta tags
        for meta in soup.find_all("meta"):
            meta.decompose()
        # Remove script tags
        for script in soup.find_all("script"):
            script.decompose()
        # Remove style tags
        for style in soup.find_all("style"):
            style.decompose()
        # Remove comments
        for comment in soup.find_all(
            text=lambda text: isinstance(text, bs4.element.Comment)
        ):
            comment.extract()
        # Remove hidden inputs
        for hidden in soup.find_all(type="hidden"):
            hidden.decompose()
        # Remove empty tags
        for empty in soup.find_all(lambda tag: not tag.contents):
            empty.decompose()
        text = md.MarkdownConverter().convert_soup(soup)

        # Add the article
        self.__add_article(url, title, image, description, text)

    def get_article(self, article_id: int):
        """Get an article from the database.

        Args:
            article_id (int): The id of the article.

        Returns:
            dict: The article.
        """
        df = pd.read_json(self.path)

        try:
            return df.iloc[article_id].to_dict()
        except:
            logging.error(f"The article with id {article_id} does not exist.")

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
        df = pd.read_json(self.path)

        # Add the article
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "article_id": len(df),
                        "url": [url],
                        "title": [title],
                        "image": [image],
                        "description": [description],
                        "text": [text],
                    }
                ),
            ]
        )

        # Save the database
        df.to_json(self.path, orient="records")
