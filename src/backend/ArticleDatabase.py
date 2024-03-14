import os
import pandas as pd
import git
import requests
from bs4 import BeautifulSoup
import markdownify as md
import logging


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
        """This method will read an article from a url and add it to the database.
        The method will try to extract the title, image, description and text of the article.

        Args:
            url (str): The url of the article.
        """
        # Check if url not already in database
        df = pd.read_csv(self.path)
        if url in df["url"].values:
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
        text = md.MarkdownConverter().convert_soup(soup)

        # Add the article
        self.__add_article(url, title, image, description, text)

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
