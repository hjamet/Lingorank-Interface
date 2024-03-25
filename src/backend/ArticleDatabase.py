import logging
import os
import re
from io import BytesIO
from urllib.parse import urljoin

import bs4
import git
import markdownify as md
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image


# Find pwd
pwd = git.Repo(os.getcwd(), search_parent_directories=True).working_tree_dir

# Path to the database
path = os.path.join(pwd, "data/articles.json")

# Create the database if it doesn't exist
if not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        pd.DataFrame(columns=["url", "title", "image", "description", "text"]).to_json(
            f, orient="records"
        )


def add_article_from_url(url: str):
    # Check if url not already in database
    df = pd.read_json(path)
    if not df.empty and url in df["url"].values:
        logging.warning(f"The url {url} is already in the database.")
        return

    # Get the page
    try:
        page = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True
        )
    except:
        logging.error(f"Could not get the page at {url}")
        return

    # Parse the page
    soup = BeautifulSoup(page.text, "html.parser")

    # Make the links absolute
    soup = __make_links_absolute(soup, url)

    # Get the title
    try:
        title = soup.title.string
    except:
        logging.warning(f"Could not get the title at {url}")
        # Default title (the domain name)
        title = url.split("//")[0]

    # Get the image
    try:
        image = __get_largest_image_url(soup)
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
    text = __extract_mardown(soup, url)

    # Add the article
    __add_article(url, title, image, description, text)


def get_article(article_id: int):
    # TODO : Add pdoc3
    df = pd.read_json(path)

    try:
        return df[df["article_id"] == article_id].iloc[0].to_dict()
    except:
        logging.error(f"The article with id {article_id} does not exist.")


def __add_article(url: str, title: str, image: str, description: str, text: str):
    """Add an article to the database.

    Args:
        url (str): The url of the article.
        title (str): The title of the article.
        image (str): The image of the article.
        description (str): The description of the article.
        text (str): The text of the article.
    """
    # Load the database
    df = pd.read_json(path)

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
    df.to_json(path, orient="records")

    # Check if the database is not corrupted
    try:
        pd.read_json(path)
    except:
        logging.error("The database is corrupted.")

    return


def __make_links_absolute(soup, base_url):
    # Modifier les liens relatifs dans les balises <a>
    for a_tag in soup.find_all("a", href=True):
        a_tag["href"] = urljoin(base_url, a_tag["href"])

    # Modifier les liens relatifs dans les balises <img>
    for img_tag in soup.find_all("img", src=True):
        img_tag["src"] = urljoin(base_url, img_tag["src"])

    # Modifier les liens relatifs dans les balises <link>
    for link_tag in soup.find_all("link", href=True):
        link_tag["href"] = urljoin(base_url, link_tag["href"])

    # Modifier les liens relatifs dans les balises <script>
    for script_tag in soup.find_all("script", src=True):
        script_tag["src"] = urljoin(base_url, script_tag["src"])

    return soup


def __extract_mardown(soup: BeautifulSoup, url: str):
    """Convert a soup to markdown.

    Args:
        soup (BeautifulSoup): The soup to convert.
        url (str): The url of the page.

    Returns:
        str: The markdown.
    """
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

    # Convert to markdown
    markdown_content = md.MarkdownConverter().convert_soup(soup)

    return markdown_content


def __get_largest_image_url(soup: BeautifulSoup):
    """Get the largest image url from a soup.

    Args:
        soup (BeautifulSoup): The soup.

    Returns:
        str: The largest image url.
    """
    # Trouver toutes les balises <img> dans la soup
    img_tags = soup.find_all("img")

    largest_image_url = None
    largest_image_width = 0

    for img in img_tags:
        try:
            # Récupérer les dimensions de l'image si disponibles
            try:
                width = int(img.get("width", 0))
            except:
                width = 0
            if width == 0:
                try:
                    response = requests.get(img["src"])
                    image = Image.open(BytesIO(response.content))
                    width = image.size[0]
                except:
                    continue

            # Vérifier si c'est la plus grande image jusqu'à présent
            if width > largest_image_width:
                largest_image_width = width
                largest_image_url = img["src"]
        except (KeyError, ValueError):
            # Ignorer les balises <img> sans attributs width, height ou src valides
            pass

    return largest_image_url


def get_article_database_length():
    return len(pd.read_json(path))
