from src.Logger import logger
import os
import re
from io import BytesIO
from urllib.parse import urljoin

import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from PIL import Image

import src.backend.Models as Models
import src.Config as Config

# Path to the databases
article_database_path = os.path.join(Config.pwd, "data/articles.json")
simplication_database_path = os.path.join(Config.pwd, "data/simplifications.json")

# Create the database if it doesn't exist
if not os.path.exists(article_database_path):
    os.makedirs(os.path.dirname(article_database_path), exist_ok=True)
    with open(article_database_path, "w") as f:
        pd.DataFrame(columns=["url", "title", "image", "description", "text"]).to_json(
            f, orient="records"
        )


def add_article_from_url(url: str):
    # Check if url not already in database
    df = pd.read_json(article_database_path)
    if not df.empty and url in df["url"].values:
        logger.warning(f"The url {url} is already in the database.")
        return

    # Get the page
    try:
        page = requests.get(
            url, headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True
        )
    except:
        logger.error(f"Could not get the page at {url}")
        return

    # Parse the page
    soup = BeautifulSoup(page.text, "html.parser")

    # Make the links absolute
    soup = __make_links_absolute(soup, url)

    # Parse the soup
    parsed = trafilatura.bare_extraction(
        soup.prettify(),
        url=url,
        include_formatting=True,
        include_images=True,
        include_links=True,
        favor_precision=True,
    )
    if parsed is None:
        logger.error(f"Could not parse the page at {url}")
        return

    # Get the title
    if parsed["title"] is not None:
        title = parsed["title"]
    else:
        logger.warning(f"Could not get the title at {url}")
        # Default title (the domain name)
        title = url.split("//")[0]

    # Get the image
    if parsed["image"] is not None:
        image = parsed["image"]
    else:
        try:
            image = __get_largest_image_url(soup)
        except:
            logger.warning(f"Could not get the image at {url}")
            # Default image
            image = "https://static.thenounproject.com/png/2684410-200.png"

    # Get the description
    if parsed["description"] is not None:
        description = parsed["description"]
    else:
        logger.warning(f"Could not get the description at {url}")
        # Default description
        description = "No description"

    text = parsed["text"]
    text = move_images_to_end_of_paragraphs(text)

    # # Compute difficulty
    difficulty_list = Models.compute_text_difficulty(text)

    # Add the article
    __add_article(url, title, image, description, text, difficulty_list)


def add_article_from_text(text: str):
    """Add an article from text to the database.

    Args:
        text (str): The text of the article.
    """
    # Create metadata
    title = text
    image = "https://cdn-icons-png.flaticon.com/512/2911/2911230.png"
    description = text
    difficulty_list = Models.compute_text_difficulty(text)

    __add_article("text", title, image, description, text, difficulty_list)


def get_article(article_id: int):
    """Get an article from the database.

    Args:
        article_id (int): The id of the article.

    Returns:
        dict: The article.
    """
    try:
        df = pd.read_json(article_database_path)
    except:
        logger.error("The database is corrupted or empty.")
        return

    try:
        return df[df["article_id"] == article_id].iloc[0].to_dict()
    except:
        logger.error(f"The article with id {article_id} does not exist.")


def get_simplification(
    article_id: int, simplification_id: int, model_to_use: str = None
):
    """Get a simplification version of the article.

    Args:
        article (int): The id of the article.
        simplification_id (int): The id of the simplification. (The higher the id, the more simplified the text)
        model_to_use (str): The model to use for the simplification. Default is None. Possible values:
            - "model id" (str): for openai models

    Returns:
        dict: The simplified article.

    Raises:
        OpenAIKeyNotSet (Exception): If the OpenAI key is not set.
    """
    # Load the database
    try:
        df = pd.read_json(simplication_database_path)
    except:
        logger.warning("The database is corrupted or empty. Creating it.")
        df = pd.DataFrame(columns=["article_id", "text"])

    # Get the article
    article = get_article(article_id)

    # Get the simplification
    try:
        article_simplifications = df[(df["article_id"] == article["article_id"])]
        return article_simplifications.iloc[simplification_id].to_dict()
    except:
        logger.info(
            f"The simplification with id {simplification_id} does not exist. Creating it."
        )
        # TODO Link Simplification using model
        if model_to_use is not None:
            model_to_use = list(Models.list_available_models().values())[
                0
            ].fine_tuned_model
        else:
            model_to_use = Models.list_available_models()[model_to_use].fine_tuned_model

        text = Models.simplify_text(
            text=article["text"],
            model_to_use=model_to_use,
        )

        difficulty_list = Models.compute_text_difficulty(text)

        __add_simplification(article=article, text=text, difficulty=difficulty_list)

        return get_simplification(article_id, simplification_id)


def __add_article(
    url: str, title: str, image: str, description: str, text: str, difficulty: list
):
    """Add an article to the database.

    Args:
        url (str): The url of the article.
        title (str): The title of the article.
        image (str): The image of the article.
        description (str): The description of the article.
        text (str): The text of the article.
        difficulty (list): The mean difficulty of the article for every label (A1, A2, B1, B2, C1, C2)
    """
    # Load the database
    df = pd.read_json(article_database_path)

    # Add the article
    article = {
        "article_id": len(df),
        "url": [url],
        "title": [
            title[: min(len(title), Config.title_length)]
            + ("..." if len(title) > Config.title_length else "")
        ],
        "image": [image],
        "description": [
            description[: min(len(description), Config.description_length)]
            + ("..." if len(description) > Config.description_length else "")
        ],
        "text": [text],
    }
    article.update(
        {["A1", "A2", "B1", "B2", "C1", "C2"][i]: [difficulty[i]] for i in range(6)}
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(article),
        ]
    )

    # Save the database
    df.to_json(article_database_path, orient="records")

    # Check if the database is not corrupted
    try:
        pd.read_json(article_database_path)
    except:
        logger.error("The database is corrupted.")

    return


def __add_simplification(article: dict, text: str, difficulty: list):
    """Add a simplification to the database.

    Args:
        article_id (dict): The article to simplify.
        text (str): The text of the simplification.
        difficulty (list): The mean difficulty of the simplification for every label (A1, A2, B1, B2, C1, C2)
    """
    # Load the database
    try:
        df = pd.read_json(simplication_database_path)
    except:
        logger.warning("The database is corrupted or empty. Creating it.")
        df = pd.DataFrame(columns=["article_id", "text"])

    # Add the simplification
    simplification = {
        "article_id": [article["article_id"]],
        "text": [text],
        "title": [article["title"]],
    }
    simplification.update(
        {["A1", "A2", "B1", "B2", "C1", "C2"][i]: [difficulty[i]] for i in range(6)}
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(simplification),
        ]
    )

    # Save the database
    df.to_json(simplication_database_path, orient="records")

    # Check if the database is not corrupted
    try:
        pd.read_json(simplication_database_path)
    except:
        logger.error("The database is corrupted.")

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


def move_images_to_end_of_paragraphs(markdown_text: str):
    # Regular expression to find Markdown images
    image_regex = r"!\[.*?\]\(.*?\)"

    # Split the text into paragraphs
    split_regex = r"\n{2,}|(?<=\n)#+"
    paragraphs = re.split(split_regex, markdown_text)

    # Initialize a list to store the modified paragraphs
    modified_paragraphs = []

    for paragraph in paragraphs:
        # Find all images in the paragraph
        images = re.findall(image_regex, paragraph)

        # Remove the images from the paragraph
        modified_paragraph = re.sub(image_regex, "", paragraph)

        # Append the images to the end of the paragraph
        modified_paragraph += "\n\n" + "\n\n".join(images)

        # Add the modified paragraph to the list
        modified_paragraphs.append(modified_paragraph)

    # Reconstruct the text with the modified paragraphs
    modified_text = "\n\n".join(modified_paragraphs)

    return modified_text


def get_article_database_length():
    return len(pd.read_json(article_database_path))
