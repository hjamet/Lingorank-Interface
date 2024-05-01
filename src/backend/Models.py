import json
import os
from collections import namedtuple
from typing import List

import nltk.data
import openai
import pandas as pd
import torch
from datasets import Dataset
from openai import OpenAI
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm as console_tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CamembertTokenizer,
    pipeline,
)

import src.Config as Config
from src.Exceptions import OpenAIKeyNotSet
from src.Logger import logger

openai_models_path = os.path.join(Config.pwd, "data", "models")


def compute_text_difficulty(text: str):
    """Compute the average difficulty of text by analyzing each sentence.

    Args:
        text (str): The text to analyze, expected in French.

    Returns:
        list: A list containing the average difficulty score for each difficulty level. (A1, A2, B1, B2, C1, C2)
    """
    # Split by sentences
    tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
    sentences = tokenizer.tokenize(text)

    # Remove too long sentences
    max_tokens = difficulty_estimation_pipeline.tokenizer.model_max_length
    short_sentences = [
        sentence
        for sentence in sentences
        if len(difficulty_estimation_pipeline.tokenizer.encode(sentence)) <= max_tokens
    ]

    # Compute difficulty of each sentence
    sentence_difficulties = __compute_sentences_difficulty(short_sentences)

    # Compute average difficulty per column
    average_difficulty = [sum(col) / len(col) for col in zip(*sentence_difficulties)]

    return average_difficulty


def simplify_text(text: str, model_to_use: str = "mistral-7B"):
    """Simplify a text by simplifying each sentence.

    Args:
        text (str): The text to simplify, expected in French.
        model_to_use (str, optional): The model to use for simplification. Can be either "mistral-7B" or "gpt-3.5-turbo-1106" Defaults to "mistral".

    Returns:
        str: The simplified text.

    Raises:
        OpenAIKeyNotSet: If the openai key is not found.
    """
    # Split by sentences
    tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
    sentences = tokenizer.tokenize(text)

    # Compute simplified sentences
    simplified_sentences = __simplify_sentences(sentences, model_to_use)

    return " ".join(simplified_sentences)


def list_available_models():
    models = {}

    # OpenAI models
    for file in os.listdir(openai_models_path):
        model_json = json.load(open(os.path.join(openai_models_path, file)))
        models[model_json["model"]["model"]] = namedtuple(
            "Model", ["model", "fine_tuned_model"]
        )(model_json["model"]["model"], model_json["model"]["fine_tuned_model"])

    return models


def connect_to_openai(openai_key: str = None):
    """Connect to OpenAI API.

    Args:
        openai_key (str, optional): The OpenAI key to use. If None, the key is read from the .openai_key file. Defaults to None.

    Returns:
        OpenAI: The OpenAI client.

    Raises:
        OpenAIKeyNotSet: If the OpenAI key is not found.
    """
    if openai_key is not None:
        with open(os.path.join(Config.pwd, "scratch", ".openai_key"), "w") as f:
            f.write(openai_key)
    try:
        with open(os.path.join(Config.pwd, "scratch", ".openai_key"), "r") as f:
            openai_key = f.read()
            return OpenAI(
                api_key=openai_key,
            )
    except:
        raise OpenAIKeyNotSet(
            "OpenAI key not found. Please provide it in .openai_key file."
        )


# ---------------------------------------------------------------------------- #
#                               PRIVATE FUNCTIONS                              #
# ---------------------------------------------------------------------------- #


def __compute_sentences_difficulty(sentences: List[str]):
    """Estimate the difficulty of multiple sentences in French.

    Args:
        sentence (List[str]): A list of sentences in French.

    Returns:
        List[List [float]]: The probabilities of each label for each sentence. (A1, A2, B1, B2, C1, C2)
    """
    predictions_logits = difficulty_estimation_pipeline(sentences)

    # Rename labels and prepare the output
    results = [
        [l["score"] for l in sorted(sentence, key=lambda l: l["label"])]
        for sentence in predictions_logits
    ]
    return results


def __simplify_sentences(sentences: List[str], model: str = "mistral-7B"):
    """Simplify multiple sentences in French.

    Args:
        sentence (List[str]): A list of sentences in French.
        model (str, optional): The model to use for simplification. Can be either "mistral-7B" or an openai model id. Defaults to "mistral-7B".

    Returns:
        List[str]: The simplified sentences.

    Raises:
        OpenAIKeyNotSet: If the openai key is not found.
    """
    # Estimate difficulty
    inputs = pd.DataFrame(columns=["Sentence", "Difficulty"])
    inputs["Sentence"] = sentences
    inputs["Difficulty"] = pd.Series(__compute_sentences_difficulty(sentences))
    inputs["Difficulty"] = inputs["Difficulty"].apply(
        lambda x: ["A1", "A2", "B1", "B2", "C1", "C2"][x.index(max(x))]
    )

    # Format data
    inputs = __format_data_mistral(inputs)

    if "gpt-3.5-turbo-1106" in model:
        predictions = evaluate_openai(inputs["formatted_chat"], model, "")
        predictions_series = predictions
    elif model == "mistral-7B":
        # Encode dataset
        encoded_dataset = __encode_dataset(inputs, mistral_tokenizer)

        # Simplify sentences
        test_loader = DataLoader(encoded_dataset, batch_size=16)

        # Generate predictions
        with torch.no_grad():
            mistral_model.eval()
            predictions_ids = []

            for batch in console_tqdm(test_loader):
                input_ids_batch = batch["input_ids"].to("cpu")
                attention_mask_batch = batch["attention_mask"].to("cpu")

                outputs = mistral_model.generate(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    max_length=max(128, input_ids_batch.shape[1] * 2),
                    num_return_sequences=1,
                )

                predictions_ids.extend(outputs)
            predictions = [
                mistral_tokenizer.decode(prediction, skip_special_tokens=True)
                for prediction in predictions_ids
            ]
            predictions_series = pd.Series(predictions)
    else:
        raise OpenAIKeyNotSet(f"Invalid model name {model}.")

    return predictions_series


def __download_tokenizer(
    model_name: str = "bofenghuang/vigostral-7b-chat", training: bool = False
):
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        truncation_side="left",
        add_eos_token=training,
        add_bos_token=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def __format_data_mistral(
    df: pd.DataFrame,
):
    # Create conversation
    logger.info("Create conversation...")

    def create_conversation(row):
        conversation = [
            {
                "role": "system",
                "content": "Vous êtes un modèle de langage naturel capable de simplifier des phrases en français. La phrase simplifiée doit avoir un sens aussi proche que possible de la phrase originale, mais elle est d'un niveau inférieur du CECRL et donc plus facile à comprendre. Par exemple, si une phrase est au niveau C1 du CECRL, simplifiez-la en B2. Si elle se situe au niveau B2, simplifiez-la en B1. Si elle se situe au niveau B1, simplifiez-la en A2. Si le niveau A2 est atteint, simplifiez en A1.",
            }
        ]
        reduced_difficulty = {
            "A1": "A1",
            "A2": "A1",
            "B1": "A2",
            "B2": "B1",
            "C1": "B2",
            "C2": "C1",
            "level1": "level1",
            "level2": "level1",
            "level3": "level2",
            "level4": "level3",
        }
        conversation.append(
            {
                "role": "user",
                "content": f"""Voici une phrase en français de niveau {row['Difficulty']} à simplifier :
                \"\"\"{row['Sentence']}\"\"\"
                Donne moi une phrase simplifiée au niveau {reduced_difficulty[row['Difficulty']]} tout en conservant au maximum son sens original
                """,
            }
        )

        return conversation

    # Create dataset
    logger.info("Create dataset...")
    conversation_list = (
        df.reset_index()
        .apply(create_conversation, axis=1)
        .rename("conversation")
        .to_list()
    )
    dataset = Dataset.from_dict({"chat": conversation_list})

    # Format dataset
    logger.info("Format dataset...")
    formatted_dataset = dataset.map(
        lambda x: {
            "formatted_chat": mistral_tokenizer.apply_chat_template(
                x["chat"], tokenize=False, add_generation_prompt=True
            )
        }
    )

    return formatted_dataset


def __encode_dataset(dataset: Dataset, tokenizer: AutoTokenizer):
    # Determine max length
    logger.info("Determine max length...")
    max_length = max(
        [
            len(tokenizer.encode(chat))
            for chat in console_tqdm(dataset["formatted_chat"])
        ]
    )

    # Encode dataset
    logger.info("Encode dataset...")
    encoded_dataset = dataset.map(
        lambda x: tokenizer(
            x["formatted_chat"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        ),
        batched=True,
    )

    # Create labels
    logger.info("Create labels...")
    encoded_dataset = encoded_dataset.map(
        lambda x: {
            "labels": x["input_ids"],
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
        },
        batched=True,
    )

    # Create dataset ready for training
    logger.info("Create dataset ready for training...")
    encoded_dataset = Dataset.from_dict(
        {
            "input_ids": torch.tensor(encoded_dataset["input_ids"]),
            "attention_mask": torch.tensor(encoded_dataset["attention_mask"]),
            "labels": torch.tensor(encoded_dataset["labels"]),
        }
    )

    # Set format
    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return encoded_dataset


def evaluate_openai(inputs: pd.Series, model: str, context: str):
    # Connect to OpenAI
    client = connect_to_openai()

    # Compute predictions
    predictions = []
    for text in console_tqdm(inputs):
        try:
            if "gpt" in model:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": text},
                    ],
                    max_tokens=len(text) * 2,
                )
                prediction = response.choices[0].message.content.strip()
            else:
                # Todo: Update to use the new API
                response = client.Completion.create(
                    engine=model,
                    prompt=f"{context}{text}",
                    max_tokens=len(text) * 2,
                )
                prediction = response.choices[0].text.strip()
        except Exception as e:
            print(e)
            print(f"Error with text: {text}")
            print("Skipping...")
            predictions.append("Error")
            continue

        predictions.append(prediction)
        # Save prediction for security
        pd.DataFrame(predictions).to_csv(
            os.path.join(Config.pwd, "scratch", "openai_predictions.csv"), index=False
        )

    return pd.Series(predictions)


# ---------------------------------------------------------------------------- #
#                                INITIALIZATION                                #
# ---------------------------------------------------------------------------- #

# Create scratch directory
scratch_path = os.path.join(Config.pwd, "scratch", "backend", "models")
if not os.path.exists(scratch_path):
    os.makedirs(scratch_path)

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the difficulty estimation model
if Config.difficulty_estimation:
    bert_tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
    difficulty_estimation_pipeline = pipeline(
        "text-classification",
        model="OloriBern/Lingorank_Bert_french_difficulty",
        device=device,
        tokenizer=bert_tokenizer,
        top_k=None,
    )

    # Set max token length
    difficulty_estimation_pipeline.tokenizer.model_max_length = (
        difficulty_estimation_pipeline.model.config.max_position_embeddings
    )

# Load the sentence simplification model
if Config.simplification_with_mistral:
    ## Add quantization
    kwargs = {
        "pretrained_model_name_or_path": "bofenghuang/vigostral-7b-chat",
        "trust_remote_code": True,
    }
    if Config.quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["quantization_config"] = bnb_config
        kwargs["torch_dtype"] = torch.bfloat16
    ## Load PeftModel
    config = PeftConfig.from_pretrained("OloriBern/Mistral-7B-French-Simplification")
    mistral_model = AutoModelForCausalLM.from_pretrained(**kwargs)
    mistral_model = PeftModel.from_pretrained(
        mistral_model,
        "OloriBern/Mistral-7B-French-Simplification",
        config=config,
    )

## Load Tokenizer
mistral_tokenizer = __download_tokenizer()

# Load the OpenAI models
# connect_to_openai()

print("Models loaded successfully.")

# ---------------------------------------------------------------------------- #
#                                     TESTS                                    #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Test the difficulty estimation model
    sentence = [
        "Le chat est sur le tapis.",
        "A n'en pas douter, la voiture peut certainement être située dans le garage !",
    ]
    print(__compute_sentences_difficulty(sentence))
    print(__simplify_sentences(sentence))
