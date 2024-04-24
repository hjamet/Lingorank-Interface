from typing import List
import os
import git
import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    CamembertTokenizer,
    pipeline,
)
from datasets import Dataset
import pandas as pd
import logging
import torch
from tqdm import tqdm as console_tqdm
from torch.utils.data import DataLoader
import src.Config as Config
import nltk.data


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
    sentence_difficulties = compute_sentences_difficulty(short_sentences)

    # Compute average difficulty per column
    average_difficulty = [sum(col) / len(col) for col in zip(*sentence_difficulties)]

    return average_difficulty


def compute_sentences_difficulty(sentences: List[str]):
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


def simplify_sentences(sentences: List[str], model: str = "mistral-7B"):
    """Simplify multiple sentences in French.

    Args:
        sentence (List[str]): A list of sentences in French.
        model (str, optional): The model to use for simplification. Can be either "mistral-7B" or "gpt-3.5-turbo-1106" Defaults to "mistral".

    Returns:
        List[str]: The simplified sentences.
    """
    # Estimate difficulty
    inputs = pd.DataFrame(columns=["Sentence", "Difficulty"])
    inputs["Sentence"] = sentences
    inputs["Difficulty"] = pd.Series(compute_sentences_difficulty(sentences))

    # Format data
    inputs = __format_data_mistral(inputs)

    if model == "gpt-3.5-turbo-1106":
        pass
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
        raise ValueError(f"Invalid model name {model}.")

    return predictions_series


# ------------------------- PRIVATE MISTRAL FUNCTIONS ------------------------ #


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
    logging.info("Create conversation...")

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
    logging.info("Create dataset...")
    conversation_list = (
        df.reset_index()
        .apply(create_conversation, axis=1)
        .rename("conversation")
        .to_list()
    )
    dataset = Dataset.from_dict({"chat": conversation_list})

    # Format dataset
    logging.info("Format dataset...")
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
    logging.info("Determine max length...")
    max_length = max(
        [
            len(tokenizer.encode(chat))
            for chat in console_tqdm(dataset["formatted_chat"])
        ]
    )

    # Encode dataset
    logging.info("Encode dataset...")
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
    logging.info("Create labels...")
    encoded_dataset = encoded_dataset.map(
        lambda x: {
            "labels": x["input_ids"],
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
        },
        batched=True,
    )

    # Create dataset ready for training
    logging.info("Create dataset ready for training...")
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


# ------------------------- PRIVATE OPENAI FUNCTIONS ------------------------- #

# ----------------------- ZERO-SHOT EVALUATION FUNCTION ---------------------- #
from tqdm import notebook as notebook_tqdm
import openai
import pandas as pd
import signal, time


# Connect to OpenAI
def connect_to_openai():
    try:
        with open(os.path.join(Config.pwd, "scratch", ".openai_key"), "r") as f:
            openai_key = f.read()
            openai.api_key = openai_key
    except:
        key = input("Please enter your OpenAI key: ")
        with open(os.path.join(Config.pwd, "scratch", ".openai_key"), "w") as f:
            f.write(key)
        openai.api_key = key


class Timeout:
    """Timeout class using ALARM signal"""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


def evaluate_openai(inputs: pd.Series, model: str, context: str):
    # Compute predictions
    predictions = []
    for text in notebook_tqdm.tqdm(inputs):
        try:
            with Timeout(15):
                if "gpt" in model:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": text},
                        ],
                        max_tokens=len(text) * 2,
                    )
                    prediction = response.choices[0]["message"]["content"].strip()
                else:
                    response = openai.Completion.create(
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

    return pd.DataFrame(predictions)


# import

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
if Config.simplification:
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
    print(compute_sentences_difficulty(sentence))
    print(simplify_sentences(sentence))
