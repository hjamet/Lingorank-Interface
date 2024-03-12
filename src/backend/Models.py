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


class Models:
    """A class for inferring the Camembert & Mistral-7B modeles used respectively for estimating the difficulty and simplifying sentences in French."""

    def __init__(self, quantization: bool = False):
        """Initialize the models. This class will handle inference of the Camembert & Mistral-7B models used respectively for estimating the difficulty and simplifying sentences in French.

        Args:
            quantization (bool, optional): Whether to use quantization for the Mistral-7B model. Defaults to False.
        """
        # Define PWD as the current git repository
        repo = git.Repo(".", search_parent_directories=True)
        self.pwd = repo.working_dir

        # Create scratch directory
        self.scratch_path = os.path.join(self.pwd, "scratch", "backend", "models")
        if not os.path.exists(self.scratch_path):
            os.makedirs(self.scratch_path)

        # Define device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the difficulty estimation model
        self.bert_tokenizer = CamembertTokenizer.from_pretrained(
            "camembert/camembert-base"
        )
        self.difficulty_estimation_pipeline = pipeline(
            "text-classification",
            model="OloriBern/Lingorank_Bert_french_difficulty",
            device=self.device,
            tokenizer=self.bert_tokenizer,
        )

        # Load the sentence simplification model
        # TODO Acheter DDR5 30GO -> 64GO
        ## Add quantization
        kwargs = {
            "pretrained_model_name_or_path": "bofenghuang/vigostral-7b-chat",
            "trust_remote_code": True,
        }
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = bnb_config
            kwargs["torch_dtype"] = torch.bfloat16
        ## Load PeftModel
        config = PeftConfig.from_pretrained(
            "OloriBern/Mistral-7B-French-Simplification"
        )
        mistral_model = AutoModelForCausalLM.from_pretrained(**kwargs)
        mistral_model = PeftModel.from_pretrained(
            mistral_model,
            "OloriBern/Mistral-7B-French-Simplification",
            config=config,
        )
        ## Load Tokenizer
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(
            "bofenghuang/vigostral-7b-chat",
            padding_side="left",
            truncation_side="left",
            add_eos_token=False,
            add_bos_token=True,
            trust_remote_code=True,
        )
        self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
        ## Create pipeline
        self.simplification_pipeline = pipeline(
            "text-generation",
            model=mistral_model,
            tokenizer=self.mistral_tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        print("Models loaded successfully.")

    def compute_sentences_difficulty(self, sentences: List[str]):
        """Estimate the difficulty of multiple sentences in French.

        Args:
            sentence (List[str]): A list of sentences in French.

        Returns:
            List[str]: The estimated difficulties of the sentences.
        """
        predictions = self.difficulty_estimation_pipeline(sentences)
        labels_map = {
            "LABEL_0": "A1",
            "LABEL_1": "A2",
            "LABEL_2": "B1",
            "LABEL_3": "B2",
            "LABEL_4": "C1",
            "LABEL_5": "C2",
        }
        return [labels_map[prediction["label"]] for prediction in predictions]

    def simplify_sentences(self, sentences: List[str]):
        """Simplify multiple sentences in French.

        Args:
            sentence (List[str]): A list of sentences in French.

        Returns:
            List[str]: The simplified sentences.
        """
        # TODO: check if I am well using my mistral and not vigostral, because results are not good
        inputs = pd.DataFrame(columns=["Sentence", "Difficulty"])
        inputs["Sentence"] = sentences
        inputs["Difficulty"] = pd.Series(self.compute_sentences_difficulty(sentences))
        inputs = self.__format_data_mistral(inputs)["formatted_chat"]
        predictions = self.simplification_pipeline(
            sentences, max_new_tokens=2 * max(map(len, inputs))
        )
        # TODO: sentences generated are too long
        generated_sentences = [
            prediction[0]["generated_text"] for prediction in predictions
        ]
        return generated_sentences

    def __format_data_mistral(
        self,
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
                "formatted_chat": self.mistral_tokenizer.apply_chat_template(
                    x["chat"], tokenize=False, add_generation_prompt=True
                )
            }
        )

        return formatted_dataset


if __name__ == "__main__":
    models = Models(quantization=True)

    # Test the difficulty estimation model
    sentence = [
        "Le chat est sur le tapis.",
        "A n'en pas douter, la voiture peut certainement être située dans le garage !",
    ]
    print(models.compute_sentences_difficulty(sentence))
    print(models.simplify_sentences(sentence))
