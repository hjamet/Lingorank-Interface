from typing import List
import os
import git
from transformers import pipeline
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import CamembertTokenizer


class Models:
    """A class for inferring the Camembert & Mistral-7B modeles used respectively for estimating the difficulty and simplifying sentences in French."""

    def __init__(self):
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
        tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
        self.difficulty_estimation_pipeline = pipeline(
            "text-classification",
            model="OloriBern/Lingorank_Bert_french_difficulty",
            device=self.device,
            tokenizer=tokenizer,
        )

        # Load the sentence simplification model
        # TODO cpu must be used for the model
        config = PeftConfig.from_pretrained(
            "OloriBern/Mistral-7B-French-Simplification"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "bofenghuang/vigostral-7b-chat",
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model, "OloriBern/Mistral-7B-French-Simplification", config=config
        )

        print("Models loaded successfully.")

    def compute_sentences_difficulty(self, sentence: List[str]) -> List[str]:
        """Estimate the difficulty of multiple sentences in French.

        Args:
            sentence (List[str]): A list of sentences in French.

        Returns:
            List[str]: The estimated difficulties of the sentences.
        """
        temp = self.difficulty_estimation_pipeline(sentence)
        print(temp)

    def simplify_sentences(self, sentence: List[str]) -> List[str]:
        """Simplify multiple sentences in French.

        Args:
            sentence (List[str]): A list of sentences in French.

        Returns:
            List[str]: The simplified sentences.
        """
        raise NotImplementedError


if __name__ == "__main__":
    models = Models()

    # Test the difficulty estimation model
    sentence = ["Le chat est sur le tapis.", "La voiture est dans le garage."]
    print(models.compute_sentences_difficulty(sentence))
