from typing import List

class Models:
    """A class for inferring the Camembert & Mistral-7B modeles used respectively for estimating the difficulty and simplifying sentences in French.
    """
    
    def __init__(self):
        raise NotImplementedError
    
    def compute_sentences_difficulty(self, sentence: List[str]) -> List[str]:
        """Estimate the difficulty of multiple sentences in French.
        
        Args:
            sentence (List[str]): A list of sentences in French.
            
        Returns:
            List[str]: The estimated difficulties of the sentences.
        """
    
    def simplify_sentences(self, sentence: List[str]) -> List[str]:
        """Simplify multiple sentences in French.
        
        Args:
            sentence (List[str]): A list of sentences in French.
            
        Returns:
            List[str]: The simplified sentences.
        """
        raise NotImplementedError