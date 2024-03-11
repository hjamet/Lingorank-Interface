# Define PWD as the current git repository
import git
import os
import sys
repo = git.Repo('.', search_parent_directories=True)
pwd = repo.working_dir
os.chdir(pwd)
sys.path.append(pwd)


import unittest

from src.backend.Models import Models

class TestModels(unittest.TestCase):
    
    models = Models()
    sentences = ["Ceci est un test.", "Il s'agit en réalité ici d'un test autrement plus délicat."]
    
    def test_difficulty_estimation(self):
        # Test with two simple sentences
        try:
            difficulties = self.models.compute_sentence_difficulty(self.sentences)
        except:
            self.fail("compute_sentence_difficulty() raised an exception.")
        
        # Check if difficulty in ["A1", "A2", "B1", "B2", "C1", "C2"]
        for difficulty in difficulties:
            self.assertIn(difficulty, ["A1", "A2", "B1", "B2", "C1", "C2"])
    
    def test_sentence_simplification(self):
        # Test with two complex sentences
        try:
            simplified_sentences = self.models.simplify_sentences(self.sentences)
        except:
            self.fail("simplify_sentences() raised an exception.")
        
        # Check if the simplified sentences are non-empty strings
        for simplified_sentence in simplified_sentences:
            self.assertIsInstance(simplified_sentence, str)
            self.assertNotEqual(simplified_sentence, "")
            
if __name__ == "__main__":
    unittest.main()   
