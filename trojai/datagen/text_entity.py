# Import packages
from abc import abstractmethod
from pyllist import dllist
import re
from nltk import sent_tokenize

from trojai.datagen.entity import Entity


# Abstract class definition
class TextEntity(Entity):
    @abstractmethod
    def get_delimiters(self):
        pass

    @abstractmethod
    def get_text(self):
        pass

    @abstractmethod
    def __deepcopy__(self, memo):
        pass


# Generic class definition
class GenericTextEntity(TextEntity):
    """
    A class to instantiate an TextEntity for text.
    """

    def __init__(self, input_text: str):
        # Regular expressions
        delimiter_pattern = r'[,:;-]'
        delimiter_regex = re.compile(delimiter_pattern)
        # Create data structures
        self.data = dllist()
        self.delimiters = dllist()
        # Break apart by sentence
        sentences = sent_tokenize(input_text)
        for sentence in sentences:
            # Add a new linked list for the sentence and delimiters
            cur_delimiters = dllist()
            cur_sentence = dllist()
            # Split by whitespace
            t_sentence = sentence.split()
            [cur_sentence.append(word) for word in t_sentence]
            [cur_delimiters.append([index, delimiter_regex.search(word).group(0)])
             for index, word in enumerate(t_sentence) if delimiter_regex.search(word)]
            # Add it in
            self.delimiters.append(cur_delimiters)
            self.data.append(cur_sentence)

    def get_data(self):
        return self.data
    
    def get_delimiters(self):
        return self.delimiters

    def get_text(self):
        # We need to reconstruct the system from the DLL
        return " ".join([" ".join([node for node in sentence]) for sentence in self.data])

    def __deepcopy__(self, memo):
        return GenericTextEntity( self.get_text() )
