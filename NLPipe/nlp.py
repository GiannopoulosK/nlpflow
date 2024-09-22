import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from NLPipe.utils.decorators import convert_input_to_string

class InvalidSpacyModelException(Exception):
    pass

# Add: Remove emails, remove mentions(@user),remove hastags, remove_socials_items(All social things together), remove 's first before any words
class NoiseRemover():
    """Remove noise in text with the spaCy NLP models."""
    
    def __init__(self, spacy_model='large'):
        if spacy_model in ['large', 'lg']:
            self.nlp = spacy.load("en_core_web_lg")
            self.model_name = 'Large spaCy english model'
        elif spacy_model in ['medium', 'md']:
            self.nlp = spacy.load("en_core_web_md")
            self.model_name = 'Medium spaCy english model'
        elif spacy_model in ['small', 'sm']:
            self.nlp = spacy.load("en_core_web_sm")
            self.model_name = 'Small spaCy english model'
        else:
            raise InvalidSpacyModelException("Invalid spaCy model. Please provide a valid spacy model name: ['large','medium','small']")
          
        ## For python >= 3.10    
        # match spacy_model:
        #     case 'large' | 'lg':
        #         self.nlp = spacy.load("en_core_web_lg")
        #         self.model_name = 'Large spaCy english model'
        #     case 'medium' | 'md':
        #         self.nlp = spacy.load("en_core_web_md")
        #         self.model_name = 'Medium spaCy english model'
        #     case 'small' | 'sm':
        #         self.nlp = spacy.load("en_core_web_sm")
        #         self.model_name = 'Small spaCy english model'
        #     case _:
        #         raise InvalidSpacyModelException("Invalid spaCy model. Please provide a valid spacy model name: ['large','medium','small']")
            
    
    @convert_input_to_string
    def remove_html_tags(self, text):
        """Remove HTML tags from the text."""
        return re.sub(r'<[^>]+>', '', text)

    @convert_input_to_string
    def remove_urls(self, text):
        """Remove URLs from the text."""
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    @convert_input_to_string
    def remove_special_characters(self, text):
        """Remove special characters from the text."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    @convert_input_to_string
    def remove_stopwords(self, text):
        """Remove stopwords using spaCy NLP model."""
        doc = self.nlp(text)
        return ' '.join([token.text for token in doc if token.text.lower() not in STOP_WORDS])

    @convert_input_to_string
    def remove_punctuation(self, text):
        """Remove punctuation using spaCy NLP model."""
        doc = self.nlp(text)
        return ' '.join([token.text for token in doc if not token.is_punct])

    @convert_input_to_string
    def remove_numbers(self, text):
        """Remove numbers from the text."""
        return re.sub(r'\d+', '', text)

    @convert_input_to_string
    def normalize_text(self, text):
        """Normalize text by converting common variations to a standard form."""
        replacements = {
            r"won't": "will not",
            r"can\'t": "can not",
            r"n\'t": " not",
            r"\'re": " are",
            r"\'s": " is",
            r"\'d": " would",
            r"\'ll": " will",
            r"\'t": " not",
            r"\'ve": " have",
            r"\'m": " am"
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text

    @convert_input_to_string
    def remove_whitespace(self, text):
        """Remove extra whitespace from the text."""
        return re.sub(r'\s+', ' ', text).strip()

    @convert_input_to_string
    def remove_accented_chars(self, text):
        """Remove accented characters from the text."""
        return ''.join(c for c in text if ord(c) < 128)

    @convert_input_to_string
    def remove_successive_characters(self, text, max_repeats=2):
        """Remove excessive repeated characters, reducing to a specified maximum."""
        return re.sub(r'(.)\1{'+str(max_repeats)+',}', r'\1' * max_repeats, text)

    @convert_input_to_string
    def noise_remove(self, text, steps=None):
        """
        Apply multiple preprocessing steps to the input text.
        If steps is None, apply all steps in a default order.
        """
        default_steps = [
            self.remove_html_tags,
            self.remove_urls,
            self.remove_special_characters,
            self.remove_stopwords,
            self.remove_punctuation,
            self.remove_numbers,
            self.normalize_text,
            self.remove_whitespace,
            self.remove_accented_chars,
            self.remove_successive_characters
        ]
        
        steps = steps or default_steps
        
        for step in steps:
            text = step(text)
        
        return text
    
    # Utility Functions
    def spacy_model(self):
        return self.model_name
    

class Preprocessor:
    """Preprocess text with the spaCy NLP models."""
    
    def __init__(self, spacy_model='large'):
        if spacy_model in ['large', 'lg']:
            self.nlp = spacy.load("en_core_web_lg")
            self.model_name = 'Large spaCy english model'
        elif spacy_model in ['medium', 'md']:
            self.nlp = spacy.load("en_core_web_md")
            self.model_name = 'Medium spaCy english model'
        elif spacy_model in ['small', 'sm']:
            self.nlp = spacy.load("en_core_web_sm")
            self.model_name = 'Small spaCy english model'
        else:
            raise InvalidSpacyModelException("Invalid spaCy model. Please provide a valid spacy model name: ['large','medium','small']")
    
    def lemmatize(self, text):
        """Lemmatize the text."""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    
    def get_sentences(self, text):
        """Split the text into sentences."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
    
    def sent_tokenize(self, text):
        """Split the text into sentences (redundant with get_sentences)."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
    
    def preprocess(self, text, steps=None):
        """Apply multiple preprocessing steps to the input text."""
        default_steps = [
            self.lemmatize,
            # self.stem,  # Uncomment when `stem` is implemented
            self.get_sentences
        ]
        
        steps = steps or default_steps

        for step in steps:
            text = step(text)
        
        return text



class NamedEntityRecogniser:
    """Apply Named Entity Recognition with the spaCy NLP models."""
    
    def __init__(self, spacy_model='large'):
        if spacy_model in ['large', 'lg']:
            self.nlp = spacy.load("en_core_web_lg")
            self.model_name = 'Large spaCy english model'
        elif spacy_model in ['medium', 'md']:
            self.nlp = spacy.load("en_core_web_md")
            self.model_name = 'Medium spaCy english model'
        elif spacy_model in ['small', 'sm']:
            self.nlp = spacy.load("en_core_web_sm")
            self.model_name = 'Small spaCy english model'
        else:
            raise InvalidSpacyModelException("Invalid spaCy model. Please provide a valid spacy model name: ['large','medium','small']")
    
    def find_entities(self, text):
        """Perform Named Entity Recognition on the spaCy Doc."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    def count_entities(self, text):
        doc = self.nlp(text)
        entity_list = [(ent.text, ent.label_) for ent in doc.ents]
        return Counter(entity_list)



class POSTagger:
    def __init__(self):
        """Tag parts of speech with the spaCy NLP models."""
        self.nlp = spacy.load("en_core_web_lg")
    
    def count_pos(self, text):
        """Count Part-of-Speech tags in a spaCy Doc."""
        doc = self.nlp(text)
        pos_list = [token.pos_ for token in doc]
        return Counter(pos_list)
    
    def get_entities(self, text):
        """Perform Named Entity Recognition on the spaCy Doc."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def pos_tagging(self, text):
        """Perform Part-of-Speech tagging on the spaCy Doc."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

class SentimentAnalyser:
    
    def __init__(self):
        """Analyze Sentiment with the spaCy NLP model"""
        pass
  