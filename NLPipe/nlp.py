import re
import spacy
from collections import Counter
from NLPipe.utils.decorators import convert_input_to_string
from NLPipe.utils.exceptions import InvalidSpacyModelException, InvalidArgumentValue
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# NOTE Add: 
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
    def remove_mentions(self, text):
        """Remove mentions (e.g., @user) from the input text."""
        return re.sub(r'@\w+', '', text).strip()
    
    @convert_input_to_string
    def remove_hashtags(self, text):
        """Remove hashtags (e.g., #hashtag) from the input text."""
        return re.sub(r'#\w+', '', text).strip()

    @convert_input_to_string
    def remove_urls(self, text):
        """Remove URLs from the text using spaCy."""
        doc = self.nlp(text)
        return " ".join(token.text for token in doc if not token.like_url) 
    
    @convert_input_to_string
    def remove_emails(self, text):
        """Remove Emails from the text."""
        regex_sub = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text, flags=re.MULTILINE)
        doc = self.nlp(regex_sub)
        return " ".join(token.text for token in doc if not token.like_email)
    
    @convert_input_to_string
    def remove_non_ascii_characters(self, text):
        """Remove non-ascii characters fromt text"""
        doc = self.nlp(text)
        return " ".join(token.text for token in doc if token.is_ascii)

    @convert_input_to_string
    def remove_special_characters(self, text):
        """Remove special characters from the text."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    @convert_input_to_string
    def remove_stopwords(self, text):
        """Remove stopwords using spaCy NLP model."""
        doc = self.nlp(text)
        return " ".join([token.text for token in doc if not token.is_stop])

    @convert_input_to_string
    def remove_punctuation(self, text):
        """Remove punctuation using spaCy NLP model."""
        doc = self.nlp(text)
        return " ".join([token.text for token in doc if not token.is_punct])

    @convert_input_to_string
    def remove_numbers(self, text):
        """Remove numbers from the text."""
        return re.sub(r'\d+', '', text)

    @convert_input_to_string
    def remove_abbreviations(self, text):
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
        return "".join([c for c in text if ord(c) < 128])

    @convert_input_to_string
    def remove_successive_characters(self, text, max_repeats=2):
        """Remove excessive repeated characters, reducing to a specified maximum."""
        return re.sub(r'(.)\1{'+str(max_repeats)+',}', r'\1' * max_repeats, text)

    # NOTE Add a good way for the user to add steps
    @convert_input_to_string
    def noise_remove(self, text, steps=None):
        """
        Apply multiple preprocessing steps to the input text.
        If steps is None, apply all steps in a default order.
        """
        default_steps = [
            self.remove_abbreviations,
            self.remove_html_tags,
            self.remove_emails,
            self.remove_mentions,
            self.remove_hashtags,
            self.remove_urls,
            self.remove_special_characters,
            self.remove_stopwords,
            self.remove_punctuation,
            self.remove_numbers,
            self.remove_whitespace,
            self.remove_accented_chars,
            self.remove_successive_characters
        ]
        
        steps = steps or default_steps
        
        for step in steps:
            text = step(text)
            # print(f"Text: {text}")
            # print("Text: ", text)
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
    
    @convert_input_to_string
    def lemmatize(self, text):
        """Lemmatize the text."""
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])
    
    @convert_input_to_string
    def get_sentences(self, text):
        """Split the text into sentences."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
    
    @convert_input_to_string
    def sent_tokenize(self, text):
        """Split the text into sentences (redundant with get_sentences)."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
    
    @convert_input_to_string
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
    
    # Utility Functions
    def spacy_model(self):
        return self.model_name


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
    
    @convert_input_to_string
    def find_entities(self, text, result_type='dict'):
        """Perform Named Entity Recognition on the spaCy Doc."""
        doc = self.nlp(text)
        if result_type == 'dict':
            return [{ent.text: ent.label_} for ent in doc.ents]
        elif result_type == 'list':
            return [(ent.text, ent.label_) for ent in doc.ents]
        else:
            raise InvalidArgumentValue("Invalid argument. Please provide a valid result type: ['dict','list']")
        
    @convert_input_to_string
    def count_entities(self, text):
        doc = self.nlp(text)
        entity_list = [ent.label_ for ent in doc.ents]
        return dict(Counter(entity_list))
    
    def find_and_count_entities(self, text):
        doc = self.nlp(text)
        ent_list = [ent.label_ for ent in doc.ents]
        counter = dict(Counter(ent_list))
        return [(ent.text, ent.label_, counter[ent.label_]) for ent in doc.ents]
            
    
    
class POSTagger:
    def __init__(self):
        """Tag parts of speech with the spaCy NLP models."""
        self.nlp = spacy.load("en_core_web_lg")
    
    @convert_input_to_string
    def count_pos(self, text):
        """Count Part-of-Speech tags in a spaCy Doc."""
        doc = self.nlp(text)
        pos_list = [token.pos_ for token in doc]
        return dict(Counter(pos_list))

    @convert_input_to_string
    def pos_tags(self, text):
        """Perform Part-of-Speech tagging on the spaCy Doc."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    @convert_input_to_string
    def pos_tag_and_count(self, text):
        doc = self.nlp(text)
        pos_list = [token.pos_ for token in doc]
        counter = dict(Counter(pos_list))
        return [(token.text, token.pos_, counter[token.pos_]) for token in doc]

class SentimentAnalyser:
    
    def __init__(self, model="vader"):
        """Analyze Sentiment with the spaCy NLP model"""
        if model == "vader":
            self.model = sia
            
            
    def get_sentiment(self, text):
        return self.model.polarity_scores(text)['compound']
    
    def get_sentiment_scores(self, text):
        scores = self.model.polarity_scores(text)
        return scores