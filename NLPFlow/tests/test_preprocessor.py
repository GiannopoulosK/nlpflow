import pytest
from nlpflow.preprocessing import Preprocessor
from nlpflow.utils.exceptions import InvalidModelException

def test_spacy_model_picker_initialization_large():
    processor = Preprocessor(model='large')
    assert processor.model_name == 'Large spaCy English model'

def test_spacy_model_picker_initialization_medium():
    processor = Preprocessor(model='medium')
    assert processor.model_name == 'Medium spaCy English model'

def test_spacy_model_picker_initialization_small():
    processor = Preprocessor(model='small')
    assert processor.model_name == 'Small spaCy English model'

def test_spacy_model_picker_invalid_model():
    with pytest.raises(InvalidModelException):
        Preprocessor(model='invalid_model')

# Test Text Preprocessing Methods
def test_remove_contractions():
    processor = Preprocessor(model='large')
    text = "I can't believe it's not butter."
    expected = "I cannot believe it is not butter."
    assert processor.remove_contractions(text) == expected

def test_remove_html_tags():
    processor = Preprocessor(model='large')
    text = "<p>This is a <b>bold</b> statement.</p>"
    expected = "This is a bold statement."
    assert processor.remove_html_tags(text) == expected

def test_remove_mentions():
    processor = Preprocessor(model='large')
    text = "Hello @user, how are you?"
    expected = "Hello , how are you?"
    assert processor.remove_mentions(text) == expected

def test_remove_urls():
    processor = Preprocessor(model='large')
    text = "Visit us at https://example.com"
    expected = "Visit us at"
    assert processor.remove_urls(text) == expected

def test_remove_emails():
    processor = Preprocessor(model='large')
    text = "Contact us at info@example.com"
    expected = "Contact us at"
    assert processor.remove_emails(text) == expected

def test_remove_special_characters():
    processor = Preprocessor(model='large')
    text = "Hello, World! Welcome to @Python3."
    expected = "Hello World Welcome to Python3"
    assert processor.remove_special_characters(text) == expected