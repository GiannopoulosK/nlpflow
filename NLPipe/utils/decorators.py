from functools import wraps

def convert_input_to_string(func):
    """Decorator to ensure the input is a string."""
    @wraps(func)
    def wrapper(self, text, *args, **kwargs):
        if text is None:
            raise TypeError("Input cannot be None")
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as e:
                raise TypeError(f"Input cannot be converted to string: {e}")
        return func(self, text, *args, **kwargs)
    return wrapper