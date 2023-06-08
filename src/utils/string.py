def has_digit(string: str) -> bool:
    return any(char.isdigit() for char in string)
