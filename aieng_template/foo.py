"""foo module."""


def foo(bar: str) -> str:
    """Return input concatenated with 'foo'.

    Parameters
    ----------
    bar : str
        Input string to be concatenated with 'foo'.

    Returns
    -------
    str
        Concatenated string.

    Examples
    --------
    >>> foo("bar")
    'foobar'
    >>> foo("baz")
    'foobaz'

    """
    return f"foo{bar}"
