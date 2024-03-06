"""bar module."""


def bar(foo: str) -> str:
    """Return input concatenated with 'bar'.

    Parameters
    ----------
    foo : str
        Input string to be concatenated with 'bar'.

    Returns
    -------
    str
        Concatenated string.

    Examples
    --------
    >>> bar("foo")
    'barfoo'
    >>> bar("baz")
    'barbaz'

    """
    return f"bar{foo}"
