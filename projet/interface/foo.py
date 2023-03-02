def foo(foo: int, bar: float) -> float:
    """
    Function to test the documentation generation

    Args:
        foo (int): an int
        bar (float): a float

    Returns:
        float: foo + bar

    >>> foo(4, 7.3)
    11.3

    >>> foo(8, 2.1)
    10.1

    """

    return foo + bar


if __name__ == "__main__":
    print("Hello World !")
