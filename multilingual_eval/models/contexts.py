class DumbContext:
    """
    Context that does nothing
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
