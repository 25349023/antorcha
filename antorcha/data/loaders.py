class PreprocessedDataLoader:
    def __init__(self, loader, func):
        self.loader = loader
        self.func = func

    def __iter__(self):
        return (self.func(*batch) for batch in self.loader)

    def __len__(self):
        return len(self.loader)
