from torchvision.transforms import Compose as _Compose


class StarCompose(_Compose):
    def __call__(self, data):
        for t in self.transforms:
            data = t(*data)
        return data
