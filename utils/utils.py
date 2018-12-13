from sacred.utils import SacredInterrupt


class CustomInterrupt(SacredInterrupt):
    def __init__(self, STATUS):
        print(STATUS)
        self.STATUS = STATUS
