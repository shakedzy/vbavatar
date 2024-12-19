import os
import shutil
from datetime import datetime
from .singleton_metaclass import SingletonMeta


class Cache(metaclass=SingletonMeta):
    _ROOT_DIR = 'cache'

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        else:
            super().__init__()
            self._initialized = True
            self._init_time = datetime.now()
            for dr in [self.root, self.directory]:
                if not os.path.exists(dr):
                    os.makedirs(dr)

    @property
    def root(self) -> str:
        return os.path.join(os.getcwd(), self._ROOT_DIR)
    
    @property
    def directory(self) -> str:
        return os.path.join(self.root, self._init_time.strftime('%Y%m%d%H%M%S'))
    
    def clear(self, *, all: bool = False) -> None:
        dr = self.root if all else self.directory
        shutil.rmtree(dr)
        os.mkdir(dr)
    