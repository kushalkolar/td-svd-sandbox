import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd
import tifffile


class Config:
    PARENT_PATH: Path = None

    def set_parent_path(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise NotADirectoryError(f"path does not exist: {path}")

        self.PARENT_PATH = path


config = Config()


def check_path_exist(path: Path):
    if not Path(path).exists():
        raise FileNotFoundError(str(path))


@pd.api.extensions.register_series_accessor("lib")
class SeriesAccessor:
    def __init__(self, series: pd.Series):
        self._series = series

    @property
    def uuid(self) -> str:
        return self._series["uuid"]

    def load_movie(self):
        if config.PARENT_PATH is None:
            raise ValueError("parent path not set")

        path = config.PARENT_PATH.joinpath(self.uuid, "movie.tiff")
        check_path_exist(path)

        return tifffile.memmap(path)

    def load_spatial(self, name: str):
        if config.PARENT_PATH is None:
            raise ValueError("parent path not set")

        path = config.PARENT_PATH.joinpath(self.uuid, f"{name}.npy")
        return np.load(path)


@pd.api.extensions.register_dataframe_accessor("lib")
class DataFrameAccessor:
    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe

    def add_item(
            self,
            name: str,
            movie_path: str | Path,
            rect: int,
            type: str,
            source: str,
            orig: str,
    ):
        if config.PARENT_PATH is None:
            raise ValueError("parent path not set")

        check_path_exist(movie_path)

        u = str(uuid4())

        os.makedirs(config.PARENT_PATH.joinpath(u))

        shutil.copy(movie_path, config.PARENT_PATH.joinpath(u, "movie.tiff"))

        d = {
            "name": name,
            "rect": rect,
            "type": type,
            "source": source,
            "orig": orig,
            "uuid": u,
        }

        s = pd.Series(d)

        self._df.loc[self._df.index.size] = s

        self._df.to_csv(config.PARENT_PATH.joinpath("dataframe.csv"))
