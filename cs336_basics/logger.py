
import csv
import io
from typing import Dict, Set, Protocol, Iterable, Any
import os
from abc import ABC,abstractmethod

def error_message(module: str, action: str, problem: str):
    return f"{module} :: {action} - {problem}"

class Logger(ABC):

    @abstractmethod
    def stream(self) -> io.TextIOBase:
        pass

    @abstractmethod
    def log(self,data:Dict):
        pass

    def close(self):
        self.stream().close()

class CSVWriter(Protocol):
    def writerow(self, row: Iterable[Any]) -> Any: ...
    def writerows(self, rows: Iterable[Iterable[Any]]) -> None: ...


class CSVLogger(Logger):

    def __init__(self):
        self.headers: Set[str] = set()

    def log(self, data: Dict):
        """
        throws ValueError
        """
        if self.first_row():
            self.headers = set(data.keys())
            self.writer().writerow(data.keys())
            self.set_first_row(False)

        keys = set(data.keys())
        if keys != self.headers:
            raise ValueError(
                error_message(
                    "CS336_basics",
                    "csv log",
                    f"Headers don't match {self.headers} != {keys}  !",
                )
            )
        self.writer().writerow([data[key] for key in self.headers])
        self.stream().flush()

    @abstractmethod
    def writer(self) -> CSVWriter:
        pass

    @abstractmethod
    def first_row(self) -> bool:
        pass

    @abstractmethod
    def set_first_row(self, val: bool):
        pass


class CSVFileLogger(CSVLogger):

    def __init__(self, path: str):
        super().__init__()
        """
        Raises:
            OSError
        """
        self._first_row = True
        if os.path.exists(path):
            self._first_row = False
        self._stream = open(path, "a", newline="")
        self._writer = csv.writer(self._stream)

    def stream(self):
        return self._stream

    def first_row(self):
        return self._first_row

    def set_first_row(self, val):
        self._first_row = val

    def writer(self) -> CSVWriter:
        return self._writer


class CSVStringLogger(CSVLogger):

    def __init__(self):
        super().__init__()
        self._stream = io.StringIO()
        self._first_row = True
        self._writer = csv.writer(self._stream)

    def stream(self):
        return self._stream

    def first_row(self):
        return self._first_row

    def set_first_row(self, val):
        self._first_row = val

    def writer(self) -> CSVWriter:
        return self._writer


