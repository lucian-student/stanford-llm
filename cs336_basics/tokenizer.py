from typing import List, BinaryIO, Optional, Dict, Tuple
import os
from multiprocessing import Pool
import math
import regex as re


class Tokenizer:

    def __init__(
        self, vocab_size: int, special_tokens: list[str], read_size: int = 4096
    ):
        """
        Asi důvod read_size 4096 -> je, že to je velikost stránky
        * mergujou se special tokeny, nebo vždy jsou samotný
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        if not self.special_tokens:
            raise ValueError(
                "Special tokens should at least contain 1 token, which is assumed to be end of document!"
            )
        self.encoded_special_tokens: List[bytes] = [
            token.encode() for token in self.special_tokens
        ]
        self.vocab: Dict[int, bytes] = {}
        self.reverse_vocab: Dict[bytes, int] = {}
        self.merges: List[Tuple[bytes, bytes]] = []

    @staticmethod
    def find_special_tokens(
        stream: BinaryIO,
        offset: int,
        read_size: int,
        special_tokens: List[bytes],
    ) -> List[Tuple[int]]:
        """
        Vrátíme dvojice počáteční pozici tokenu
        """
        positions: List[int] = []
        stream.seek(offset)
        longest_token = max(special_tokens, key=len)
        data = stream.read(read_size + len(longest_token) - 1)
        pattern = b"|".join(special_tokens)
        for m in re.finditer(pattern, data):
            if m.start() < read_size:
                positions.append(m.start())
        return positions

    @staticmethod
    def find_special_tokens_open(
        file_name: str | os.PathLike,
        offset: int,
        read_size: int,
        special_tokens: List[bytes],
    ):
        """
        Raises:
            OSError, because iam calling open
        """
        positions: List[int] = []
        with open(file_name, "rb") as f:
            positions = Tokenizer.find_special_tokens(
                f, offset, read_size, special_tokens
            )
        return positions

    def fit(
        self,
        file_name: str | os.PathLike,
        num_processes: int = 1,
        verbose: bool = False,
    ):
        """
        Základní implementace nasplituju data podle prvního speciálního tokenu
        Raises:
            OSError, jelikož kontroluje délku souboru
        """
        # zíkskání délky souboru
        with open(file_name, "rb") as f:
            f.seek(0, os.SEEK_END)
            length = f.tell()

        offset = math.ceil(length / num_processes)
        special_token_positions: List[int] = []
        with Pool(processes=num_processes) as p:
            results = [
                p.apply_async(
                    func=Tokenizer.find_special_tokens_open,
                    args=(file_name, i * offset, offset, self.encoded_special_tokens),
                )
                for i in range(num_processes)
            ]
            for r in results:
                positions = r.get()
                special_token_positions.extend(positions)
        # teď by měla mít v sobě listina pozice všech delimeter tokenů
        if verbose:
            print(len(special_token_positions))
        """
        Teď bych měl spustit úlohy, které pretokenizují chunky.
        Taky je potřeba vhodně upravit pozice na chunk boundry, takže start(inclusive) a konec(exclusive)
        Dále se vytvoří histogramy chunků
        Histogramy se zmergujou na jeden histogram, asi sekvenčně.
        """
        chunk_boundary: List[Tuple[int, int]] = []
        """
        Potom v cyklu dokud se nenaplní slovník, nebo dojdou nová slova, tak hledám nejpopulárnější páry tokenů a merguju tokeny
        """

    def tranform(self):
        pass
