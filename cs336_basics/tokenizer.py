from typing import List, BinaryIO, Optional, Dict, Tuple, DefaultDict
import os
from multiprocessing import Pool
import math
import regex as re
import collections


class Tokenizer:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

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
    ) -> List[Tuple[int, int]]:
        """
        Vrátíme dvojice počáteční pozici tokenu
        """
        positions: List[Tuple[int, int]] = []
        stream.seek(offset)
        longest_token = max(special_tokens, key=len)
        data = stream.read(read_size + len(longest_token) - 1)
        pattern = b"|".join(special_tokens)
        for m in re.finditer(pattern, data):
            if m.start() < read_size:
                positions.append((m.start() + offset, m.end() + offset))
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
        with open(file_name, "rb") as f:
            positions = Tokenizer.find_special_tokens(
                f, offset, read_size, special_tokens
            )
        return positions

    @staticmethod
    def pretokenize(
        stream: BinaryIO, start: int, end: int
    ) -> DefaultDict[Tuple[bytes, ...], int]:
        stream.seek(start)
        length = end - start
        data = stream.read(length).decode()
        histogram: DefaultDict[str, int] = collections.defaultdict(int)
        for token in re.finditer(Tokenizer.PAT, data):
            histogram[token.group()] += 1
        preped_histogram: DefaultDict[Tuple[bytes, ...], int] = (
            collections.defaultdict(int)
        )
        for str_token, count in histogram.items():
            token = str_token.encode()
            preped_histogram[tuple(token[i : i + 1] for i in range(len(token)))] = count
        return preped_histogram

    @staticmethod
    def pretokenize_open(
        file_name: str | os.PathLike, start: int, end: int
    ) -> DefaultDict[Tuple[bytes, ...], int]:
        with open(file_name, "rb") as f:
            histogram = Tokenizer.pretokenize(f, start, end)
        return histogram

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
        special_token_positions: List[Tuple[int, int]] = []
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
            print(special_token_positions[:10])
            print(len(special_token_positions))
        """
        Teď bych měl spustit úlohy, které pretokenizují chunky.
        Taky je potřeba vhodně upravit pozice na chunk boundry, takže start(inclusive) a konec(exclusive)
        Dále se vytvoří histogramy chunků
        Histogramy se zmergujou na jeden histogram, asi sekvenčně.
        """
        chunk_boundary: List[Tuple[int, int]] = []
        previous_postion: int = 0
        for start, end in special_token_positions:
            chunk_boundary.append((previous_postion, start))
            previous_postion = end
        if special_token_positions:
            chunk_boundary.append((special_token_positions[-1][1], length))

        pretoken_histogram: DefaultDict[Tuple[bytes, ...], int] = (
            collections.defaultdict(int)
        )
        with Pool(processes=num_processes) as p:
            results = [
                p.apply_async(
                    func=Tokenizer.pretokenize_open, args=(file_name, start, end)
                )
                for (start, end) in chunk_boundary
            ]
            for r in results:
                hist = r.get()
                for token, count in hist.items():
                    pretoken_histogram[token] += count

        if verbose:
            print(len(pretoken_histogram))
        """
        Potom v cyklu dokud se nenaplní slovník, nebo dojdou nová slova, tak hledám nejpopulárnější páry tokenů a merguju tokeny
        """

    def tranform(self):
        pass
