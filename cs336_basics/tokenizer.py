from typing import List, BinaryIO, Optional, Dict, Tuple, DefaultDict, Set
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
        for token in self.special_tokens:
            self.add_token(token.encode())
        for i in range(256):
            self.add_token(i.to_bytes(1))
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
        stream: BinaryIO,
        chunks: List[Tuple[int, int]],
    ) -> DefaultDict[Tuple[bytes, ...], int]:
        preped_histogram: DefaultDict[Tuple[bytes, ...], int] = collections.defaultdict(
            int
        )
        pattern = re.compile(Tokenizer.PAT)
        for start, end in chunks:
            stream.seek(start)
            length = end - start
            data = stream.read(length).decode()
            for token in re.finditer(pattern, data):
                token = token.group().encode()
                preped_histogram[
                    tuple(token[i : i + 1] for i in range(len(token)))
                ] += 1
        return preped_histogram

    @staticmethod
    def pretokenize_open(
        file_name: str | os.PathLike, chunks: List[Tuple[int, int]]
    ) -> DefaultDict[Tuple[bytes, ...], int]:
        with open(file_name, "rb") as f:
            histogram = Tokenizer.pretokenize(f, chunks)
        return histogram

    def add_token(self, token: bytes):
        id = len(self.vocab)
        self.vocab[id] = token
        self.reverse_vocab[token] = id

    def get_chunk_boundaries(
        self,
        file_name: str | os.PathLike,
        num_processes: int = 1,
        verbose: bool = False,
    ):
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

        chunk_boundary: List[Tuple[int, int]] = []
        previous_postion: int = 0
        for start, end in special_token_positions:
            chunk_boundary.append((previous_postion, start))
            previous_postion = end
        if special_token_positions:
            chunk_boundary.append((special_token_positions[-1][1], length))
        if not chunk_boundary:
            return [(0, length)]
        return chunk_boundary

    def get_pretoken_hisogram(
        self,
        chunk_boundary: List[Tuple[int, int]],
        file_name: str | os.PathLike,
        num_processes: int = 1,
        verbose: bool = False,
    ):
        pretoken_histogram: DefaultDict[Tuple[bytes, ...], int] = (
            collections.defaultdict(int)
        )
        chunks_per_process = math.ceil(len(chunk_boundary) / num_processes)
        with Pool(processes=num_processes) as p:
            results = [
                p.apply_async(
                    func=Tokenizer.pretokenize_open,
                    args=(
                        file_name,
                        chunk_boundary[
                            i * chunks_per_process : (i + 1) * chunks_per_process
                        ],
                    ),
                )
                for i in range(num_processes)
            ]
            for r in results:
                hist = r.get()
                for token, count in hist.items():
                    pretoken_histogram[token] += count

        if verbose:
            print(len(pretoken_histogram))
        return pretoken_histogram

    def build_vocabulary(self):
        pass

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
        chunk_boundary = self.get_chunk_boundaries(file_name, num_processes, verbose)
        """
        Teď bych měl spustit úlohy, které pretokenizují chunky.
        Taky je potřeba vhodně upravit pozice na chunk boundry, takže start(inclusive) a konec(exclusive)
        Dále se vytvoří histogramy chunků
        Histogramy se zmergujou na jeden histogram, asi sekvenčně.
        """
        pretoken_histogram = self.get_pretoken_hisogram(
            chunk_boundary, file_name, num_processes, verbose
        )
        """
        Potom v cyklu dokud se nenaplní slovník, nebo dojdou nová slova, tak hledám nejpopulárnější páry tokenů a merguju tokeny.
        * Mergování, prý nejde paralelizovat
        * důležitý je cachovat, počty párů

        1. zkombinovat, dvojice otkenů na 1 -> v pretoken_histogramu
        2. při tom je vhodné updatovat histogram párů (newtoken) = (první,druhý) -> (nultý,první)--,(první_token , durhý_token)--, (druhý_token,třetí_token)--
        2. po tom je vhodné inkrementovat nové páry -> (nultý,newtoken)++,(newtoken,třetí)
        """
        pair_histogram: DefaultDict[Tuple[bytes, bytes], int] = collections.defaultdict(
            int
        )
        pair_pretoken: DefaultDict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]] = (
            collections.defaultdict(set)
        )
        for token, count in pretoken_histogram.items():
            for i in range(1, len(token)):
                pair_histogram[(token[i - 1], token[i])] += count
                pair_pretoken[(token[i - 1], token[i])].add(token)
        if verbose:
            print("pair histogram length: ", len(pair_histogram))

        while len(self.vocab) < self.vocab_size:
            try:
                max_token = max(
                    pair_histogram, key=lambda x: (pair_histogram.get(x), x)
                )
                # print(pair_histogram.get(max_token))
            except ValueError:
                return
            new_token = max_token[0] + max_token[1]
            self.add_token(new_token)
            self.merges.append(max_token)
            encoded_tokens = set(pair_pretoken[max_token])
            for encoded_token in encoded_tokens:
                new_encoded_token = []
                i = 1
                while i < len(encoded_token):
                    combined = encoded_token[i - 1] + encoded_token[i]
                    if combined == new_token:
                        new_encoded_token.append(combined)
                        i += 2
                        # new token found need to update pair_histogram
                        # jednoduché řešení, je odstranit počty pro daný pretoken, potom znovu inkrementovat páry pretokenu -> pokud zjistime, že pretoken obsahuje nový token
                        # includes_newtoken = True
                    else:
                        new_encoded_token.append(encoded_token[i - 1])
                        i += 1
                # print(i,len(encoded_token),encoded_token,new_encoded_token)
                if (
                    i == len(encoded_token)
                    or not new_encoded_token
                    or new_encoded_token[-1] != new_token
                ):
                    new_encoded_token.append(encoded_token[-1])
                # updates pairs if new_token is included, this isnt optimal, but it was easy to code -> maybe i will improve it someday
                # if includes_newtoken:
                new_encoded_token = tuple(new_encoded_token)
                for i in range(1, len(encoded_token)):
                    pair_histogram[
                        (encoded_token[i - 1], encoded_token[i])
                    ] -= pretoken_histogram[encoded_token]
                    pair_pretoken[(encoded_token[i - 1], encoded_token[i])].discard(
                        encoded_token
                    )
                    if pair_histogram[(encoded_token[i - 1], encoded_token[i])] <= 0:
                        del pair_histogram[(encoded_token[i - 1], encoded_token[i])]
                    if not pair_pretoken[(encoded_token[i - 1], encoded_token[i])]:
                        del pair_pretoken[(encoded_token[i - 1], encoded_token[i])]

                for i in range(1, len(new_encoded_token)):
                    pair_histogram[
                        (new_encoded_token[i - 1], new_encoded_token[i])
                    ] += pretoken_histogram[encoded_token]
                    pair_pretoken[(new_encoded_token[i - 1], new_encoded_token[i])].add(
                        new_encoded_token
                    )
                pretoken_histogram[new_encoded_token] = pretoken_histogram[
                    encoded_token
                ]
                del pretoken_histogram[encoded_token]

    def tranform(self):
        pass
