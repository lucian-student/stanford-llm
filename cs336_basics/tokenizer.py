from typing import (
    Iterable,
    Iterator,
    List,
    BinaryIO,
    Optional,
    Dict,
    Tuple,
    DefaultDict,
    Set,
)
import os
from multiprocessing import Pool
import math
import regex as re
import collections


class Tokenizer:

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: Optional[Dict[int, bytes]] = None,
        merges: Optional[List[Tuple[bytes, bytes]]] = None,
        special_tokens: list[str] = None,
    ):
        """
        Asi důvod read_size 4096 -> je, že to je velikost stránky
        * mergujou se special tokeny, nebo vždy jsou samotný
        """
        self.special_tokens = special_tokens if special_tokens else []
        self.encoded_special_tokens: List[bytes] = [
            token.encode() for token in self.special_tokens
        ]
        self.encoded_special_tokens.sort(key=len,reverse=True)
        self.special_tokens.sort(key=len,reverse=True)
        if not vocab:
            self.vocab: Dict[int, bytes] = {}
            self.reverse_vocab: Dict[bytes, int] = {}
            for token in self.special_tokens:
                self.add_token(token.encode())
            for i in range(256):
                self.add_token(i.to_bytes(1))
        else:
            self.vocab = vocab
            self.reverse_vocab: Dict[bytes, int] = {}
            for id, token in self.vocab.items():
                self.reverse_vocab[token] = id
            for token in self.encoded_special_tokens:
                if token not in self.reverse_vocab:
                    self.add_token(token)
        self.merges: List[Tuple[bytes, bytes]] = merges if merges else []

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
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        pattern = b"|".join(escaped_special_tokens)
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
        read_size: Optional[int] = None,
    ):
        with open(file_name, "rb") as f:
            f.seek(0, os.SEEK_END)
            length = f.tell()
        if not read_size:
            offset = math.ceil(length / num_processes)
        else:
            offset = read_size
        read_chunks = math.ceil(length / offset)
        special_token_positions: List[Tuple[int, int]] = []
        with Pool(processes=num_processes) as p:
            results = [
                p.apply_async(
                    func=Tokenizer.find_special_tokens_open,
                    args=(file_name, i * offset, offset, self.encoded_special_tokens),
                )
                for i in range(read_chunks)
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

    def fit(
        self,
        file_name: str | os.PathLike,
        vocab_size: int,
        num_processes: int = 1,
        verbose: bool = False,
        read_size: Optional[int] = None,
    ):
        """
        Základní implementace nasplituju data podle prvního speciálního tokenu
        Raises:
            OSError, jelikož kontroluje délku souboru
        """
        # zíkskání délky souboru
        chunk_boundary = self.get_chunk_boundaries(
            file_name, num_processes, verbose, read_size
        )
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

        while len(self.vocab) < vocab_size:
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

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        pass

    def encode_chunk(self, text: str, pattern: re.Pattern[str]):
        for pretoken_match in re.finditer(pattern, text):
            pretoken = pretoken_match.group().encode()
            pretoken = tuple(pretoken[i : i + 1] for i in range(len(pretoken)))
            for merge in self.merges:
                new_encoded_token = []
                i = 1
                new_token = merge[0] + merge[1]
                while i < len(pretoken):
                    combined = pretoken[i - 1] + pretoken[i]
                    if combined == new_token:
                        new_encoded_token.append(combined)
                        i += 2
                    else:
                        new_encoded_token.append(pretoken[i - 1])
                        i += 1
                if (
                    i == len(pretoken)
                    or not new_encoded_token
                    or new_encoded_token[-1] != new_token
                ):
                    new_encoded_token.append(pretoken[-1])
                pretoken = tuple(new_encoded_token)
            for token in pretoken:
                yield self.reverse_vocab[token]

    def encode(self, text: str) -> list[int]:
        """ """
        encoded_text = text.encode()
        special_token_positions: List[Tuple[int, int, int]] = []
        if self.encoded_special_tokens:
            escaped_special_tokens = [
                re.escape(token) for token in self.encoded_special_tokens
            ]
            pattern = b"|".join(escaped_special_tokens)
            # start,end,id
            for m in re.finditer(pattern, encoded_text):
                special_token_positions.append(
                    (m.start(), m.end(), self.reverse_vocab[m.group()])
                )
        ids: List[int] = []
        pretoken_pattern = re.compile(Tokenizer.PAT)
        start = 0
        for special_info in special_token_positions:
            for id in self.encode_chunk(
                encoded_text[start : special_info[0]].decode(), pretoken_pattern
            ):
                ids.append(id)
            ids.append(special_info[2])
            start = special_info[1]
        for id in self.encode_chunk(encoded_text[start:].decode(), pretoken_pattern):
            ids.append(id)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            ids = self.encode(text)
            for id in ids:
                yield id

    def decode(self, ids: list[int]) -> str:
        decoded = bytearray()
        for id in ids:
            decoded.extend(self.vocab[id])
        return decoded.decode(errors="replace")
