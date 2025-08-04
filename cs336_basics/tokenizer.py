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
from dataclasses import dataclass
import argparse
from functools import lru_cache


@dataclass
class TokenizerArguments:
    # cesta k trenovacím datům, pokud není none, tak se natrénuje tokenizer
    special_tokens: List[str]
    vocab_size: Optional[int]
    train_path: Optional[str] = None
    # vocab_path
    input_vocab_path: Optional[str] = None
    input_merge_path: Optional[str] = None
    output_vocab_path: Optional[str] = None
    output_merge_path: Optional[str] = None
    # data, která se transformují
    decode: bool = False
    data_path: Optional[str] = None
    data_ouptut_path: Optional[str] = None


def parse_tokenizer_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--vocab_size", type=int)
    parser.add_argument("-t", "--train_path")
    parser.add_argument("-v", "--input_vocab_path")
    parser.add_argument("-V", "--output_vocab_path")
    parser.add_argument("-m", "--input_merge_path")
    parser.add_argument("-M", "--output_merge_path")
    parser.add_argument("-d", "--data_path")
    parser.add_argument("-o", "--data_ouptut_path")
    parser.add_argument("-s", "--special_tokens", nargs="*")
    parser.add_argument("-y", "--decode", action="store_true")
    args = parser.parse_args()
    arguments = TokenizerArguments(**vars(args))
    if arguments.train_path:
        if (arguments.input_vocab_path and not arguments.input_merge_path) or (
            not arguments.input_vocab_path and arguments.input_merge_path
        ):
            raise ValueError("Need input vocab and merges!")
        if (arguments.output_vocab_path and not arguments.output_merge_path) or (
            not arguments.output_vocab_path and arguments.output_merge_path
        ):
            raise ValueError("Need output vocab and merges!")
    return arguments


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
        self.encoded_special_tokens.sort(key=len, reverse=True)
        self.special_tokens.sort(key=len, reverse=True)
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
        if not self.special_tokens:
            return [(0, length)], []

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
            return [(0, length)], special_token_positions
        return chunk_boundary, special_token_positions

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
        chunk_boundary, _ = self.get_chunk_boundaries(
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

    def to_file(self, vocab_filepath: str, merges_filepath: str):
        """
        Uloží vocab a merges_filepath
        Formát mergů (length,token,length,token)*
        Formát vocabulary (length,token,id)*
        Raises
            OSError
        """
        print("vocab: ", len(self.vocab))
        with open(vocab_filepath, "wb") as vocab_file:
            for id, token in self.vocab.items():
                word = bytearray()
                word.extend(len(token).to_bytes(2, "little"))
                word.extend(token)
                word.extend(id.to_bytes(2, "little"))
                vocab_file.write(word)

        with open(merges_filepath, "wb") as merges_file:
            for a, b in self.merges:
                pair = bytearray()
                pair.extend(len(a).to_bytes(2, "little"))
                pair.extend(a)
                pair.extend(len(b).to_bytes(2, "little"))
                pair.extend(b)
                merges_file.write(pair)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Načte tokenizer z souboru
        """
        vocab: Dict[int, bytes] = {}
        with open(vocab_filepath, "rb") as vocab_file:
            while True:
                token_length_bytes = vocab_file.read(2)
                if not token_length_bytes:
                    break
                token_length = int.from_bytes(token_length_bytes, "little")
                token = vocab_file.read(token_length)
                if not token:
                    break
                id_bytes = vocab_file.read(2)
                if not id_bytes:
                    break
                id = int.from_bytes(id_bytes, "little")
                vocab[id] = token

        merges: List[Tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as merges_file:
            while True:
                a_length_bytes = merges_file.read(2)
                if not a_length_bytes:
                    break
                a_length = int.from_bytes(a_length_bytes, "little")
                a = merges_file.read(a_length)
                if not a:
                    break
                b_length_bytes = merges_file.read(2)
                if not b_length_bytes:
                    break
                b_length = int.from_bytes(b_length_bytes, "little")
                b = merges_file.read(b_length)
                if not b:
                    break
                merges.append((a, b))
        return Tokenizer(vocab, merges, special_tokens)

    def encode_to_stream(
        self,
        input_path: str,
        output_path: str,
        read_size: int = None,
        num_processes: int = 1,
    ):
        """
        Dostane input a output stream a tam uloží zakódovaná data
        Raises:
            OSError
        """
        print("vocab_size: ", len(self.vocab))
        chunk_boundaries, special_token_positions = self.get_chunk_boundaries(
            input_path, read_size=read_size, num_processes=num_processes
        )
        pretoken_pattern = re.compile(Tokenizer.PAT)
        with open(input_path, "rb") as input, open(output_path, "wb") as output:
            for i in range(len(chunk_boundaries)):
                start_chunk, end_chunk = chunk_boundaries[i]
                chunk_length = end_chunk - start_chunk
                input.seek(start_chunk)
                chunk = input.read(chunk_length).decode()
                for id in self.encode_chunk(chunk, pretoken_pattern):
                    output.write(id.to_bytes(2, "little"))
                if i < len(special_token_positions):
                    start_token, end_token = special_token_positions[i]
                    token_length = end_token - start_token
                    token = input.read(token_length)
                    output.write(self.reverse_vocab[token].to_bytes(2, "little"))

    def decode_to_stream(
        self,
        input_path: str,
        output_path: str,
        read_size: int = None,
    ):
        if read_size % 2 == 1:
            raise ValueError("Read size should be multiple of 2!")

        with (
            open(input_path, "rb") as input,
            open(output_path, "w", encoding="utf-8") as ouput,
        ):
            while True:
                chunk = input.read(read_size)
                if not chunk:
                    break
                chunk_bytes = bytearray()
                for i in range(0, len(chunk), 2):
                    id = int.from_bytes(chunk[i : i + 2], "little")
                    chunk_bytes.extend(self.vocab[id])
                ouput.write(chunk_bytes.decode(errors="replace"))

    @lru_cache(maxsize=100000)
    def pretoken_to_tokens(self, pretoken: bytes):
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
        return pretoken

    def encode_chunk(self, text: str, pattern: re.Pattern[str]):
        for pretoken_match in re.finditer(pattern, text):
            pretoken = pretoken_match.group().encode()
            pretoken = self.pretoken_to_tokens(pretoken)
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
