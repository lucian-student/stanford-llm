from dataclasses import dataclass
import argparse


@dataclass
class InferenceArguments:
    model_path: str
    prompt_path: str


def parse_arguments() -> InferenceArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-p", "--prompt", required=True)


def inference():
    pass
