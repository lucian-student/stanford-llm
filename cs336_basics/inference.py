from dataclasses import dataclass
import argparse

@dataclass
class InferenceArguments:
    model_path: str
    prompt_path: str

def parse_arguments()->InferenceArguments:
    parser = argparse.ArgumentParser()
    

def inference():
    pass
