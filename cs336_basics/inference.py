from dataclasses import dataclass
import argparse
from cs336_basics.train import load_checkpoint
from cs336_basics.configuration import TrainingSchema
import yaml
from pydantic import ValidationError
from cs336_basics.layers import TransformerLM, softmax_with_temperature, top_p_sampling
from cs336_basics.optim import AdamW
from cs336_basics.tokenizer import Tokenizer
from typing import Literal, List
import torch
from jaxtyping import Float, Int


@dataclass
class InferenceArguments:
    model_path: str
    config_path: str
    prompt_path: str
    vocab_path: str
    merges_path: str
    device: Literal["cuda", "cpu"]
    special_token: str
    temperature: float
    probability: float


def parse_arguments() -> InferenceArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-p", "--prompt_path", required=True)
    parser.add_argument("-c", "--config_path", required=True)
    parser.add_argument("-M", "--merges_path", required=True)
    parser.add_argument("-V", "--vocab_path", required=True)
    parser.add_argument("-d", "--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("-s", "--special_token", default="<|endoftext|>")
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("-P", "--probability", type=float, default=1.0)
    args = parser.parse_args()
    return InferenceArguments(**vars(args))


def decodeText(
    device_str: str,
    special_token: str,
    temperature: float,
    probabiliy: float,
    tokenizer: Tokenizer,
    model: TransformerLM,
    prompt: str,
) -> str:
    device = torch.device(device_str)
    model = model.to(device=device)
    context_length = model.context_length
    current_tokens: Int[torch.Tensor, "seq"] = torch.tensor(
        tokenizer.encode(prompt), dtype=torch.long, device=device
    )
    output_tokens: List[int] = []
    while current_tokens.shape[0] < context_length:
        with torch.no_grad():
            logits: Float[torch.Tensor, "seq vocab_size"] = model(current_tokens)
            last_logit = logits[-1]
            proba: Float[torch.Tensor, "vocab_size"] = softmax_with_temperature(
                last_logit, temperature, dim=-1
            )
            top_p = top_p_sampling(
                proba,p=0.1,dim=0
            )
            normalized_top_p = top_p/torch.sum(top_p)
            select = torch.rand(1, device=device, dtype=torch.float32).item()
            cdf = normalized_top_p.cumsum(dim=0)
            mask = cdf > select
            selected_index = torch.nonzero(mask)[0].item()
            output_tokens.append(selected_index)
            current_tokens = torch.cat(
                [
                    current_tokens,
                    torch.tensor([selected_index], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            if tokenizer.vocab[selected_index] == special_token.encode():
                break
    return tokenizer.decode(output_tokens[:-1])


def inference():
    args = parse_arguments()
    try:
        with open(args.config_path) as config_file:
            config_dict = yaml.safe_load(config_file)
        schema = TrainingSchema(**config_dict)
        config = schema.build()
        params = config.first_model_params()
        model = TransformerLM(**params)
        optim = AdamW(model.parameters())
        load_checkpoint(args.model_path, model, optim)
        tokenizer = Tokenizer.from_files(
            args.vocab_path, args.merges_path, [args.special_token]
        )
        with open(args.prompt_path) as prompt_file:
            prompt = prompt_file.read() + args.special_token
        answer = decodeText(
            args.device,
            args.special_token,
            args.temperature,
            args.probability,
            tokenizer,
            model,
            prompt,
        )
        print(answer)
    except OSError as e:
        print(e)
    except ValidationError as e:
        print(e)
