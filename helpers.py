import json
import numpy as np
import os
from typing import Any


def store_metadata(
    graph: str,
    prompt: str,
    output: str,
    overall_goal: str,
):
    metadata = {"prompt": prompt, "output": output, "overall_goal": overall_goal}
    os.makedirs(graph, exist_ok=True)
    if not os.path.exists(f"{graph}/metadata.json"):
        with open(f"{graph}/metadata.json", "w") as f:
            json.dump(metadata, f)


def store_feature(
    graph: str,
    node_id: str,
    index: str,
    examples_quantiles: dict[str, dict[str, Any]],
):
    quantile_data = dict()
    for quantile in examples_quantiles:
        quantile_name = quantile["quantile_name"]
        examples = quantile["examples"]
        full_examples = []
        for example in examples:
            text = ""
            tokens = example["tokens"]
            tokens_acts_list = example["tokens_acts_list"]
            threshold = float(np.percentile(list(set(tokens_acts_list)), 50))
            substrings = []
            for idx, token in enumerate(tokens):
                start_space = (
                    True
                    if token.startswith(" ")
                    else False
                    if token.endswith(" ")
                    else None
                )
                effective_token = (
                    token
                    if tokens_acts_list[idx] < threshold
                    else f" <token-of-interest>{token[1:]}</token-of-interest>"
                    if start_space
                    else f"<token-of-interest>{token[:-1]}</token-of-interest> "
                    if start_space is False
                    else f"<token-of-interest>{token}</token-of-interest>"
                )
                if tokens_acts_list[idx] > threshold:
                    substrings.append(
                        "".join(
                            tokens[max(0, idx - 15) : idx]
                            + [effective_token]
                            + tokens[idx : min(idx + 15, len(tokens))]
                        )
                    )
                text += effective_token
            full_examples.append({"example": text, "substrings": substrings})
        quantile_data = {
            "node_id": node_id,
            "quantile_name": quantile_name,
            "examples": full_examples,
        }
        break
    os.makedirs(graph, exist_ok=True)
    with open(f"{graph}/{index}.json", "w") as f:
        json.dump(quantile_data, f)
