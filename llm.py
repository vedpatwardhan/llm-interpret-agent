import json
from litellm import completion, RateLimitError
import time
from models import ClassificationResponse, GroupingResponse, RelatednessResponse
from prompts import (
    classification_system_prompt,
    grouping_system_prompt,
    relatedness_system_prompt,
)


def rate_limit_completion(fn, *args, **kwargs):
    """
    Wrapper for the litellm completion function to retry on rate limit errors.
    """
    try:
        response = completion(*args, **kwargs)
        with open("latest_response.json", "w") as f:
            json.dump(response.model_dump(), f)
        return response
    except RateLimitError:
        print(f"We have hit the rate limits at {fn}!")
        seconds = 15
        while seconds > 0:
            print(
                f"Next try to call the API in {'' if seconds >= 10 else '0'}{seconds} seconds",
                end="\r",
            )
            seconds -= 1
            time.sleep(1)
        return rate_limit_completion(fn, *args, **kwargs)


def get_relatedness(graph: str, feature_id: str):
    with open(f"{graph}/{feature_id}.json") as f:
        feature = json.load(f)
    examples = feature.get("examples", [])

    with open(f"{graph}/metadata.json") as f:
        data = json.load(f)
        prompt, output = data["prompt"], data["output"]

    relatedness_user_prompt = f"""
PROMPT: {prompt}

OUTPUT: {output}

FEATURE:
------------
    Examples:
    ------------
    """

    for idx, example in enumerate(examples):
        relatedness_user_prompt += f"""
    Example {idx + 1}:
        Full Text: {example["example"]}
        Substrings:
    """
        for substring in example["substrings"]:
            relatedness_user_prompt += f"""            - {substring}
    """

    response = RelatednessResponse(
        **json.loads(
            rate_limit_completion(
                fn="get_relatedness",
                model="gemini/gemini-2.5-flash-lite-preview-06-17",
                messages=[
                    {"role": "system", "content": relatedness_system_prompt},
                    {"role": "user", "content": relatedness_user_prompt},
                ],
                response_format={
                    "type": "json_object",
                    "response_schema": RelatednessResponse.model_json_schema(),
                },
            )
            .choices[0]
            .message.content
        )
    )
    return feature["node_id"], response


def get_grouping(graph: str, nodes: list[dict[str, str]], current_group: str):
    # filter the nodes to ignore not related ones
    nodes = list(filter(lambda node: node["final_verdict"] != "NOT RELATED", nodes))

    # load prompt, output and the overall goal
    with open(f"{graph}/metadata.json") as f:
        data = json.load(f)
        prompt, output, overall_goal = (
            data["prompt"],
            data["output"],
            data["overall_goal"],
        )

    # construct user prompt
    grouping_user_prompt = f"""
PROMPT: {prompt}

OUTPUT: {output}

OVERALL GOAL:
------------
{overall_goal}
------------

CURRENT GROUP: {current_group}

NODES:
------------
    """
    for idx, node in enumerate(nodes):
        grouping_user_prompt += f"""
    Node {idx + 1}:
        Justification: {node["justification"]}
        Verdict: {node["final_verdict"]}
        """

    # get groups
    groups = GroupingResponse(
        **json.loads(
            rate_limit_completion(
                fn="get_grouping",
                model="gemini/gemini-2.5-flash-lite-preview-06-17",
                messages=[
                    {"role": "system", "content": grouping_system_prompt},
                    {"role": "user", "content": grouping_user_prompt},
                ],
                response_format={
                    "type": "json_object",
                    "response_schema": GroupingResponse.model_json_schema(),
                },
            )
            .choices[0]
            .message.content
        )
    )

    return groups


def classify_node(
    classification_user_prompt: str,
    node: dict[str, str],
):
    return ClassificationResponse(
        **json.loads(
            rate_limit_completion(
                fn="classify_node",
                model="gemini/gemini-2.5-flash-lite-preview-06-17",
                messages=[
                    {"role": "system", "content": classification_system_prompt},
                    {
                        "role": "user",
                        "content": classification_user_prompt
                        + f"""
NODE:
    Verdict: {node["final_verdict"]}
    Justification: {node["justification"]}
                    """,
                    },
                ],
                response_format={
                    "type": "json_object",
                    "response_schema": ClassificationResponse.model_json_schema(),
                },
            )
            .choices[0]
            .message.content
        )
    ).group_title
