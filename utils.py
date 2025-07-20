import json
import matplotlib.pyplot as plt
import os
import requests
from tqdm import tqdm
from typing import Any
from helpers import store_feature, store_metadata
from llm import classify_node, get_grouping, get_relatedness


def get_graph_nodes_and_links(
    graph: str, start_node_ids: list[str], end_node_ids: list[str]
):
    with open(f"circuit-tracer/graph_files/{graph}.json") as f:
        graph_data = json.load(f)

    # create feature type dict
    feature_type = {
        node["node_id"]: node["feature_type"] for node in graph_data["nodes"]
    }

    # get node list
    start_node_list = list(
        filter(
            lambda node: (
                node["node_id"] in start_node_ids
                and feature_type[node["node_id"]] != "mlp reconstruction error"
            ),
            graph_data["nodes"],
        )
    )
    end_node_list = list(
        filter(
            lambda node: (
                node["node_id"] in end_node_ids
                and feature_type[node["node_id"]] != "mlp reconstruction error"
            ),
            graph_data["nodes"],
        )
    )

    # get node dict
    start_nodes = dict(
        zip(map(lambda node: node["node_id"], start_node_list), start_node_list)
    )
    end_nodes = dict(
        zip(map(lambda node: node["node_id"], end_node_list), end_node_list)
    )

    # get links for all start and end nodes
    start_links = {
        start_node: sorted(
            list(
                filter(
                    lambda link: (
                        link["source"] == start_node
                        and feature_type[link["target"]] != "mlp reconstruction error"
                    ),
                    graph_data["links"],
                )
            ),
            key=lambda link: -link["weight"],
        )[:10]
        for start_node in start_nodes
    }
    end_links = {
        end_node: sorted(
            list(
                filter(
                    lambda link: (
                        link["target"] == end_node
                        and feature_type[link["source"]] != "mlp reconstruction error"
                    ),
                    graph_data["links"],
                )
            ),
            key=lambda link: -link["weight"],
        )[:10]
        for end_node in end_nodes
    }

    return graph_data, start_links, end_links


def store_nodes_and_examples(
    graph_data: dict[str, Any],
    start_links: dict[str, dict[str, str]],
    end_links: dict[str, dict[str, str]],
    overall_goal: str,
):
    # get node ids in the links
    linked_node_ids = []
    for links in start_links.values():
        linked_node_ids += list(map(lambda link: link["target"], links))
    for links in end_links.values():
        linked_node_ids += list(map(lambda link: link["source"], links))
    linked_node_ids = list(set(linked_node_ids))

    # get prompt
    prompt = graph_data["metadata"]["prompt"]

    # get output from last layer nodes
    output = "".join(
        [
            node["clerp"].split('"')[1]
            for node in graph_data["nodes"]
            if node["layer"] == "27"
        ][::-1]
    )

    # store metadata
    store_metadata(graph_data["metadata"]["slug"], prompt, output, overall_goal)

    # for each node, get the examples and store them
    skipped_node_ids = []
    for node_id in tqdm(sorted(linked_node_ids)):
        layer = node_id.split("_")[0]
        feature = node_id.split("_")[1]
        if layer != "E":
            # get feature details
            json_name = (layer + feature.zfill(5)) if layer != "0" else feature
            response = requests.get(
                "https://d1fk9w8oratjix.cloudfront.net/"
                f"features/gemma-2-2b/{json_name}.json"
            )

            # skip if request failed
            if response.status_code != 200:
                skipped_node_ids.append(node_id)
                continue

            # store feature
            json_response = response.json()
            store_feature(
                graph_data["metadata"]["slug"],
                node_id,
                json_response["index"],
                json_response["examples_quantiles"],
            )
        else:
            skipped_node_ids.append(node_id)
    print(skipped_node_ids)


def get_all_relatedness(graph: str):
    outputs = []

    # iterate over all features relevant to the graph and get relatedness
    for file in tqdm(os.listdir(graph)):
        if file in [
            "classified.json",
            "groups.json",
            "metadata.json",
            "relatedness.json",
        ]:
            continue
        node_id = file.rstrip(".json")
        original_node_id, relatedness = get_relatedness(graph, node_id)
        outputs.append(
            {
                "original_node_id": original_node_id,
                "node_id": node_id,
                **relatedness.model_dump(),
            }
        )

    # plot the distribution of the verdict
    plt.figure(figsize=(8, 4))
    plt.hist(list(map(lambda op: op["final_verdict"], outputs)))
    plt.show()

    return outputs


def classify_nodes(
    graph: str, nodes: list[dict[str, str]], groups: list[dict[str, str]]
):
    # filter the nodes to ignore not related ones
    nodes = list(filter(lambda node: node["final_verdict"] != "NOT RELATED", nodes))

    # load prompt, output and the overall goal
    with open(f"circuit-tracer/{graph}/metadata.json") as f:
        graph = json.load(f)
        prompt, output, overall_goal = (
            graph["prompt"],
            graph["output"],
            graph["overall_goal"],
        )

    # classify each node
    group_counts = {group["title"]: 0 for group in groups}
    for node in tqdm(nodes):
        # construct user prompt
        classification_user_prompt = f"""
PROMPT: {prompt}

OUTPUT: {output}

OVERALL GOAL:
------------
{overall_goal}
------------

GROUPS:
------------
        """
        for idx, group in enumerate(groups):
            classification_user_prompt += f"""
    Group {idx + 1}:
        Title: {group["title"]}
        Description: {group["description"]}
        Count: {group_counts[group["title"]]}
        """

        # classify node
        node["group"] = classify_node(classification_user_prompt, node)
        group_counts[node["group"]] += 1

    # plot distribution of groups
    plt.figure(figsize=(8, 4))
    plt.hist(list(map(lambda node: node["group"], nodes)))
    plt.show()

    return nodes


def recursive_grouping(graph: str, nodes: list[dict[str, str]]):
    # initialize the process
    groups_to_process = [{"name": "Root", "nodes": nodes}]
    final_groups = []
    max_iter = 8

    while len(groups_to_process) > 0 and max_iter > 0:
        # get current group
        group = groups_to_process.pop(0)
        print("\nCurrent group:", group["name"])

        # create sub-groups and classify all nodes into them
        sub_groups = get_grouping(graph, group["nodes"], group).model_dump()["groups"]
        classified_nodes = classify_nodes(graph, group["nodes"], sub_groups)

        # reorganize the classifications
        sub_group_dict = {
            group["title"]: list(
                filter(lambda node: node["group"] == group["title"], classified_nodes)
            )
            for group in sub_groups
        }

        # iterate over all groups
        print("Next groups: ")
        for group_name in sub_group_dict:
            print("    ", group_name)
            nodes = sub_group_dict[group_name]

            # if we classified all nodes to a single group
            if len(nodes) == len(group["nodes"]) or len(nodes) == 0:
                print("Classification locked in to a single class...")
                final_groups.append({"title": group_name, "nodes": nodes})
                continue

            # if group larger than 5 nodes then repeat the process on that group
            if len(nodes) > 5:
                groups_to_process.append({"name": group_name, "nodes": nodes})
            else:
                final_groups.append({"title": group_name, "nodes": nodes})

        # check number of iterations
        max_iter -= 1
        if max_iter == 0:
            print("Max iterations reached...")
            break
    return final_groups


def get_url(graph: str, classified_nodes: list[dict[str, str]]):
    # pin all nodes
    url = f"http://localhost:8046/?slug={graph}&clerps=[]&pinnedIds="
    url += ",".join(list(map(lambda node: node["original_node_id"], classified_nodes)))

    # reorganize node groups
    node_groups = dict()
    for node in classified_nodes:
        if node["final_verdict"] == "NOT RELATED":
            continue
        if node["group"] in node_groups:
            node_groups[node["group"]].append(node["original_node_id"])
        else:
            node_groups[node["group"]] = [node["original_node_id"]]

    # add supernodes to the url
    url += "&supernodes=["
    for group in node_groups:
        url += json.dumps([group.replace("'", "*"), *node_groups[group]]) + ","
    url = url[:-1]
    url += "]"

    return url
