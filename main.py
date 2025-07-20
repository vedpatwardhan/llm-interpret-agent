import sys
import json
from utils import (
    get_all_relatedness,
    get_graph_nodes_and_links,
    get_url,
    recursive_grouping,
    store_nodes_and_examples,
)


def main():
    overall_goal = """
The prompt I'm providing is related to Michael Jordan and basketball.

We see that the output of the model is in spanish, just like the prompt.

The goal here for me is to understand how the model manipulates the response across sports and languages.
"""
    graph = "gemma-michael-jordan-es"
    start_node_ids = ["E_7939_3", "E_18853_4", "E_113501_5", "E_717_6"]
    end_node_ids = ["27_143831_6"]
    # graph_data, start_links, end_links = get_graph_nodes_and_links(
    #     graph, start_node_ids, end_node_ids
    # )

    # store_nodes_and_examples(graph_data, start_links, end_links, overall_goal)

    # nodes = get_all_relatedness(graph)
    # with open(f"circuit-tracer/{graph}/relatedness.json", "w") as f:
    #     json.dump(nodes, f)

    # final_groups = recursive_grouping(nodes)

    # classified_nodes = []
    # for group in final_groups:
    #     for node in group["nodes"]:
    #         classified_nodes.append(node)
    # with open(f"circuit-tracer/{graph}/classified_nodes.json", "w") as f:
    #     json.dump(classified_nodes, f)

    # url = get_url(graph, classified_nodes)
    # print(url)


if __name__ == "__main__":
    main()
