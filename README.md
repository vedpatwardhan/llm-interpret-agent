## LLM Interpret Agent

An LLM-powered agent designed to streamline and enhance the process of discovering and analyzing prominent computational circuits within large language models, leveraging the [circuit-tracer](https://github.com/safety-research/circuit-tracer).


### Setting Up

#### Set up the project folder

```
git clone https://github.com/vedpatwardhan/llm-interpret-agent.git
cd llm-interpret-agent
uv venv
uv sync
git submodule update --init
cd circuit-tracer
mkdir graph_files
cp graph-metadata.json circuit-tracer/graph_files
uv pip install -e circuit_tracer
```

#### Generate graph from Neuronpedia

Go to [Neuronpedia](https://www.neuronpedia.org/), generate a new graph or use an existing graph, and download the json for it into the `circuit-tracer/graph_files` folder (through the `Graph Info` option).

Then copy over the "metadata" in the downloaded json to the `circuit-tracer/graph_files/graph-metadata.json` as a list item under `"graphs"`.

Start the circuit tracer server,

```
uv run start_server.py
```

#### Set Params for Analysis

In order to do the analysis, you'd need to view the attribution graph for your chosen example through the server.

Then identify the input and output nodes you care about along with the overall goal, and edit the `main.py` with those accordingly (be sure to use the node ids in the format that matches to the `clickedId` in the url after you click it).

#### Run the script

```
uv run main.py
```

Once done, it will output a url where you can view the grouping (provided the circuit-tracer server is running).


### General Idea

1. Select the input and output nodes that we want to analyze, along with the overall goal we're trying to achieve with the analysis.
2. Select the top feature nodes associated with either of the input and output nodes on the influence.
3. Check each such node for its relevance to the input-output behaviour of the model based on the top 1% examples.
4. Recursively break down the nodes into separate groups, directed by their contents and the overall goal we're trying to achieve. After every grouping step, classify the nodes in that group among its sub-groups and iterate on those that still have more than 5 nodes.
5. Generate the url to view the attribution graph with those nodes pinned and grouped.

![attribution](attribution.png)

(The localhost url for this particular graph is [here](http://localhost:8046/?slug=gemma-michael-jordan-es&clerps=%5B%5D&pinnedIds=2_2808_3%2C0_14975_3%2C15_15208_6%2C6_7377_5%2C23_14713_6%2C0_8003_4%2C2_13198_4%2C23_5458_6%2C24_3018_6%2C1_8628_6%2C6_7157_5%2C23_8855_6%2C3_4335_3%2C21_4818_6%2C4_14954_6%2C1_15876_6%2C21_9324_6%2C1_1173_4%2C24_3865_6%2C2_10957_6%2C4_1305_6%2C25_13416_6%2C3_14567_6%2C17_10384_6%2C18_6953_4%2CE_7939_3%2CE_18853_4%2CE_113501_5%2CE_717_6%2C27_143831_6&supernodes=%5B%5B%22Recognizing+Michael+Jordan+and+the+concept+of+playing+sports%22%2C+%222_2808_3%22%2C+%220_14975_3%22%2C+%2215_15208_6%22%5D%2C%5B%22Language+and+context+processing%22%2C+%226_7377_5%22%2C+%2223_14713_6%22%2C+%220_8003_4%22%5D%2C%5B%22Tangential+or+indirectly+related+concepts%22%2C+%222_13198_4%22%5D%2C%5B%22Recognizing+Sports+Vocabulary+and+Related+Terms%22%2C+%2223_5458_6%22%5D%2C%5B%22Listing+and+Categorizing+Sports%22%2C+%2224_3018_6%22%2C+%221_8628_6%22%5D%2C%5B%22Identify+Michael+Jordan*s+Sport%22%2C+%226_7157_5%22%2C+%2223_8855_6%22%2C+%223_4335_3%22%2C+%2221_4818_6%22%2C+%224_14954_6%22%5D%2C%5B%22List+Related+Sports%22%2C+%221_15876_6%22%5D%2C%5B%22Core+Concept+Recognition%22%2C+%2221_9324_6%22%2C+%221_1173_4%22%5D%2C%5B%22Sports+and+Related+Activities%22%2C+%2224_3865_6%22%2C+%222_10957_6%22%2C+%224_1305_6%22%2C+%2225_13416_6%22%2C+%223_14567_6%22%5D%2C%5B%22Linguistic+and+Contextual+Clues%22%2C+%2217_10384_6%22%2C+%2218_6953_4%22%5D%5D&clickedId=2_2808_3))


### Use-Case

1. The first section of [this](https://github.com/safety-research/circuit-tracer/blob/main/demos/intervention_demo.ipynb) example demonstrates how interventions on certain features can vastly affect the output of the model.
2. My goal here was primarily to understand the effect of intervening on nodes involved in recognizing Michael Jordan to understand how much they affect the identification of the sport.
2. I wanted to have a repeatable process to generate those supernodes in order to quickly find more such nodes that would affect the output.
3. Using some of the nodes selected in graph, I was able to significantly reduce the probability of the sport identified (available in the `demo.ipynb`).

![intervention](intervention.png)


### Future Improvements

1. There seems some sort of redundancy where there's other features that hold similar information to the ones intervened but aren't activated at first, so it would be useful to have an iterative process to identify new nodes that demonstrate this behaviour.
2. The agent doesn't have any control over the actual interventions and observe the effects, so end-to-end access to the full process can improve performance further.
