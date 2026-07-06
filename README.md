# When LLM Agent Teams Fail at Topological Spatial Reasoning Under Partial Observability

This repository contains the implementation used in [**When LLM Agent Teams Fail at Topological Spatial Reasoning Under Partial Observability**](https://doi.org/10.65109/RTFT9958), published as an extended abstract in the proceedings of AAMAS 2026. The paper investigates the topological spatial-reasoning capabilities of LLM agent teams under partial observability.

The codebase originated as a fork of Huao Li et al.'s multi-LLM Theory of Mind implementation for the EMNLP 2023 paper [Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://aclanthology.org/2023.emnlp-main.13/). The current repository has since been substantially extended and is maintained as the codebase accompanying the AAMAS 2026 publication.

Compared with the upstream project, this version:

- Corrects bugs from the earlier codebase.
- Expands the framework with communication cutoff experiments.
- Improves scalability across different map sizes.
- Adds graph-compression and topological spatial-reasoning experiments.
- Adds/extends visualization support for analysis and debugging.

## Citation

If you use this repository, its experiments, or its extensions in your research, please cite the AAMAS 2026 publication:

```bibtex
@inproceedings{gama2026llm,
  title     = {When {LLM} Agent Teams Fail at Topological Spatial Reasoning Under Partial Observability},
  author    = {Gama, Heitor and Diguet, Jean-Philippe and Ranasinghe, Damith C.},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems},
  pages     = {3567--3569},
  year      = {2026},
  doi       = {10.65109/RTFT9958},
  url       = {https://doi.org/10.65109/RTFT9958}
}
```

This repository builds on the original EMNLP 2023 implementation. Please also cite the upstream work when your use relies on that contribution:

```bibtex
@inproceedings{li2023theory,
  title={Theory of Mind for Multi-Agent Collaboration via Large Language Models},
  author={Li, Huao and Chong, Yu and Stepputtis, Simon and Campbell, Joseph P and Hughes, Dana and Lewis, Charles and Sycara, Katia},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  pages={180--192},
  year={2023}
}
```


## Installation

First, install the USAR environment gym-dragon.

```
# downgrade setuptools for gym=0.21
pip install setuptools==65.5.0 "wheel<0.40.0"
cd gym-dragon
pip install -e .
```

Next, we need to install dependencies for LLM agents including openai API package. For doing that run:

```
pip install -r requirements.txt
```

Finally, we need to set up the openai API key in your system environment variables as instructed in this [guideline](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).
```
echo "export OPENAI_API_KEY='yourkey'" >> ~/.bash_profile
source ~/.bash_profile
echo $OPENAI_API_KEY
```
## Running

Once everything is installed, the recommended way to run experiments is with `run.sh`.

```bash
bash run.sh
```

All experiment parameters and flags are defined in `run.sh` (including defaults and optional flags). Edit that file to configure your runs.

You can also run directly with `python dragonExp.py` using commands like the examples below.

### Urban Search and Rescue (i.e. gym_dragon)
- GPT-4-turbo on mini_dragon with 5 nodes
```
python dragonExp.py --model gpt-4-turbo-preview --exp_name gpt-4 --allow_comm --belief --seed 0
```
- GPT-4-turbo w/o communication on mini_dragon with 5 nodes
```
python dragonExp.py --model gpt-4-turbo-preview --exp_name gpt-4 --belief --seed 0
```
- GPT-4-turbo on mini_dragon with 5 nodes and ToM inference tasks
```
python dragonExp.py --model gpt-4-turbo-preview --exp_name gpt-4 --allow_comm --ToM --belief --seed 0
```

## Contributors

Original repository:
- Huao Li ([@romanlee6](https://github.com/romanlee6))
- Ini Oguntola ([@ini](https://github.com/ini))

Current repository:
- Heitor Gama ([@baunilhamarga](https://github.com/baunilhamarga))

## License

Code is available under MIT license.
