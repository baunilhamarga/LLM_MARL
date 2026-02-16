# Theory of Mind for Multi-Agent Collaboration via Large Language Models

This repository is an extension of the original multi-LLM ToM implementation from the EMNLP 2023 paper, **Theory of Mind for Multi-Agent Collaboration via Large Language Models**, available at [https://aclanthology.org/2023.emnlp-main.13/](https://aclanthology.org/2023.emnlp-main.13/).

Compared with the original project, this extension:
- Corrects bugs from the earlier codebase.
- Expands the framework with communication cutoff experiments.
- Improves scalability across different map sizes.
- Adds/extends visualization support for analysis and debugging.

## Cite

If you use this code in your work, please cite the original work:

```
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

This extension:
- Margarete Baunilha ([@baunilhamarga](https://github.com/baunilhamarga))

## License

Code is available under MIT license.
