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

Next, install the dependencies for the LLM agents:

```
pip install -r requirements.txt
```

Finally, configure the environment variable for your inference provider. Use
`OPENAI_API_KEY` for OpenAI or `GROQ_API_KEY` for Groq. Never commit API keys to
the repository; see OpenAI's [API key safety guidance](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

## Running

Once everything is installed, the recommended way to run experiments is with `run.sh`.

```bash
bash run.sh
```

All experiment parameters and flags are defined in `run.sh` (including defaults and optional flags). Edit that file to configure your runs.

You can also run directly with `python dragonExp.py` using commands like the examples below.

### Model providers

OpenAI remains the default provider, so existing commands continue to work. It
can also be selected explicitly:

```bash
python dragonExp.py \
  --provider openai \
  --model gpt-4.1-nano-2025-04-14 \
  --exp_name openai-smoke \
  --preset default \
  --max_step 3
```

Groq uses its OpenAI-compatible Chat Completions endpoint:

```bash
python dragonExp.py \
  --provider groq \
  --model llama-3.1-8b-instant \
  --exp_name groq-smoke \
  --preset default \
  --allow_comm \
  --max_step 3
```

#### Local Transformers inference

The `local` provider loads one Transformers model and shares it across all
agents. No API key or inference server is required. The default local model is
`Llama-3.1-8B-Instruct`, loaded from the existing Hugging Face cache at
`/usr/users/xai/gama_hei/projects/llm_dce/models/huggingface`.

On the GPU compute node, install the local dependencies into `tom-heavy` once:

```bash
conda activate tom-heavy
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-local.txt
```

Verify that the allocated GPU is visible:

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Then run the three-round smoke experiment:

```bash
bash run_local.sh
```

The equivalent explicit command is:

```bash
python dragonExp.py \
  --provider local \
  --model Llama-3.1-8B-Instruct \
  --model_path meta-llama/Llama-3.1-8B-Instruct \
  --model_cache_dir /usr/users/xai/gama_hei/projects/llm_dce/models/huggingface \
  --local_dtype float16 \
  --max_completion_tokens 256 \
  --exp_name llama-local-smoke \
  --preset default \
  --allow_comm \
  --max_step 3 \
  --temperature 0 \
  --seed 0
```

`run_local.sh` accepts additional arguments, whose later values override its
smoke-test defaults. For example, `bash run_local.sh --max_step 30 --seed 3`.

After the interactive GPU smoke test succeeds, submit the full map suite with:

```bash
mkdir -p logs
sbatch jobs/llama_local_all_presets.sbatch
```

The batch runs `village`, `default`, `medium`, `hard`, and `extreme`
sequentially on one GPU for 30 rounds with communication enabled, repeating
each preset with seeds 0, 1, and 2 (15 experiments total). `easy` is excluded
from this batch by design. Seeds, round count, output length, and the optional
ToM measurements are configurable near the top of the batch file. Monitor it
with `squeue -u "$USER"` and inspect
`logs/llama31_marl_maps-<job-id>.out`.

Any OpenAI-compatible server can be selected with an explicit base URL:

```bash
python dragonExp.py \
  --provider openai-compatible \
  --base_url http://localhost:8000/v1 \
  --model your-served-model \
  --exp_name local-smoke \
  --preset default \
  --max_step 3
```

Provider usage fields are normalized and accumulated in `results.json` while
retaining the top-level `prompt_tokens` and `completion_tokens` fields used by
the dashboard. Each `chat_log.jsonl` request also records latency, available
usage fields, request identifiers, and backend fingerprints. Missing usage
metadata is recorded but does not stop an experiment. `runtime_metrics` in
`results.json` captures wall time, process CPU time, peak memory, and basic host
information for cross-backend comparisons. Local runs additionally record model
load time, parameter count, PyTorch and Transformers versions, GPU identity,
CUDA memory usage, exact tokenizer counts, generation time, and aggregate output
tokens per second.

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
