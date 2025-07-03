# Models: o4-mini-2025-04-16, gpt-4.1-nano-2025-04-14, o3-mini-2025-01-31, gpt-3.5-turbo-0125, gpt-4o-2024-08-06, gpt-4-0613 (expensive)

python dragonExp.py \
    --model o4-mini-2025-04-16 \
    --exp_name o4-mini-2025-04-16 \
    --allow_comm \
    --tom \
    --save_path data/ \
    --tool_per_agent 2 \
    --temperature 0 \
    --max_step 30 \
    --cutoff 0 \
    --memory_size 2 \
    --seed 3

# Default arguments
#    --save_path data/ \
#    --seed 0 \
#    --model gpt-4-turbo-preview \
#    --tool_per_agent 2 \
#    --temperature 0 \
#    --max_step 30 \
#    --exp_name default_exp \
#    --cutoff 0.0 \
#    --memory_size 2
# Optional flags (default: off, add to enable):
#   --tom
#   --belief
#   --allow_comm
#   --include_agent_action
#   --tom_reasoning
#   --improved
#   --tips