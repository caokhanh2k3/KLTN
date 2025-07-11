# from exp.exp_model import Exp_Model
import argparse
# import torch
import numpy as np
import random
# from explain_module.util import summarize_trial, remove_reflections, save_results#, save_agents

fix_seed = 100
random.seed(fix_seed)
# torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='generating')

# load data
parser.add_argument("--price_dir", type=str, default="data/sample_price/preprocessed/")
parser.add_argument("--tweet_dir", type=str, default="data/sample_tweet/raw/")
parser.add_argument("--seq_len", type=int, default=5)

parser.add_argument("--technical_indicator_dir", type=str, default="data/sample_price/technical_indicator/") #****
parser.add_argument("--llm_summarize", type=str, default="OpenAILLM") #**** # OpenAILLM // DeepSeekLLM


# supervised finetuning
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="./data/merge_sample.json")
parser.add_argument("--output_path", type=str, default="./saved_models/lora-Vicuna")
parser.add_argument("--model_path", type=str, default="lmsys/vicuna-7b-v1.5-16k")
parser.add_argument("--eval_steps", type=int, default=200)
parser.add_argument("--save_steps", type=int, default=200)
parser.add_argument("--resume_from_supervised_checkpoint", type=str, default=None)
parser.add_argument("--ignore_data_skip", type=str, default="False")

# training reward model
parser.add_argument("--num_reflect_trials", type=int, default=2)
parser.add_argument("--datasets_dir", type=str, default="./datasets/")
parser.add_argument('--local_rank', type=int, default=0, help="Used for multi-gpu")
parser.add_argument('--resume_from_reward_checkpoint', type=bool, default=False, help="If you want to resume training where it left off.")
parser.add_argument('--deepspeed', type=str, default=None, help="Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU.")
parser.add_argument('--per_device_train_batch_size', type=int, default=1)
parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
parser.add_argument('--reward_gradient_accumulation_steps', type=int, default=32)
parser.add_argument('--reward_learning_rate', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=int, default=0.001)
parser.add_argument('--reward_base_model', type=str, default="lmsys/vicuna-7b-v1.5-16k", help="The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc.")
parser.add_argument('--bf16', type=bool, default=False, help="This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU.")
parser.add_argument('--num_train_epochs', type=int, default=1, help="The number of training epochs for the reward model.")
parser.add_argument('--train_subset', type=int, default=100000, help="The size of the subset of the training data to use")
parser.add_argument('--eval_subset', type=int, default=50000, help="The size of the subset of the eval data to use")
parser.add_argument('--gradient_checkpointing', type=bool, default=False, help="Enables gradient checkpointing.")
parser.add_argument('--optim', type=str, default="adamw_hf", help="Enables gradient checkpointing.")
parser.add_argument('--lr_scheduler_type', type=str, default="linear", help="The lr scheduler")
parser.add_argument('--reward_adapter', type=str, default="./saved_models/reward_model_vicuna-7b")

# reinforcement learning
parser.add_argument('--rl_base_model', type=str, default="./saved_models/lora-Vicuna-adapter-merged", help="the model name")
parser.add_argument('--tokenizer_name', type=str, default="lmsys/vicuna-7b-v1.5-16k", help="the tokenizer name")
parser.add_argument('--reward_model_name', type=str, default="./saved_models/reward_model_vicuna-7b-adapter-merged", help="the reward model name")
parser.add_argument('--log_with', type=str, default=None, help="use 'wandb' to log with wandb")
parser.add_argument('--rl_learning_rate', type=float, default=1.4e-5, help="the learning rate")
parser.add_argument('--output_max_length', type=int, default=128, help="maximum length for generation")
parser.add_argument('--mini_batch_size', type=int, default=1, help="the PPO minibatch size")
parser.add_argument('--batch_size', type=int, default=1, help="the batch size")
parser.add_argument('--ppo_epochs', type=int, default=4, help="the number of ppo epochs")
parser.add_argument('--rl_gradient_accumulation_steps', type=int, default=1, help="the number of gradient accumulation steps")
parser.add_argument('--adafactor', type=bool, default=False, help="whether to use the adafactor optimizer")
parser.add_argument('--early_stopping', type=bool, default=True, help="whether to early stop")
parser.add_argument('--target_kl', type=float, default=0.1, help="kl target for early stopping")
parser.add_argument('--reward_baseline', type=float, default=0, help="a baseline value that is subtracted from the reward")
parser.add_argument('--batched_gen', type=bool, default=True, help="whether to use the batched text gen")
parser.add_argument('--save_freq', type=int, default=None, help="n steps to save the model")
parser.add_argument('--output_dir', type=str, default="./saved_models/tuning_llama_rl_checkpoints/", help="directory to save the model")
parser.add_argument('--seed', type=int, default=0, help="the seed")

# evaluation
parser.add_argument("--num_shots", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="results/")

args = parser.parse_args()
print('Args in experiment:')
# print(args)

import pandas as pd
# Đọc tệp CSV
df_loaded = pd.read_csv("data_sample2.csv")

# Hiển thị nội dung DataFrame
print("Nội dung tệp CSV:")
# print(df_loaded)
# df_loaded = df_loaded[:2]

from explain_module.util import summarize_trial, remove_reflections, save_results#, save_agents
from explain_module.agents import PredictReflectAgent
import os, json
agent_cls = PredictReflectAgent
agents = [agent_cls(row['ticker'], row['summary'], row['target']) for _, row in df_loaded.iterrows()]
print("Loaded Train Agents.")
agents

for agent in agents:
    agent.run()

    if agent.is_correct():
        prompt = agent._build_agent_prompt()
        response = agent.scratchpad.split('Price Movement: ')[-1]
        sample = {"instruction": prompt, "input": "", "output": response}
        with open(args.data_path, 'a') as f:
            f.write(json.dumps(sample) + "\n")
correct, incorrect = summarize_trial(agents)
print(f'Finished Trial 0, Correct: {len(correct)}, Incorrect: {len(incorrect)}')

# # Train supervised policy
# supervised_finetune(self.args)
# merge_peft_adapter(model_name=self.args.output_path, output_name=self.args.rl_base_model)
print('===================================================')
print('Collect comparison data')

# Collect comparison data
comparison_data = []
for trial in range(args.num_reflect_trials):
    for idx, agent in enumerate([a for a in agents if not a.is_correct()]):
        prev_response = agent.scratchpad.split('Price Movement: ')[-1]
        agent.run()
        if agent.is_correct():
            print(agent._build_agent_prompt(), "\n\n\n")
            prompt = remove_reflections(agent._build_agent_prompt())
            response = agent.scratchpad.split('Price Movement: ')[-1]
            sample = {"user_input": prompt, "completion_a": prev_response, "completion_b": response}
            comparison_data.append(sample)
            # print("ádnasikdnaskd")
    correct, incorrect = summarize_trial(agents)
    print(f'Finished Trial {trial+1}, Correct: {len(correct)}, Incorrect: {len(incorrect)}')
os.makedirs(args.datasets_dir, exist_ok=True)
comparison_data_path = os.path.join(args.datasets_dir, "comparison_data.json")
if comparison_data:
    with open(comparison_data_path, 'w') as f:
        f.write(json.dumps(comparison_data))