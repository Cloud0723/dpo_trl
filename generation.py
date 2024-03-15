import torch
from transformers import AutoConfig, GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM
import datasets
import json


name="model_Mistral"
if name=="huggyllama_test":
    checkpoint="./model_llama/huggyllama_test_5e-7/checkpoint-1000"
if name=="model_Mistral":
    checkpoint="./model_Mistral/ultra_hh_sft_2/checkpoint-1000"
if name=="DPO":
    checkpoint="./model_llama/ultra_hh_dpo/checkpoint-1000"
if name=="SFT":
    checkpoint="./model_llama/ultra_hh/checkpoint-22000"

if __name__ == "__main__":
    # parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelConfig))
    # args, training_args, model_config = parser.parse_args_into_dataclasses()

    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model = model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    result=[]
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    for example in eval_set:
        # tokens = tokenizer(temp_instruction, return_tensors="pt").to("cuda:0")
        tokens = tokenizer(example["instruction"], return_tensors="pt").to("cuda:0")
        # output_start_ix = len(example["instruction"])
        example["output"]=model.generate(**tokens,max_new_tokens=512)
        decoded_output=tokenizer.decode(example["output"][0],skip_special_tokens=False)
        example["output"]=decoded_output
        result.append(example)
        print(len(result))
        if len(result)>50:
            break
    with open(f"./eval_data/{name}_output.json",'w') as f:
        json.dump(result,f,indent=4)
