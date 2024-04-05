# reference: https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import jsonlines
import warnings
warnings.filterwarnings("ignore")

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='eval_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_dir', type=str, default="data/loser_0.jsonl")
    parser.add_argument('--split', type=str, default='train')
    return parser.parse_args()

def prepare_prompts(prompts, tokenizer, batch_size=4):
    """Prepare prompts for tokenization."""
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

def main():
    args = parse_arguments()
    model_path = args.model
    data_frac = args.data_frac
    batch_size = args.batch_size
    output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,    
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)   
    tokenizer.pad_token = tokenizer.eos_token

    # load data
    data = load_dataset(args.input_dir, split=args.split)
    data = data.shuffle(seed=42)
    print(f"load data with length {len(data)}")
    if args.frac_len > 0:
        sub_len = args.frac_len 
        if sub_len*(data_frac+1) > len(data):
            data = data[sub_len*data_frac:]['real']
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]['real']
    elif args.trun_len > 0:
        data = data[:args.trun_len]['real']
    else:
        data = data[:]['real']


    # prompts_all=data["demon_prompt"]
    # demonstration=data["demon"]
    # prompts_all = ["### Instruction: " + data[idx][0]['content'] + "\n\n### Response: " for idx in range(len(data))]
    prompts_all = [data[idx][0]['content'] for idx in range(len(data))]
    demonstration = [data[idx][1]['content'] for idx in range(len(data))]

    accelerator.wait_for_everyone()    
    start=time.time()
    f=open(args.output_dir,'w')
    writer = jsonlines.Writer(f)
    # divide the prompt list onto the available GPUs 
    with accelerator.split_between_processes(data) as batch_data:
        # print(f"batch data: {batch_data}")
        # exit()
        results = {'prompt':[],'continuation':[],'demonstration':[]}
        prompt_batches=prepare_prompts([data[idx][0]['content'] for idx in range(len(batch_data))], tokenizer, batch_size=args.batch_size)

        for prompts_tokenized in tqdm(prompt_batches):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)

            # remove prompt from gen. tokens
            outputs_tokenized=[ tok_out[len(tok_in):] 
                for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ] 
            # decode gen. tokens 
            outputs=tokenizer.batch_decode(outputs_tokenized)
            results['prompt'].extend([data[idx][0]['content'] for idx in range(len(batch_data))])
            results['continuation'].extend(outputs)
            results['demonstration'].extend([data[idx][1]['content'] for idx in range(len(batch_data))])

    # collect results from all the GPUs and remove paddings
    results_gathered=gather_object([results])
    result_to_save = {'prompt':[],'continuation':[],'demonstration':[]}
    for r in results_gathered:
        for i in range(len(r['continuation'])):
            result_to_save['continuation'].append(r['continuation'][i].replace("</s>","").lstrip())
            result_to_save['prompt'].append(r['prompt'][i])
            result_to_save['demonstration'].append(r['demonstration'][i])
    # results = [r.replace("</s>","").lstrip() for r in results_gathered]

    if accelerator.is_local_main_process:
        timediff=time.time()-start
        print(f"time elapsed: {timediff}")
        for the_prompt, the_continuation, the_demonstration in zip(result_to_save['prompt'], result_to_save['continuation'], result_to_save['demonstration']):
            writer.write({"prompt": the_prompt, "continuation": the_continuation, "demonstration": the_demonstration})
    
        # for i in range(len(results)):
        #     writer.write({"prompts":prompts[i],"agent":results[i], "demon":demonstration[i]})


if __name__ == "__main__":
    main()