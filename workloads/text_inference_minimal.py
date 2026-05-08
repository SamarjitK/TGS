import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer import Trainer
import utils
import time

import argparse

parser = argparse.ArgumentParser(description='Minimal text inference workload',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='distilgpt2', help='HuggingFace model id')
parser.add_argument('--max-new-tokens', type=int, default=128, help='tokens to generate per prompt chunk')
parser.add_argument('--prompt-batch-size', type=int, default=4, help='number of prompts to generate per TGS iteration')
parser.add_argument('--iterations', type=int, default=100, help='number of TGS iterations to report')
parser.add_argument('--prompt', type=str, default=None, help='single prompt to use')
parser.add_argument('--prompts-file', type=str, default=None, help='file with one prompt per line')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--scheduler_ip', type=str, required=True)
parser.add_argument('--scheduler_port', type=int, default=6889)
parser.add_argument('--trainer_port', type=int)
parser.add_argument('--job_id', type=int, default=-1)


def load_prompts(args):
    if args.prompts_file is not None and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        return lines
    if args.prompt is not None:
        return [args.prompt] * args.iterations
    # default prompts
    return [
        "Hello, how are you?",
        "Tell me a short story about a robot.",
        "Summarize the following: Machine learning is...",
        "Translate to French: The quick brown fox jumps over the lazy dog.",
        "Write a one-line poem about coffee."
    ]


if __name__ == '__main__':
    print("Starting minimal text inference workload")
    args = parser.parse_args()

    args.cuda = not args.no_cuda and hasattr(sys, 'getsizeof') and False
    # We detect CUDA at runtime via torch if available
    try:
        import torch
        args.cuda = not args.no_cuda and torch.cuda.is_available()
    except Exception:
        args.cuda = False

    # Initialize trainer with batch_size=1 (one prompt per TGS iteration)
    trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, args.job_id, 1)

    prompts = load_prompts(args)

    # Each TGS iteration now runs one batched generation call so the GPU does more work
    # before we report back to the scheduler.
    completed_iterations = 0
    prompt_idx = 0

    # Try to use transformers pipeline if available
    try:
        # Prefer explicit model/tokenizer so we can set pad token and avoid warnings
        from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
        device = 0 if args.cuda else -1
        set_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        # ensure pad token exists to silence warnings
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else ""

        model = AutoModelForCausalLM.from_pretrained(args.model)
        # ensure model config pad token id is set
        if getattr(model.config, 'pad_token_id', None) is None and getattr(model.config, 'eos_token_id', None) is not None:
            model.config.pad_token_id = model.config.eos_token_id

        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    except Exception:
        generator = None
        print("Transformers not available or failed to load, falling back to dummy generation")

    while completed_iterations < args.iterations:
        batch_prompts = []
        for _ in range(max(1, args.prompt_batch_size)):
            batch_prompts.append(prompts[prompt_idx])
            prompt_idx = (prompt_idx + 1) % len(prompts)

        start = time.time()
        if generator is not None:
            try:
                _ = generator(
                    batch_prompts,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.config.eos_token_id,
                )
            except Exception:
                # On generation error, fallback to dummy
                time.sleep(0.01 * max(1, len(batch_prompts)))
        else:
            # Dummy generation: produce simple characters to simulate work
            time.sleep(0.01 * max(1, len(batch_prompts)))

        elapsed = time.time() - start
        trainer.record(elapsed)
        completed_iterations += 1

    try:
        trainer.close()
    except Exception:
        pass
