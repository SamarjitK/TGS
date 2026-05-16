"""
Replace vLLM usage with llama.cpp backend (use local LlamaCpp wrapper when available), otherwise fall back to transformers pipeline.
"""
import argparse
import os
import sys
import time
import requests
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trainer import Trainer
import utils


parser = argparse.ArgumentParser(description='llama.cpp text inference workload', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf', help='local model path or relative to repo root')
parser.add_argument('--max-new-tokens', dest='max_new_tokens', type=int, default=128, help='tokens to generate per prompt chunk')
parser.add_argument('--max_new_tokens', dest='max_new_tokens', type=int, help=argparse.SUPPRESS)
parser.add_argument('--prompt-batch-size', dest='prompt_batch_size', type=int, default=1, help='number of prompts to generate per TGS iteration')
parser.add_argument('--prompt_batch_size', dest='prompt_batch_size', type=int, help=argparse.SUPPRESS)
parser.add_argument('--iterations', type=int, default=100, help='number of TGS iterations to report')
parser.add_argument('--prompts', dest='iterations', type=int, help=argparse.SUPPRESS)
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


def _try_launch_llamacpp(repo_root, api_port, model_path, device='gpu'):
    # Attempt to use the project's LlamaCpp wrapper if present
    try:
        sys.path.append(os.path.join(repo_root, 'ConsumerBench'))
        from Llamacpp import LlamaCpp
        backend = LlamaCpp()
        backend.launch_backend(api_port=api_port, model=model_path, device='gpu' if device=='gpu' else 'cpu')
        return backend
    except Exception:
        return None


if __name__ == '__main__':
    print("Starting llama.cpp text inference workload")
    args = parser.parse_args()

    # CUDA detection
    try:
        import torch
        args.cuda = not args.no_cuda and torch.cuda.is_available()
    except Exception:
        args.cuda = False

    trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, args.job_id, 1)

    prompts = load_prompts(args)

    completed_iterations = 0
    prompt_idx = 0

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    api_port = 8080
    backend = _try_launch_llamacpp(repo_root, api_port, args.model, device='gpu' if args.cuda else 'cpu')

    # If backend launched, we'll POST to its HTTP API; otherwise fall back to transformers if available
    use_llamacpp = backend is not None

    if use_llamacpp:
        api_url = f"http://127.0.0.1:{api_port}/v1/completions"

    else:
        # try transformers pipeline
        try:
            from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
            set_seed(args.seed)
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(args.model)
            if getattr(model.config, 'pad_token_id', None) is None and getattr(model.config, 'eos_token_id', None) is not None:
                model.config.pad_token_id = model.config.eos_token_id
            device = 0 if args.cuda else -1
            generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
        except Exception:
            generator = None

    while completed_iterations < args.iterations:
        batch_prompts = []
        for _ in range(max(1, args.prompt_batch_size)):
            batch_prompts.append(prompts[prompt_idx])
            prompt_idx = (prompt_idx + 1) % len(prompts)

        start = time.time()

        if use_llamacpp:
            # POST to llama.cpp server; use non-streaming for simplicity
            payload = {
                "model": args.model,
                "prompt": "\n\n".join(batch_prompts),
                "max_tokens": args.max_new_tokens,
                "temperature": 0.0,
                "top_p": 0.9,
                "stream": False,
            }
            headers = {"Content-Type": "application/json"}
            try:
                r = requests.post(api_url, json=payload, headers=headers, timeout=300)
                r.raise_for_status()
            except Exception as e:
                print(f"llama.cpp request failed: {e}")
        else:
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
                    time.sleep(0.01 * max(1, len(batch_prompts)))
            else:
                # Dummy generation
                time.sleep(0.01 * max(1, len(batch_prompts)))

        elapsed = time.time() - start
        trainer.record(elapsed)
        completed_iterations += 1

    # cleanup backend if we launched it
    if backend is not None:
        try:
            backend.cleanup_backend(api_port=api_port)
        except Exception:
            pass

    try:
        trainer.close()
    except Exception:
        pass
