import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from runtime.rpc import trainer_client
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Text inference workload that reports TTFT/TPOT to TGS worker.")
    parser.add_argument("--scheduler_ip", type=str, required=True)
    parser.add_argument("--scheduler_port", type=int, default=6889)
    parser.add_argument("--trainer_port", type=int)
    parser.add_argument("--job_id", type=int, default=-1)

    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--prompt", type=str, default="Write a short sentence about GPUs.")
    parser.add_argument("--prompts", type=int, default=50, help="How many prompts to run.")
    parser.add_argument("--max_new_tokens", type=int, default=16, help="How many new tokens to generate per prompt.")
    parser.add_argument("--device", type=str, default="cuda")

    # If transformers isn't available, we fall back to a simple sleep-based token generator.
    parser.add_argument("--fallback_ttft_ms", type=float, default=40.0)
    parser.add_argument("--fallback_tpot_ms", type=float, default=8.0)
    return parser.parse_args()


def try_transformers_measurements(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    else:
        device = "cpu"
        model.to("cpu")

    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    # TTFT proxy: prefill forward + first-token selection.
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize() if device == "cuda" else None
    ttft_ms = (time.perf_counter() - start) * 1000.0

    # TPOT: decode tokens one-by-one (no KV cache; simple + portable).
    tokens_to_generate = max(args.max_new_tokens - 1, 1)
    torch.cuda.synchronize() if device == "cuda" else None
    decode_start = time.perf_counter()
    with torch.no_grad():
        cur = torch.cat([input_ids, next_token], dim=1)
        for _ in range(tokens_to_generate):
            out = model(input_ids=cur)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
    torch.cuda.synchronize() if device == "cuda" else None
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    tpot_ms = decode_ms / tokens_to_generate

    return ttft_ms, tpot_ms


def fallback_measurements(args):
    ttft_ms = args.fallback_ttft_ms
    tpot_ms = args.fallback_tpot_ms
    time.sleep(ttft_ms / 1000.0)
    time.sleep(max(args.max_new_tokens - 1, 1) * tpot_ms / 1000.0)
    return ttft_ms, tpot_ms


def main():
    args = parse_args()
    logger = utils.make_logger(__name__)
    client = trainer_client.TrainerClientForScheduler(logger, args.scheduler_ip, args.scheduler_port)

    logger.info(f"job {args.job_id}, text_infer, start, model={args.model}, prompts={args.prompts}, max_new_tokens={args.max_new_tokens}")

    for prompt_idx in range(args.prompts):
        measured = try_transformers_measurements(args)
        if measured is None:
            ttft_ms, tpot_ms = fallback_measurements(args)
            mode = "fallback"
        else:
            ttft_ms, tpot_ms = measured
            mode = "transformers"

        # finished_iterations: use generated token count so worker still gets a notion of progress.
        finished_iterations = int(args.max_new_tokens)
        ok = client.report_stats(args.job_id, finished_iterations, ttft_ms=ttft_ms, tpot_ms=tpot_ms)
        logger.info(
            f"job {args.job_id}, text_infer, prompt={prompt_idx}, mode={mode}, "
            f"ttft_ms={ttft_ms:.3f}, tpot_ms={tpot_ms:.3f}, report_ok={ok}"
        )


if __name__ == "__main__":
    main()

