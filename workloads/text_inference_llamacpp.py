import argparse
import os
import signal
import subprocess
import sys
import time

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trainer import Trainer
import utils


DEFAULT_MODEL_PATH = '/cluster/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf'


def parse_args():
	parser = argparse.ArgumentParser(
		description='Minimal llama.cpp-backed text inference workload',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument('--model', type=str, default=os.getenv('LLAMACPP_MODEL', DEFAULT_MODEL_PATH), help='Path to GGUF model inside container')
	parser.add_argument('--max-new-tokens', dest='max_new_tokens', type=int, default=128, help='Tokens to generate per prompt')
	parser.add_argument('--max_new_tokens', dest='max_new_tokens', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--prompt-batch-size', dest='prompt_batch_size', type=int, default=1, help='Prompts per TGS iteration')
	parser.add_argument('--prompt_batch_size', dest='prompt_batch_size', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--iterations', type=int, default=100, help='Number of TGS iterations to report')
	parser.add_argument('--prompts', dest='iterations', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--prompt', type=str, default=None, help='Single prompt to replay')
	parser.add_argument('--prompts-file', type=str, default=None, help='One prompt per line')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Force CPU mode')
	parser.add_argument('--api-port', type=int, default=int(os.getenv('LLAMACPP_API_PORT', '8080')))
	parser.add_argument('--ctx-size', type=int, default=int(os.getenv('LLAMACPP_CTX_SIZE', '8192')))
	parser.add_argument('--parallel', type=int, default=int(os.getenv('LLAMACPP_PARALLEL', '4')))
	parser.add_argument('--n-gpu-layers', dest='n_gpu_layers', type=int, default=int(os.getenv('LLAMACPP_N_GPU_LAYERS', '99')))
	parser.add_argument('--server-bin', type=str, default=os.getenv('LLAMACPP_SERVER_BIN', ''), help='Optional path to llama-server binary')
	parser.add_argument('--server-start-timeout', type=int, default=180, help='Seconds to wait for llama-server readiness')
	parser.add_argument('--request-timeout', type=int, default=300, help='HTTP timeout per completion request')
	parser.add_argument('--scheduler_ip', type=str, required=True)
	parser.add_argument('--scheduler_port', type=int, default=6889)
	parser.add_argument('--trainer_port', type=int)
	parser.add_argument('--job_id', type=int, default=-1)
	return parser.parse_args()


def load_prompts(args):
	if args.prompts_file is not None and os.path.exists(args.prompts_file):
		with open(args.prompts_file, 'r', encoding='utf-8') as handle:
			prompts = [line.strip() for line in handle.readlines() if line.strip()]
		if prompts:
			return prompts
	if args.prompt is not None:
		return [args.prompt] * max(1, args.iterations)
	return [
		'Hello, how are you?',
		'Tell me a short story about a robot.',
		'Summarize this: machine learning enables computers to learn patterns from data.',
		'Translate to French: The quick brown fox jumps over the lazy dog.',
		'Write a one-line poem about coffee.',
	]


def resolve_server_bin(explicit_server_bin):
	if explicit_server_bin and os.path.isfile(explicit_server_bin) and os.access(explicit_server_bin, os.X_OK):
		return explicit_server_bin

	for candidate in (
		'/usr/local/bin/llama-server',
		'/opt/llama.cpp/build/bin/llama-server',
	):
		if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
			return candidate

	raise FileNotFoundError(
		'Unable to locate llama-server binary. '
		'Set --server-bin or LLAMACPP_SERVER_BIN, or ensure /usr/local/bin/llama-server exists in the image.'
	)


def wait_until_ready(api_port, timeout_seconds):
	start = time.time()
	endpoints = [
		f'http://127.0.0.1:{api_port}/health',
		f'http://127.0.0.1:{api_port}/v1/models',
	]

	while time.time() - start < timeout_seconds:
		for endpoint in endpoints:
			try:
				response = requests.get(endpoint, timeout=2)
				if response.status_code == 200:
					return
			except requests.RequestException:
				pass
		time.sleep(1)

	raise TimeoutError(f'llama-server did not become ready on port {api_port} within {timeout_seconds} seconds')


def launch_server(args, use_cuda):
	server_bin = resolve_server_bin(args.server_bin)
	model_path = args.model
	if not os.path.exists(model_path):
		raise FileNotFoundError(
			f'Model file not found: {model_path}. Mount/download your GGUF model and pass --model or LLAMACPP_MODEL.'
		)

	ngl = args.n_gpu_layers if use_cuda else 0
	cmd = [
		server_bin,
		'--port', str(args.api_port),
		'-m', model_path,
		'-ngl', str(ngl),
		'--parallel', str(max(1, args.parallel)),
		'-c', str(max(512, args.ctx_size)),
	]
	proc = subprocess.Popen(
		cmd,
		stdout=None,
		stderr=None,
		preexec_fn=os.setsid,
	)
	wait_until_ready(args.api_port, args.server_start_timeout)
	return proc


def cleanup_server(proc):
	if proc is None:
		return
	if proc.poll() is not None:
		return
	try:
		os.killpg(proc.pid, signal.SIGTERM)
		proc.wait(timeout=10)
	except Exception:
		try:
			os.killpg(proc.pid, signal.SIGKILL)
		except Exception:
			pass


def run_request(api_url, model, prompt, max_new_tokens, timeout):
	payload = {
		'model': model,
		'prompt': prompt,
		'max_tokens': max_new_tokens,
		'temperature': 0.0,
		'top_p': 0.9,
		'stream': False,
	}
	response = requests.post(api_url, json=payload, timeout=timeout)
	response.raise_for_status()


if __name__ == '__main__':
	args = parse_args()

	try:
		import torch
		use_cuda = (not args.no_cuda) and torch.cuda.is_available()
		print(f'CUDA available: {torch.cuda.is_available()}, using CUDA: {use_cuda}')
	except Exception:
		print('PyTorch not available, falling back to CPU mode')
		use_cuda = False

	trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, args.job_id, 1)
	prompts = load_prompts(args)

	server_proc = None
	try:
		server_proc = launch_server(args, use_cuda)
		api_url = f'http://127.0.0.1:{args.api_port}/v1/completions'

		completed_iterations = 0
		prompt_idx = 0
		while completed_iterations < args.iterations:
			batch_prompts = []
			for _ in range(max(1, args.prompt_batch_size)):
				batch_prompts.append(prompts[prompt_idx])
				prompt_idx = (prompt_idx + 1) % len(prompts)

			start = time.time()
			for prompt in batch_prompts:
				try:
					run_request(api_url, args.model, prompt, args.max_new_tokens, args.request_timeout)
				except requests.RequestException as exc:
					raise RuntimeError(f'llama.cpp request failed: {exc}') from exc

			trainer.record(time.time() - start)
			completed_iterations += 1
	finally:
		cleanup_server(server_proc)
		try:
			trainer.close()
		except Exception:
			pass