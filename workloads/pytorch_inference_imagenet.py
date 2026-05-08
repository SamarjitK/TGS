import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer import Trainer
import utils
import time
trainer = None

import torch
import argparse
import torch.backends.cudnn as cudnn
from torchvision import models
from tqdm import tqdm

# Inference settings
parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for inference')
parser.add_argument('--iterations', type=int, default=1000,
                    help='number of inference iterations')
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA inference')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--scheduler_ip', type=str, required=True)
parser.add_argument('--scheduler_port', type=int, default=6889)
parser.add_argument('--trainer_port', type=int)
parser.add_argument('--job_id', type=int, default=-1)


def inference():
    """Run inference iterations and measure throughput"""
    model.eval()
    
    last_timestamp = time.time()
    remaining_iterations = args.iterations
    
    with tqdm(total=args.iterations,
              desc='Inference',
              disable=not verbose) as t:
        while remaining_iterations > 0:
            # Generate random input data (ImageNet input size: 3x224x224)
            data = torch.randn(args.batch_size, 3, 224, 224)
            
            if args.cuda:
                data = data.cuda()
            
            # Run inference without gradient computation
            with torch.no_grad():
                output = model(data)
            
            # Synchronize to ensure accurate timing
            if args.cuda:
                torch.cuda.synchronize()
            
            t.update(1)
            remaining_iterations -= 1
            
            # Record timing for throughput calculation
            timestamp = time.time()
            trainer.record(timestamp - last_timestamp)
            last_timestamp = timestamp


if __name__ == '__main__':
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    verbose = 1

    # Load pre-trained model
    model = getattr(models, args.model)(pretrained=True)

    if args.cuda:
        model.cuda()

    # Initialize trainer for TGS integration
    trainer = Trainer(args.scheduler_ip, args.scheduler_port, utils.get_host_ip(), args.trainer_port, args.job_id, args.batch_size)

    # Run inference
    inference()
    
    trainer.close() if hasattr(trainer, 'close') else None
