import argparse

# from sympy import false
import utils
import threading
import time
import subprocess
import os
import pynvml
import csv
import queue

from runtime.rpc import scheduler_server
from task import Task, JobInfo


class Worker(object):
    def __init__(self, trace_file_path: str, worker_ip, worker_port, gpus: str, mount: list, log_path: str, need_throughput) -> None:
        super().__init__()

        self._logger = utils.make_logger(__name__)
        self._writer = utils.Writer(log_path)

        self.parse_trace_config(trace_file_path)
        
        self._worker_ip = worker_ip
        self._worker_port = worker_port
        self._worker_id = None
        self.need_throughput = need_throughput
        
        self._gpus = gpus.split(',')
        self._num_gpus = len(self._gpus)

        self._mount = mount if mount != None else []

        self.tgs_init()
        
        self._tasks = dict()

        self._server_for_trainer = self.make_server_for_trainer(worker_port)

        self._start_time = time.time()

        self._latest_reports = {}
        self._slo_enabled = os.getenv('TGS_SLO_MODE', '0') == '1'
        self._slo_ratio = float(os.getenv('TGS_SLO_RATIO', '1.25'))
        self._slo_sleep_seconds = float(os.getenv('TGS_SLO_SLEEP_MS', '120')) / 1000.0
        self._slo_stale_seconds = float(os.getenv('TGS_SLO_STALE_SEC', '6'))

        # Plain-text metrics (comma-separated): one row per ReportStats RPC, independent of Python logging.
        # Set TGS_SLO_METRICS_PATH to enable. Optional TGS_REPORT_INTERVAL_SEC on the trainer lowers interval.
        self._slo_metrics_path = os.getenv('TGS_SLO_METRICS_PATH')
        self._slo_metrics_lock = threading.Lock()
        self._slo_metrics_fp = None
        if self._slo_metrics_path:
            new_file = not os.path.exists(self._slo_metrics_path) or os.path.getsize(self._slo_metrics_path) == 0
            self._slo_metrics_fp = open(self._slo_metrics_path, 'a', buffering=1, encoding='utf-8')
            if new_file:
                self._slo_metrics_fp.write(
                    '# report: kind,unix_ts,job_id,throughput,finished_iterations,ttft_ms,tpot_ms\n'
                    '# slo: kind,unix_ts,leader_job,lagger_job,leader_tp,lagger_tp,ratio,action\n'
                )
    

    def parse_trace_config(self, trace_file_path):
        assert trace_file_path[-4:] == '.csv'
        trace_file = open(trace_file_path, 'r')

        reader = csv.DictReader(trace_file, delimiter=',', skipinitialspace=True)

        self._submit_queue = list()
        self.next_job_id = 1
        for row in reader:
            self.parse_job(row)
        
        trace_file.close()
        self._submit_queue = sorted(self._submit_queue, key=lambda x: (x['submit_time'], 0 if x['priority'] == 'high' else 1))


    def parse_job(self, job_spec):
        assert 'submit_time' in job_spec
        assert 'model_name' in job_spec
        assert 'batch_size' in job_spec
        assert 'iterations' in job_spec
        assert 'gpu_requests' in job_spec
        assert 'priority' in job_spec

        # if job_spec['model_name'] == 'shufflenet':
        #     job_spec['model_name'] = 'shufflenet_v2_x1_0'

        spec = {
            'submit_time': float(job_spec['submit_time']),
            'job_id': self.next_job_id,
            'model_name': job_spec['model_name'],
            'batch_size': job_spec['batch_size'],
            'iterations': int(job_spec['iterations']),
            'num_gpus': int(job_spec['gpu_requests']),
            'priority': job_spec['priority'],
            'thread_percentage': job_spec['thread_percentage'] if 'thread_percentage' in job_spec else None,
            'image_name': job_spec['image_name'] if 'image_name' in job_spec else 'tf_torch',
            'antman_config': job_spec['antman_config'] if 'antman_config' in job_spec else None,
            'antman_status': job_spec['antman_status'] if 'antman_status' in job_spec else None,
        }
        
        self._submit_queue.append(spec)
        self.next_job_id += 1


    def tgs_init(self):
        assert subprocess.call(['./hijack/build.sh']) == 0
        root_path = os.path.abspath('.')

        self.tgs_mounts = {
            'high': [
                root_path + ':/cluster',
                root_path + '/hijack/high-priority-lib/libcontroller.so:/libcontroller.so:ro',
                root_path + '/hijack/high-priority-lib/libcuda.so:/libcuda.so:ro',
                root_path + '/hijack/high-priority-lib/libcuda.so.1:/libcuda.so.1:ro',
                root_path + '/hijack/high-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro',
                root_path + '/hijack/high-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro',
                root_path + '/hijack/high-priority-lib/ld.so.preload:/etc/ld.so.preload:ro',
                root_path + '/gsharing:/etc/gsharing',
            ],
            'low': [
                root_path + ':/cluster',
                root_path + '/hijack/low-priority-lib/libcontroller.so:/libcontroller.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so:/libcuda.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so.1:/libcuda.so.1:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro',
                root_path + '/hijack/low-priority-lib/ld.so.preload:/etc/ld.so.preload:ro',
                root_path + '/gsharing:/etc/gsharing',
            ],
            'Ex': [
                root_path + ':/cluster',
            ],
            'Co-ex': [
                root_path + ':/cluster',
            ],
            'mig-high': [
                root_path + ':/cluster',
            ],
            'mig-low': [
                root_path + ':/cluster',
            ],
            'mps': [
                root_path + ':/cluster',
                '/tmp/nvidia-mps:/tmp/nvidia-mps',
            ],
            'shared': [
                root_path + ':/cluster',
                root_path + '/hijack/low-priority-lib/libcontroller.so:/libcontroller.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so:/libcuda.so:ro',
                root_path + '/hijack/low-priority-lib/libcuda.so.1:/libcuda.so.1:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so:/libnvidia-ml.so:ro',
                root_path + '/hijack/low-priority-lib/libnvidia-ml.so.1:/libnvidia-ml.so.1:ro',
                root_path + '/hijack/low-priority-lib/ld.so.preload:/etc/ld.so.preload:ro',
                root_path + '/gsharing:/etc/gsharing',
            ],
        }


    def check_tasks(self):
        finished_tasks = []

        for job_id, task in self._tasks.items():
            if task.return_code != 0 and task.return_code is not None:
                print(f"Task {task._job_id} failed with exit code {task.return_code}")
            if task.return_code == None:
                continue
            if task.return_code == 0:
                if task._finished_iterations != task._iterations:
                    print(
                        f"Task {task._job_id} finished cleanly but only reported "
                        f"{task._finished_iterations}/{task._iterations} iterations"
                    )
                    task._finished_iterations = task._iterations

            finished_tasks.append(task)

        if len(finished_tasks) > 0:
            self.record()
        for task in finished_tasks:
            self._tasks.pop(task._job_id)

        return finished_tasks
    

    def execute(self, job_info) -> bool:
        success = True

        task = Task(job_info, self._worker_ip, self.tgs_mounts, self.need_throughput)
        self._tasks[task._job_id] = task
        cmd = task.run(self._mount)

        self._logger.info(f'{self._worker_id}, execute, {task._job_id}, {task._gpus}, {task._priority}, {" ".join(cmd)}')

        return success
    

    def kill(self, job_info) -> bool:
        job_id = job_info.job_id

        if job_id not in self._tasks:
            return False

        task = self._tasks.pop(job_id)
        task.terminate()

        self._logger.info(f'{self._worker_id}, kill, {job_id}, {job_info.gpus}, {job_info.priority}')

        return True
    

    def query_node_stats(self):
        utilizations = []
        pynvml.nvmlInit()
        for gpu_id in range(self._num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            utilizations.append(str(utilization))
        pynvml.nvmlShutdown()

        self._logger.info(f'{self._worker_id}, query, {"-".join(utilizations)}')
        utilizations = ','.join(utilizations)
        return utilizations


    def _append_slo_metrics_file(
        self,
        kind: str,
        ts: float,
        fields,
    ):
        """fields is a tuple/stringable sequence written as CSV continuation after fixed prefix."""
        if self._slo_metrics_fp is None:
            return
        rest = ','.join(str(x) for x in fields)
        line = f'{kind},{ts:.6f},{rest}\n'
        with self._slo_metrics_lock:
            self._slo_metrics_fp.write(line)

    def _report_stats_impl(self, job_id, finished_iterations, ttft_ms=0.0, tpot_ms=0.0) -> bool:
        success = True
        assert job_id in self._tasks
        task = self._tasks[job_id]
        throughput = task.update(finished_iterations)
        now = time.time()
        self._latest_reports[job_id] = {
            'throughput': throughput,
            'priority': task._priority,
            'timestamp': now,
        }

        self._logger.info(f'worker, report, {job_id}, {throughput}, {task._finished_iterations}')
        if ttft_ms or tpot_ms:
            self._logger.info(f'worker, slo_metrics, {job_id}, ttft_ms={ttft_ms:.3f}, tpot_ms={tpot_ms:.3f}')

        self._append_slo_metrics_file(
            'report',
            now,
            (job_id, f'{throughput:.6f}', task._finished_iterations, f'{ttft_ms:.6f}', f'{tpot_ms:.6f}'),
        )

        if self._slo_enabled:
            shared_reports = []
            for report_job_id, report in self._latest_reports.items():
                if report_job_id not in self._tasks:
                    continue
                if report['priority'] != 'shared':
                    continue
                if now - report['timestamp'] > self._slo_stale_seconds:
                    continue
                shared_reports.append((report_job_id, report['throughput']))

            if len(shared_reports) >= 2:
                leader_job, leader_tp = max(shared_reports, key=lambda item: item[1])
                lagger_job, lagger_tp = min(shared_reports, key=lambda item: item[1])
                ratio = leader_tp / max(lagger_tp, 1e-6)
                action = 'none'
                if job_id == leader_job and ratio >= self._slo_ratio:
                    action = 'sleep'
                    time.sleep(self._slo_sleep_seconds)
                self._logger.info(
                    f'worker, slo, leader={leader_job}, lagger={lagger_job}, '
                    f'leader_tp={leader_tp:.3f}, lagger_tp={lagger_tp:.3f}, '
                    f'ratio={ratio:.3f}, action={action}'
                )
                self._append_slo_metrics_file(
                    'slo',
                    now,
                    (
                        leader_job,
                        lagger_job,
                        f'{leader_tp:.6f}',
                        f'{lagger_tp:.6f}',
                        f'{ratio:.6f}',
                        action,
                    ),
                )
        return success


    def make_server_for_trainer(self, port):
        callbacks = {
            'ReportStats' : self._report_stats_impl,
        }

        return scheduler_server.serve(port, self._logger, callbacks)


    def has_ready_jobs(self):
        current_time = time.time()
        elapsed_time = current_time - self._start_time

        if len(self._submit_queue) > 0:
            job_spec = self._submit_queue[0]
            if job_spec['submit_time'] <= elapsed_time:
                return True
        
        return False


    def record(self):
        timestamp = time.time() - self._start_time
        for task in self._tasks.values():
            task.record(timestamp, self._writer)


    def close(self):
        if self._slo_metrics_fp is not None:
            with self._slo_metrics_lock:
                self._slo_metrics_fp.close()
                self._slo_metrics_fp = None
        self._writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_port', type=int, default=6889)
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--mount', action='append')
    parser.add_argument('--trace', type=str,  required=True) # default='config/test_tgs.csv')
    parser.add_argument('--log_path', type=str,  required=True) # default='results/test_tgs_results.csv')
    parser.add_argument('--need_throughput', action='store_true', default=False)
    args = parser.parse_args()

    subprocess.call('docker stop $(docker ps -q)', shell=True)
    subprocess.call('docker rm $(docker ps -aq)', shell=True)

    worker_ip = utils.get_host_ip()
    worker = Worker(args.trace, worker_ip, args.worker_port, args.gpus, args.mount, args.log_path, args.need_throughput)

    runnable_tasks = list()
    gpu_list = args.gpus.split(',')
    machine = [{
        'Co-ex': list(),
        'mps': list()
    } for i in range(len(gpu_list))]
    while len(worker._submit_queue) + len(worker._tasks) + len(runnable_tasks) > 0:
        while worker.has_ready_jobs():
            job_spec = worker._submit_queue.pop(0)
            jobinfo = JobInfo(job_spec['job_id'], job_spec['model_name'], job_spec['batch_size'],
                 job_spec['iterations'], job_spec['num_gpus'], job_spec['priority'],
                 job_spec['thread_percentage'], job_spec['image_name'],
                 job_spec['antman_config'], job_spec['antman_status']
                )
            runnable_tasks.append(jobinfo)

        finished_tasks = worker.check_tasks()
        for task in finished_tasks:
            for gpu_id in task._gpus.split(','):
                if task._priority in ['Co-ex', 'mps']:
                    machine[int(gpu_id)][task._priority].remove(task._job_id)
                else:
                    machine[int(gpu_id)].pop(task._priority)
            # writer.save(task)
        
        new_runnable_tasks = []
        record_flag = (len(finished_tasks) != 0)
        for jobinfo in runnable_tasks:
            available_gpus = 0
            for gpu_instance in machine:
                if jobinfo.priority not in gpu_instance:
                    available_gpus += 1
                elif jobinfo.priority in ['Co-ex', 'mps'] and len(gpu_instance[jobinfo.priority]) < 2:
                    available_gpus += 1
            
            if available_gpus >= jobinfo.num_gpus:
                record_flag = True
                used_gpus = []
                for gpu_id, gpu_instance in enumerate(machine):
                    if jobinfo.priority not in gpu_instance:
                        used_gpus.append(str(gpu_id))
                        gpu_instance[jobinfo.priority] = jobinfo.job_id
                    elif jobinfo.priority in ['Co-ex', 'mps'] and len(gpu_instance[jobinfo.priority]) < 2:
                        used_gpus.append(str(gpu_id))
                        gpu_instance[jobinfo.priority].append(jobinfo.job_id)
                    
                    if len(used_gpus) == jobinfo.num_gpus:
                        break
                jobinfo.gpus = ','.join(used_gpus)
                worker.execute(jobinfo)
            else:
                new_runnable_tasks.append(jobinfo)

        if record_flag:
            worker.record()
        runnable_tasks = new_runnable_tasks

        sleep_time = 2
        if len(worker._submit_queue) > 0:
            sleep_time = min(sleep_time, (worker._start_time + worker._submit_queue[0]['submit_time'] - time.time()))
        time.sleep(sleep_time)
    
    worker.close()