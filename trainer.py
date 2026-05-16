from runtime.rpc import trainer_client

import argparse
import utils
import time
import os

REPORT_INTERVAL = float(os.getenv('TGS_REPORT_INTERVAL_SEC', os.getenv('REPORT_INTERVAL', '2.0')))


class Trainer(object):
    def __init__(self, worker_ip, worker_port, trainer_ip, trainer_port, job_id, batch_size) -> None:
        super().__init__()

        self._trainer_ip = trainer_ip
        self._trainer_port = trainer_port
        self._job_id = job_id
        self._batch_size = batch_size

        self._logger = utils.make_logger(__name__)
        self._start_time = time.time()
        self._finished_iteraions = 0

        self._client_for_scheduler = trainer_client.TrainerClientForScheduler(self._logger, worker_ip, worker_port)
        self.init_stats()

        self._logger.info(f'job {self._job_id}, trainer, start, {self._start_time}')
    

    def init_stats(self):
        self._last_report_time = time.time()
        self._report_interval = REPORT_INTERVAL
        LOG_FILE_PATH = os.getenv('TGS_LOG_FILE_PATH')
        self._fd = open(LOG_FILE_PATH, 'w') if LOG_FILE_PATH != None else None
        self._first_iteration_time = None
        self._last_iteration_time = None
    

    def update_stats(self, iteration_time):
        self._finished_iteraions += 1
        if self._first_iteration_time is None:
            self._first_iteration_time = iteration_time
        self._last_iteration_time = iteration_time
        if self._fd != None:
            print('%lf %lf' % (time.time(), self._batch_size / iteration_time), file=self._fd)


    def record(self, iteration_time):
        self.update_stats(iteration_time)

        if time.time() - self._last_report_time >= self._report_interval:
            self._last_report_time = time.time()
            # For training workloads, these are proxies; inference workloads can
            # override via their own trainer wrapper later.
            ttft_ms = None
            tpot_ms = None
            if self._first_iteration_time is not None:
                ttft_ms = self._first_iteration_time * 1000.0
            if self._last_iteration_time is not None:
                tpot_ms = self._last_iteration_time * 1000.0
            if self._client_for_scheduler.report_stats(
                self._job_id,
                self._finished_iteraions,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
            ):
                self._finished_iteraions = 0


    def close(self):
        for _ in range(5):
            if self._client_for_scheduler.report_stats(self._job_id, self._finished_iteraions):
                self._finished_iteraions = 0
                return
            time.sleep(0.2)


if __name__  == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_ip', type=str, required=True)
    parser.add_argument('--worker_port', type=int, default=6889)
    parser.add_argument('--trainer_port', type=int)
    parser.add_argument('--job_id', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    trainer = Trainer(args.worker_ip, args.worker_port, utils.get_host_ip(), args.trainer_port, args.job_id, args.batch_size)