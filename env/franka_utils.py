import time


class ControlFreqGuard:
    def __init__(self, control_hz, slack_time=0.001):
        self.control_hz = control_hz
        self.slack_time = slack_time

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t_curr = time.time()
        t_end = self.t_start + 1 / self.control_hz
        t_wait = t_end - t_curr
        if t_wait > 0:
            t_sleep = t_wait - self.slack_time
            if t_sleep > 0:
                time.sleep(t_sleep)
            while time.time() < t_end:
                pass
