import time
from datetime import datetime, timedelta

class APICallManager:
    def __init__(self, daily_limit=49):
        self.daily_limit = daily_limit
        self.calls_made = 0
        self.reset_time = None

    def can_make_call(self):
        current_time = datetime.now()
        if self.reset_time is None or current_time >= self.reset_time:
            self.calls_made = 0
            self.reset_time = current_time + timedelta(days=1)
        
        return self.calls_made < self.daily_limit

    def record_call(self):
        self.calls_made += 1

    def time_until_reset(self):
        if self.reset_time is None:
            return 0
        current_time = datetime.now()
        return max(0, (self.reset_time - current_time).total_seconds())
