import json
from datetime import datetime, timedelta
import time
from threading import Thread
import asyncio
class APICallManager:
    def __init__(self, daily_limit=100000000, state_file='api_call_state.json'):
        self.daily_limit = daily_limit
        self.state_file = state_file
        self.calls_made = 0
        self.reset_time = None
    async def initialize(self):
        await self.load_state()

    async def can_make_call(self):
        current_time = datetime.now()
        if self.reset_time is None or current_time >= self.reset_time:
            self.calls_made = 0
            self.reset_time = current_time + timedelta(days=1)
        return self.calls_made < self.daily_limit

    async def record_call(self):
        self.calls_made += 1
        await self.save_state()

    async def time_until_reset(self):
        if self.reset_time is None:
            return 0
        current_time = datetime.now()
        return max(0, (self.reset_time - current_time).total_seconds())

    async def save_state(self):
        state = {
            'calls_made': self.calls_made,
            'reset_time': self.reset_time.isoformat() if self.reset_time else None,
            'stop_time': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    async def load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.calls_made = state['calls_made']
            self.reset_time = datetime.fromisoformat(state['reset_time']) if state['reset_time'] else None
            stop_time = datetime.fromisoformat(state['stop_time'])
            elapsed_time = datetime.now() - stop_time
            
            if self.reset_time and datetime.now() >= self.reset_time:
                self.calls_made = 0
                self.reset_time = datetime.now() + timedelta(days=1)
            elif elapsed_time < timedelta(days=1):
                calls_to_subtract = int(elapsed_time.total_seconds() / (24 * 3600) * self.daily_limit)
                self.calls_made = max(0, self.calls_made - calls_to_subtract)
            await self.save_state()
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            async def stop(self):
                await self.save_state()
                # Add a pass statement to create an indented block
                pass
    def start_api_state_saver(self):
        async def save_state_loop():
            while True:
                await self.save_state()
                await asyncio.sleep(1)  # Save state every 5 seconds

        asyncio.create_task(save_state_loop())

    def record_backtest_call(self):
        if hasattr(self, 'backtest_call_made'):
            raise Exception("Multiple API calls attempted during backtest")
        self.backtest_call_made = True
