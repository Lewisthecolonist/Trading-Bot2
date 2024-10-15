import json
from datetime import datetime, timedelta

class APICallManager:
    def __init__(self, daily_limit=49, state_file='api_call_state.json'):
        self.daily_limit = daily_limit
        self.state_file = state_file
        self.load_state()

    def save_state(self):
        state = {
            'calls_made': self.calls_made,
            'reset_time': self.reset_time.isoformat() if self.reset_time else None
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)

    def load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.calls_made = state['calls_made']
            self.reset_time = datetime.fromisoformat(state['reset_time']) if state['reset_time'] else None
        except (FileNotFoundError, json.JSONDecodeError):
            self.calls_made = 0
            self.reset_time = None

    async def record_call(self):
        self.calls_made += 1
        self.save_state()
