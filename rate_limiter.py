import asyncio
import time

class RateLimiter:
    def __init__(self, rate, per):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.time()

    async def wait(self):
        current = time.time()
        time_passed = current - self.last_check
        self.last_check = current
        self.allowance += time_passed * (self.rate / self.per)
        if self.allowance > self.rate:
            self.allowance = self.rate
        if self.allowance < 1:
            await asyncio.sleep((1 - self.allowance) * (self.per / self.rate))
            self.allowance = 0
        else:
            self.allowance -= 1
