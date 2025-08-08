import asyncio

class State:
    def __init__(self):
        self.messages = asyncio.Queue()
        self.active = True

    async def update(self, new_message):
        await self.messages.put(new_message)
        await asyncio.sleep(0.01) 

    def close(self):
        self.active = False
    