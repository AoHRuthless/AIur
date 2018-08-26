import sc2
from sc2 import Race
from sc2.constants import *

class ExpansionProtocol(object):
    def __init__(self, bot):
        self.bot = bot

    async def expand(self, amount):
        if self.bot.townhalls.amount >= amount:
            return

        if self.bot.can_afford(self.townhall_unit):
            await self.bot.expand_now()

    @property
    def townhall_unit(self):
        races = {
            Race.Terran: COMMANDCENTER,
            Race.Zerg: HATCHERY,
            Race.Protoss: NEXUS
        }
        return races[self.bot.race]