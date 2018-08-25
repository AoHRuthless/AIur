import sc2
from sc2 import Race
from sc2.constants import *

class ExpansionProtocol(object):
    def __init__(self, sc2):
        self.sc2 = sc2

    async def expand(self, amount):
        if self.sc2.townhalls.amount >= amount:
            return

        if self.sc2.can_afford(self.townhall_unit):
            await self.sc2.expand_now()

    @property
    def townhall_unit(self):
        races = {
            Race.Terran: COMMANDCENTER,
            Race.Zerg: HATCHERY,
            Race.Protoss: NEXUS
        }
        return races[self.sc2.race]