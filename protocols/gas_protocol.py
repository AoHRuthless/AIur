import sc2
from sc2 import race_gas

class GasProtocol(object):
    def __init__(self, bot):
        self.bot = bot

    async def manage_gas(self, base_vg_ratio = 2):
        if self.bot.geysers.amount < self.bot.townhalls.amount * base_vg_ratio:
            for cc in self.bot.townhalls:
                for vg in self.bot.state.vespene_geyser.closer_than(15, cc):
                    if self.bot.geysers.closer_than(1, vg).exists:
                        break

                    wrkr = self.bot.select_build_worker(vg.position)
                    if wrkr is None or not self.bot.can_afford(self.gas_unit):
                        break

                    await self.bot.do(wrkr.build(self.gas_unit, vg))
                    break
    @property
    def gas_unit(self):
        return race_gas[self.bot.race]