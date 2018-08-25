import sc2
from sc2 import race_gas

class GasProtocol(object):
    def __init__(self, sc2):
        self.sc2 = sc2

    async def manage_gas(self, base_vg_ratio = 2):
        if self.sc2.geysers.amount < self.sc2.townhalls.amount * base_vg_ratio:
            for cc in self.sc2.townhalls:
                for vg in self.sc2.state.vespene_geyser.closer_than(15, cc):
                    if self.sc2.geysers.closer_than(1, vg).exists:
                        break

                    wrkr = self.sc2.select_build_worker(vg.position)
                    if wrkr is None or not self.sc2.can_afford(self.gas_unit):
                        break

                    await self.sc2.do(wrkr.build(self.gas_unit, vg))
                    break
    @property
    def gas_unit(self):
        return race_gas[self.sc2.race]