import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import COMMANDCENTER, SCV, SUPPLYDEPOT, REFINERY, \
 BARRACKS, BARRACKSTECHLAB, MARINE

class Spark(sc2.BotAI):
    async def on_step(self, iteration):
        """
        Ticking function called every iteration

        :param <iteration>: The iteration of the step
        """

        await self.distribute_workers()
        await self.train_workers()
        await self.construct_supply_depots()
        await self.construct_refineries()
        await self.expand()
        await self.construct_barracks()
        await self.train_military()

    async def train_workers(self):
        """
        Checks each available command center and trains an SCV 
        asynchonrously.
        """

        for base in self.ready_bases.noqueue:
            if self.can_afford(SCV):
                await self.do(base.train(SCV))

    async def construct_supply_depots(self):
        """
        Constructs supply depots automatically if supply is running low
        """

        if self.supply_left() >= 3 or self.already_pending():
            return

        if self.ready_bases.exists and self.can_afford(SUPPLYDEPOT):
            await self.build(SUPPLYDEPOT, near=self.ready_bases.first)

    async def construct_refineries(self):
        """
        Constructs available refineries near constructed command centers
        """
        if not self.can_afford(REFINERY):
             return

        for base in self.ready_bases:
            vespenes = self.state.vespene_geyser.closer_than(15.0, 
                base)
            for vespene in vespenes:
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break

                if not self.units(REFINERY.closer_than(1.0, vespene)).exists:
                    await self.do(worker.build(REFINERY, vespene))

    async def expand(self):
        """
        Expands now if affordable. Currently, expansion is arbitrarily 
        limited to 3 command centers.
        """

        if self.num_bases >= 3:
            return

        if self.can_afford(COMMANDCENTER):
            self.expand_now()

    async def construct_barracks(self):
        """
        Builds up to two barracks and selects one to build a tech lab add-on.
        """

        if self.units(BARRACKS).ready.exists and self.units(BARRACKS).
        amount < 2:
            await self.upgrade_barracks()
        elif self.can_afford(BARRACKS) and not self.already_pending(BARRACKS):
            await self.build(BARRACKS, near=self.ready_bases.first)

    async def upgrade_barracks(self):
        """
        Upgrades a barrack by adding a tech-lab. Reactors not supported yet.
        """

        for barrack in self.units(BARRACKS).ready:
            if not self.units(BARRACKSTECHLAB):
                if self.can_afford(BARRACKSTECHLAB) and not self.
                already_pending(BARRACKSTECHLAB):
                    await self.build(BARRACKSTECHLAB, near=barrack)
            else:
                break

    async def train_military(self):
        """
        Trains our army.

        Hoorah for marines!
        """

        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARINE) and self.supply_left > 0:
                await self.do(barrack.train(MARINE))

    @property
    def num_workers(self):
        """
        Grabs the number of scvs.
        """
        return self.units(SCV).amount
    
    @property
    def num_bases(self):
        """
        Grabs the number of command centers.
        """
        return self.bases.amount

    @property
    def ready_bases(self):
        """
        Grabs the number of ready command centers.
        """
        return self.bases.ready

    @property
    def bases(self):
        """
        Grabs all command center units.
        """
        return self.units(COMMANDCENTER)
    


run_game(maps.get('(2)RedshiftLE'), 
    [Bot(Race.Terran, Spark()), Computer(Race.Protoss, Difficulty.Easy)], 
    realtime=True)