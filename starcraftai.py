import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
from sc2.constants import COMMANDCENTER, SCV, SUPPLYDEPOT, REFINERY, BARRACKS

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

    async def train_workers(self):
        """
        Checks each available command center and trains an SCV 
        asynchonrously. We cap SCVs at 15 per command center.

        """
        threshold = self.num_cmd_centers * 15

        for cmd_center in self.ready_cmd_centers.noqueue:
            if self.can_afford(SCV) and self.num_workers() < self.
            num_cmd_centers * 15:
                await self.do(cmd_center.train(SCV))

    async def construct_supply_depots(self):
        """
        Constructs supply depots automatically if supply is running low
        """

        if self.supply_left() >= 3 or self.already_pending():
            return

        if self.ready_cmd_centers.exists and self.can_afford(SUPPLYDEPOT):
            await self.build(SUPPLYDEPOT, near=self.ready_cmd_centers.first)

    async def construct_refineries(self):
        """
        Constructs available refineries near constructed command centers
        """
        if not self.can_afford(REFINERY):
             return

        for cmd_center in self.ready_cmd_centers:
            vespenes = self.state.vespene_geyser.closer_than(15.0, 
                cmd_center)
            for vespene in vespenes:
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break

                if not self.units(REFINERY.closer_than(1.0, vespene)).exists:
                    await self.do(worker.build(REFINERY, vespene))

    async def expand(self):
        if self.num_cmd_centers >= 3:
            return

        if self.can_afford(COMMANDCENTER):
            self.expand_now()

    @property
    def num_workers(self):
        return self.units(SCV).amount()
    
    @property
    def num_cmd_centers(self):
        return self.cmd_centers.amount()

    @property
    def ready_cmd_centers(self):
        return self.cmd_centers.ready

    @property
    def cmd_centers(self):
        return self.units(COMMANDCENTER)
    


run_game(maps.get('(2)RedshiftLE'), 
    [Bot(Race.Terran, Spark()), Computer(Race.Protoss, Difficulty.Easy)], 
    realtime=True)