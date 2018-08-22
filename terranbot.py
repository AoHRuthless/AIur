import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer, Human
from sc2.constants import *
from sc2.ids.ability_id import *

class TerranBot(sc2.BotAI):
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
        await self.upgrade_barracks()
        await self.train_military()
        await self.attack()

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

        if self.supply_cap >= 200:
            return

        if self.supply_left >= 3 or self.already_pending(SUPPLYDEPOT):
            return

        if self.ready_bases.exists and self.can_afford(SUPPLYDEPOT):
            await self.build(SUPPLYDEPOT, near=self.ready_bases.first)

        for depot in self.units(SUPPLYDEPOT).ready:
            await self.do(depot(MORPH_SUPPLYDEPOT_LOWER))

    async def construct_refineries(self):
        """
        Constructs available refineries near constructed command centers
        """

        for base in self.ready_bases:
            vespenes = self.state.vespene_geyser.closer_than(15.0, 
                base)
            for vespene in vespenes:
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break

                if not self.units(REFINERY).closer_than(1.0, vespene).exists \
                and self.can_afford(REFINERY):
                    await self.do(worker.build(REFINERY, vespene))

    async def expand(self):
        """
        Expands now if affordable. Currently, expansion is arbitrarily 
        limited to 3 command centers.
        """

        if self.num_bases >= 3:
            return

        if self.can_afford(COMMANDCENTER):
            await self.expand_now()

    async def construct_barracks(self):
        """
        Builds up to 4 barracks and upgrades as necessary.
        """

        if not self.units.of_type([SUPPLYDEPOT, 
            SUPPLYDEPOTLOWERED, 
            SUPPLYDEPOTDROP]).ready.exists:
            return

        if self.can_afford(BARRACKS) and not self.already_pending(BARRACKS) \
        and self.units(BARRACKS).ready.amount < 4:
            await self.build(BARRACKS, 
                near=self.ready_bases.first.position, 
                placement_step=4)

    async def upgrade_barracks(self):
        """
        Upgrades barracks by building one tech lab add-on and up to two 
        reactors.
        """

        ready_rax = self.units(BARRACKS).ready
        # if ready_rax.amount <= 0:
        #     return

        # first_rax = ready_rax.first
        # if not first_rax.has_add_on and first_rax.noqueue \
        # and self.can_afford(BARRACKSTECHLAB):
        #     await self.do(first_rax.build(BARRACKSTECHLAB))

        for index in range(0, len(ready_rax)):
            unit = BARRACKSTECHLAB if index == 0 else BARRACKSREACTOR

            rax = ready_rax[index]
            if not self.can_afford(unit):
                break

            if not rax.has_add_on and rax.noqueue:
                await self.do(rax.build(unit)) 


    async def train_military(self):
        """
        Trains our army.

        Hoorah for marines!
        """

        for barrack in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARINE) and self.supply_left > 0:
                await self.do(barrack.train(MARINE))

    async def attack(self):
        """
        Determines whether or not to send our marines to seek the enemy or 
        defend the base.
        """

        marines = self.units(MARINE)

        if marines.amount >= 15:
            for marine in marines.idle:
                await self.do(marine.attack(self.seek_target))
        elif marines.amount >= 3:    
            for marine in marines.idle:
                if len(self.known_enemy_units) == 0:
                    break

                enemy_unit = self.known_enemy_units.random
                await(self.do(marine.attack(enemy_unit)))

    @property
    def seek_target(self):
        """
        Seeks out the enemy to attack with the army. Prioritizes known units > 
        known structures > start location
        """

        if len(self.known_enemy_units) > 0:
            return self.known_enemy_units.random
        elif len(self.known_enemy_structures) > 0:
            return self.known_enemy_structures.random
        else:
            return self.enemy_start_locations[0]

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
    [Bot(Race.Terran, TerranBot()), Computer(Race.Protoss, Difficulty.Easy)], 
    realtime=False)