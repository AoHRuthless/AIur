import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer, Human
from sc2.helpers import ControlGroup
from sc2.constants import *

class MMMBot(sc2.BotAI):
    def __init__(self):
        self.attack_waves = set()
        self.iterations_per_min = 165
        self.max_workers = 65

    async def on_step(self, iteration):
        self.iteration = iteration
        if not self.townhalls.exists:
            for unit in self.units:
                await self.do(unit.attack(self.target))
            return

        await self.distribute_workers()
        if 5 > self.townhalls.amount < self.minutes_elapsed / 2.9:
            await self.expand()
            return

        self.prepare_attack()
        await self.manage_workers()
        await self.manage_supply()
        await self.manage_military_training_structures()
        await self.train_military()
        await self.attack()

    async def manage_workers(self):
        """
        Manages workers and trains SCVs as needed.
        """

        if self.townhalls.amount * 16 <= self.workers.amount \
        or self.max_workers <= self.workers.amount:
            return

        for base in self.units(COMMANDCENTER).ready.noqueue:
            if self.can_afford(SCV):
                await self.do(base.train(SCV))
        await self.manage_gas()

    async def manage_gas(self):
        """
        Logic to have workers build refineries then assign workers as needed.
        """

        if self.units(BARRACKS).exists \
        and self.units(REFINERY).amount < self.townhalls.amount * 1.5:
            if self.can_afford(REFINERY):
                for cc in self.townhalls:
                    for vg in self.state.vespene_geyser.closer_than(10, cc):
                        if self.units(REFINERY).closer_than(1, vg).exists:
                            break

                        worker = self.select_build_worker(vg.position)
                        if worker is None:
                            break

                        await self.do(worker.build(REFINERY, vg))
                        break

    async def manage_supply(self):
        """
        Manages supply limits. Supply depots are lowered when built.
        """

        if self.supply_cap >= 200 and self.units(SUPPLYDEPOT) > 0:
            return

        if self.supply_left < 4 and self.can_afford(SUPPLYDEPOT) \
        and self.already_pending(SUPPLYDEPOT) < 2:
            position = self.townhalls.random.position.towards(
                self.game_info.map_center, 5)
            await self.build(SUPPLYDEPOT, position)

        for depot in self.units(SUPPLYDEPOT).ready:
            await self.do(depot(MORPH_SUPPLYDEPOT_LOWER))

    async def expand(self):
        """
        Constructs a new expansion when possible. # Expansions scale with time.
        """

        if self.townhalls.amount >= max(3, 2 + self.minutes_elapsed / 4):
            return

        if self.can_afford(COMMANDCENTER):
            await self.expand_now()

    async def manage_military_training_structures(self):
        """
        Trains and upgrades military structures. Since this bot is a bio army, 
        we only need one factory. # Barracks and Starports scale with time. 
        Algorithm attempts to place a tech lab onto each barrack, and all 
        upgrades are researched.
        """

        if not self.units.of_type([
            SUPPLYDEPOT, 
            SUPPLYDEPOTLOWERED, 
            SUPPLYDEPOTDROP
            ]).ready.exists:
            return 

        if self.units(BARRACKS).amount < max(4, self.minutes_elapsed / 2):
            if self.can_afford(BARRACKS):
                await self.build(BARRACKS, 
                    near=self.townhalls.random.position, 
                    placement_step=5)

        if self.units(BARRACKS).ready.amount < 1:
            return

        for barrack in self.units(BARRACKS).ready.noqueue:
            if not barrack.has_add_on and self.can_afford(BARRACKSTECHLAB):
                await self.do(barrack.build(BARRACKSTECHLAB))
        for rax_lab in self.units(BARRACKSTECHLAB).ready.noqueue:
            abilities = await self.get_available_abilities(rax_lab)
            for ability in abilities:
                if self.can_afford(ability):
                    await self.do(rax_lab(ability))

        if self.units(FACTORY).amount < 1:
            if self.can_afford(FACTORY):
                await self.build(FACTORY, 
                    near=self.townhalls.random.position, 
                    placement_step=7)

        if self.units(FACTORY).ready.amount < 1:
            return

        if self.units(STARPORT).amount < max(2, self.minutes_elapsed / 2 - 5):
            if self.can_afford(STARPORT):
                await self.build(STARPORT, 
                    near=self.townhalls.random.position, 
                    placement_step=7)

    async def train_military(self):
        """
        Trains our bio-terran MMM army.
        """

        for rax in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARAUDER) and rax.has_add_on:
                await self.do(rax.train(MARAUDER))
            elif self.can_afford(MARINE):
                await self.do(rax.train(MARINE))
        for sp in self.units(STARPORT).ready.noqueue:
            if not self.can_afford(MEDIVAC):
                break
            await self.do(sp.train(MEDIVAC))

    def prepare_attack(self):
        """
        Prepares an attack wave every 10 seconds, when applicable. Each wave must have at least 12 units, and is added to in batches.
        """

        if self.seconds_elapsed % 10 != 0:
            return

        offensive_ratio = {
            MARINE: 7,
            MARAUDER: 3,
            MEDIVAC: 2
        }

        added_non_medic_units = False
        attack_wave = None
        for unit in offensive_ratio:
            if not added_non_medic_units and unit == MEDIVAC:
                continue

            amount = offensive_ratio[unit]
            units = self.units(unit)

            if units.idle.amount >= amount:
                if unit != MEDIVAC:
                    added_non_medic_units = True
                if attack_wave is None:
                    attack_wave = ControlGroup(units.idle)
                else:
                    attack_wave.add_units(units.idle)
        if attack_wave is not None and 0 < len(attack_wave) > 12:
            self.attack_waves.add(attack_wave)

    async def attack(self):
        """
        Sends any attack group out to target. No micro is done on the army 
        dispatch.
        """
        for wave in list(self.attack_waves):
            alive_units = wave.select_units(self.units)
            if alive_units.exists and alive_units.idle.exists:
                for unit in wave.select_units(self.units):
                    await self.do(unit.attack(self.target))
            else:
                self.attack_waves.remove(wave)

    @property
    def target(self):
        """
        Seeks a random enemy structure or prioritizes the start location
        """
        return self.known_enemy_structures.random_or(self.
            enemy_start_locations[0]).position

    @property
    def minutes_elapsed(self):
        """
        Grabs the minutes currently elapsed.
        """
        return self.iteration / self.iterations_per_min

    @property
    def seconds_elapsed(self):
        """
        Grabs the seconds currently elapsed.
        """
        return self.iteration / (self.iterations_per_min / 60)
    

# RUN GAME
sc2.run_game(sc2.maps.get('(2)RedshiftLE'), [
    Bot(Race.Terran, MMMBot()), 
    Computer(Race.Zerg, Difficulty.Medium)
    ], realtime=False)