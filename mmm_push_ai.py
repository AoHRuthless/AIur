from protocols import gas_protocol, expansion_protocol

from abstract_bot import AbstractBot

import sc2
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer, Human
from sc2.constants import *

import cv2 as cv
import numpy as np

class MMMBot(AbstractBot):
    def __init__(self):
        super().__init__()
        self.gas_handler = gas_protocol.GasProtocol(self)
        self.expansion_handler = expansion_protocol.ExpansionProtocol(self)

    async def on_step(self, iteration):
        await super().on_step(iteration)

        await self.collect_intel()

        if 5 > self.townhalls.amount < self.minutes_elapsed / 2.9:
            await self.expansion_handler.expand(
                max(3, 2 + self.minutes_elapsed / 4))
            return

        military = {
            MARINE: 7,
            MARAUDER: 3#,
            #MEDIVAC: 2
        }

        self.prepare_attack(military)
        await self.manage_workers()
        await self.manage_gas()
        await self.manage_supply()
        await self.handle_military()

    async def collect_intel(self):
        map_size = self.game_info.map_size
        width, height = map_size[1], map_size[0]
        data = np.zeros((width, height, 3), np.uint8)

        for base in self.units(COMMANDCENTER):
            cc = base.position
            cv.circle(data, (int(cc[0]), int(cc[1])), 10, (0, 255, 0), -1)

        flipped = cv.flip(data, 0)
        resized = cv.resize(flipped, dsize=None, fx=2, fy=2)
        cv.imshow('INTELLIGENCE', resized)
        cv.waitKey(1)

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

    async def manage_gas(self):
        """
        Logic to have workers build refineries then assign workers as needed.
        """

        if self.units(BARRACKS).exists:
            await self.gas_handler.manage_gas(1.5)

    async def manage_supply(self):
        """
        Manages supply limits. Supply depots are lowered when built.
        """

        if self.supply_cap >= 200 and self.units.of_type([SUPPLYDEPOT, 
            SUPPLYDEPOTLOWERED, 
            SUPPLYDEPOTDROP
            ]).ready.exists:
                return

        if self.supply_left < 4 and self.can_afford(SUPPLYDEPOT) \
        and self.already_pending(SUPPLYDEPOT) < 2:
            position = self.townhalls.random.position.towards(
                self.game_info.map_center, 5)
            await self.build(SUPPLYDEPOT, position)

        for depot in self.units(SUPPLYDEPOT).ready:
            await self.do(depot(MORPH_SUPPLYDEPOT_LOWER))

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

        if self.units(FACTORY).amount < 1:
            if self.can_afford(FACTORY):
                await self.build(FACTORY, 
                    near=self.townhalls.random.position, 
                    placement_step=7)

        if self.units(FACTORY).ready.amount < 1:
            return

        # if self.units(STARPORT).amount < max(2, self.minutes_elapsed / 2 - 5):
        #     if self.can_afford(STARPORT):
        #         await self.build(STARPORT, 
        #             near=self.townhalls.random.position, 
        #             placement_step=7)

    async def manage_military_add_ons(self):
        for rax_lab in self.units(BARRACKSTECHLAB).ready.noqueue:
            abilities = await self.get_available_abilities(rax_lab)
            for ability in abilities:
                if self.can_afford(ability):
                    await self.do(rax_lab(ability))

    async def manage_military_research_structures(self):
        pass

    async def train_military(self):
        """
        Trains our bio-terran MMM army.
        """

        for rax in self.units(BARRACKS).ready.noqueue:
            if self.can_afford(MARAUDER) and rax.has_add_on:
                await self.do(rax.train(MARAUDER))
            elif self.can_afford(MARINE):
                await self.do(rax.train(MARINE))
        # for sp in self.units(STARPORT).ready.noqueue:
        #     if not self.can_afford(MEDIVAC):
        #         break
        #     await self.do(sp.train(MEDIVAC))
    

# RUN GAME
sc2.run_game(sc2.maps.get('(2)RedshiftLE'), [
    Bot(Race.Terran, MMMBot()), 
    Computer(Race.Protoss, Difficulty.Hard)
    ], realtime=False)