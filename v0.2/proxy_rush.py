import sc2
from sc2 import BotAI
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.helpers import ControlGroup
from sc2.player import Bot, Computer

ITERATIONS_PER_MINUTE = 165

import cv2 as cv
import numpy as np

class ProxyRaxRushBot(sc2.BotAI):
    async def on_step(self, iteration):
        self.iteration = iteration
        self.attack_waves = set()

        if not self.townhalls.exists:
            for unit in self.workers | self.marines:
                await self.do(unit.attack(self.target))
            return
        self.command_center = self.townhalls.first

        military = {
            MARINE: 15
        }
        self.prepare_attack(military, interval=21)
        await self.manage_workers()
        await self.manage_supply()
        await self.manage_military_training_structures()
        await self.train_military()
        await self.collect_intel()
        await self.attack()
        await self.task_workers()

    async def collect_intel(self):
        # game coordinates need to be represented as (y, x) in 2d arrays
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        rgb = (0, 255, 0)
        for cc in self.townhalls:
            posn = cc.position
            cv.circle(game_data, (int(posn[0]), int(posn[1])), 10, rgb, -1)

        # cv assumes (0, 0) top-left => need to flip along horizontal axis
        flipped = cv.flip(game_data, 0)

        cv.imshow('Map', cv.resize(flipped, dsize=None, fx=2, fy=2))
        cv.waitKey(1)

    async def manage_workers(self):
        if self.can_afford(SCV) and self.workers.amount <= 15 \
        and self.command_center.noqueue:
            await self.do(self.command_center.train(SCV)) 

    async def manage_supply(self):
        supply_threshold = 2 if self.barracks.amount < 3 else 4

        if self.supply_left < supply_threshold and \
        self.can_afford(SUPPLYDEPOT) and self.already_pending(SUPPLYDEPOT) < 2:
            position = self.command_center.position.towards(
                self.game_info.map_center, 5)
            await self.build(SUPPLYDEPOT, position)

    async def manage_military_training_structures(self): 
        if not self.units.of_type([
            SUPPLYDEPOT, 
            SUPPLYDEPOTLOWERED, 
            SUPPLYDEPOTDROP
            ]).ready.exists:
            return   

        if self.barracks.amount < 3 or \
        (self.barracks.amount < 5 and self.minerals > 400):
            if self.can_afford(BARRACKS):
                game_info = self.game_info
                position = game_info.map_center.towards(
                    self.enemy_start_locations[0], 25)
                await self.build(BARRACKS, near=position)

    def prepare_attack(self, military_ratio, interval=10):
        """
        Prepares an attack wave every interval when ready. Attack waves are 
        comprised of military units specified by the given ratio.

        :param military_ratio: <dict> [UnitId: int] specifying military 
        composition.
        :param interval: <int> time interval in seconds to prepare a wave.
        """

        if self.seconds_elapsed % interval != 0:
            return

        attack_wave = None
        for unit in military_ratio:
            amount = military_ratio[unit]
            units = self.units(unit)

            if units.idle.amount >= amount:
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

    async def task_workers(self):
        min_field = self.state.mineral_field.closest_to(self.command_center)
        for scv in self.units(SCV).idle:
            await self.do(scv.gather(min_field))

    async def train_military(self):
        for rax in self.barracks.ready.noqueue:
            if not self.can_afford(MARINE):
                break
            await self.do(rax.train(MARINE))

    @property
    def target(self):
        """
        Seeks a random enemy structure or prioritizes the start location
        """
        return self.known_enemy_structures.random_or(self.
            enemy_start_locations[0]).position

    @property
    def barracks(self):
        return self.units(BARRACKS)
    
    @property
    def marines(self):
        return self.units(MARINE)

    @property
    def minutes_elapsed(self):
        """
        Grabs the minutes currently elapsed.
        """
        return self.iteration / ITERATIONS_PER_MINUTE

    @property
    def seconds_elapsed(self):
        """
        Grabs the seconds currently elapsed.
        """
        return self.iteration / (ITERATIONS_PER_MINUTE / 60)


# Can beat elite protoss and terran AI with ease
# Loses occasionally to elite early zergling/roach push
result = sc2.run_game(sc2.maps.get("(2)RedshiftLE"), [
    Bot(Race.Terran, ProxyRaxRushBot()),
    Computer(Race.Protoss, Difficulty.VeryHard)
    ], realtime=False)

print("----")
print(result)