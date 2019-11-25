import sc2
from sc2 import BotAI, Race, Difficulty, Result
from sc2.constants import *
from sc2.helpers import ControlGroup
from sc2.player import Bot, Computer

import cv2 as cv
import numpy as np

import random
from time import time

ITERATIONS_PER_MINUTE = 165
NUM_EPISODES = 15 # TODO: increase

class ProxyRaxRushBot(sc2.BotAI):

    def __init__(self):
        # Parameters to randomly train on
        self.num_marines = random.randrange(1, 25)
        self.attack_interval_seconds = random.randrange(8, 40)
        self.attack_wave_threshold = random.randrange(0, 20)
        self.num_barracks = random.randrange(2, 8)

        print('--- Parameters ---')
        print(f'num marines = {self.num_marines}')
        print(f'interval attack = {self.attack_interval_seconds}')
        print(f'threshold attack = {self.attack_wave_threshold}')
        print(f'num barracks = {self.num_barracks}')
        print('------------------')

        self.parameters = [
            self.num_marines, 
            self.attack_interval_seconds, 
            self.attack_wave_threshold,
            self.num_barracks
        ]
        self.states = []

    async def on_step(self, iteration):
        self.iteration = iteration
        self.attack_waves = set()

        if not self.townhalls.exists:
            for unit in self.workers | self.marines:
                await self.do(unit.attack(self.target))
            return
        self.command_center = self.townhalls.first

        military = {
            MARINE: self.num_marines
        }

        # note that colors are in BGR representation
        self.unit_intel = {
            COMMANDCENTER: [12, (0, 255, 0), "commandcenter"],
            NEXUS:         [12, (0, 255, 0), "nexus"],
            HATCHERY:      [12, (0, 255, 0), "hatchery"],

            SUPPLYDEPOT:        [3, (55, 120, 0), "supplydepot"],
            SUPPLYDEPOTLOWERED: [3, (55, 120, 0), "supplydepotlowered"],
            PYLON:              [3, (55, 120, 0), "pylon"],
            OVERSEER:           [3, (55, 120, 0), "overseer"],
            OVERLORD:           [3, (55, 120, 0), "overlord"],

            REFINERY:    [3, (100, 100, 100), "refinery"],
            ASSIMILATOR: [3, (100, 100, 100), "assimilator"],

            BARRACKS: [5, (200, 40, 0), "barracks"],
            GATEWAY: [5, (200, 40, 0), "gateway"],
            CYBERNETICSCORE: [5, (175, 45, 40), "cyberneticscore"],
            ROBOTICSFACILITY: [5, (128, 32, 32), "cyberneticscore"],

            MARINE: [1, (0, 0, 240), "marine"],
            ZEALOT: [1, (0, 0, 240), "zealot"],
            ZERGLING: [1, (0, 0, 240), "zergling"],
            STALKER: [1, (0, 75, 215), "stalker"],
            OBSERVER: [1, (65, 60, 30), "observer"],

            SCV:   [1, (34, 237, 200), "scv"],
            PROBE: [1, (34, 237, 200), "probe"],
            DRONE: [1, (34, 237, 200), "drone"]
        }

        self.prepare_attack(military, interval=self.attack_interval_seconds)
        await self.manage_workers()
        await self.manage_supply()
        await self.manage_military_training_structures()
        await self.train_military()
        await self.visualize()
        await self.attack()
        await self.task_workers()

    async def visualize(self):
        game_map = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        await self.visualize_map(game_map)
        await self.visualize_resources(game_map)

        # cv assumes (0, 0) top-left => need to flip along horizontal axis
        flipped = cv.flip(game_map, 0)
        self.states.append(flipped)

        cv.imshow('Map', cv.resize(flipped, dsize=None, fx=2, fy=2))
        cv.waitKey(1)

    async def visualize_map(self, game_map):
        # game coordinates need to be represented as (y, x) in 2d arrays
        for typ, intel in self.unit_intel.items():
            for unit in self.units(typ).ready:
                posn = unit.position
                cv.circle(game_map, (int(posn[0]), int(posn[1])), intel[0], intel[1], -1)
            for unit in self.known_enemy_units:
                if unit.name.lower() != intel[2].lower():
                    continue
                posn = unit.position
                x = posn[0]
                y = posn[1]
                l = intel[0] * 1.75
                cv.rectangle(game_map, (int(x), int(y)), (int(x + l), int(y + l)), intel[1], -1)

    async def visualize_resources(self, game_map):
        line_scalar = 40
        minerals = min(1.0, self.minerals / 1200)
        vespene = min(1.0, self.vespene / 1200)
        pop_space = min(1.0, self.supply_left / self.supply_cap)
        supply_usage = self.supply_cap / 200
        military = (self.supply_cap - self.supply_left - self.workers.amount) \
        / (self.supply_cap - self.supply_left)


        cv.line(game_map, (0, 16), (int(line_scalar*minerals), 16), (255, 40, 37), 2)  
        cv.line(game_map, (0, 12), (int(line_scalar*vespene), 12), (25, 240, 20), 2)
        cv.line(game_map, (0, 8),  (int(line_scalar*pop_space), 8), (150, 150, 150), 2)
        cv.line(game_map, (0, 4),  (int(line_scalar*supply_usage), 4), (64, 64, 64), 2)
        cv.line(game_map, (0, 0),  (int(line_scalar*military), 0), (0, 0, 255), 2)

    async def manage_workers(self):
        if self.can_afford(SCV) and self.workers.amount <= 15 \
        and self.command_center.noqueue:
            await self.do(self.command_center.train(SCV))

    async def manage_supply(self):
        supply_threshold = 2 if self.barracks.amount < 3 else 5

        supply_units = self.units.of_type([
            SUPPLYDEPOT, 
            SUPPLYDEPOTLOWERED, 
            SUPPLYDEPOTDROP
        ])

        if self.can_afford(SUPPLYDEPOT):
            if supply_units.amount < 1 or \
            (self.supply_left < supply_threshold and 
                self.already_pending(SUPPLYDEPOT) < 2):
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

        if self.barracks.amount < self.num_barracks:
        # if self.barracks.amount < 3 or \
        # (self.barracks.amount < 5 and self.minerals > 400):
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
        if attack_wave is not None \
        and self.attack_wave_threshold < len(attack_wave):
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
        return self.minutes_elapsed * 60

for _ in range(NUM_EPISODES):
    bot = ProxyRaxRushBot()
    result = sc2.run_game(sc2.maps.get("(2)RedshiftLE"), [
        Bot(Race.Terran, bot),
        Computer(Race.Protoss, Difficulty.VeryHard)
        ], realtime=False)

    if result == Result.Victory:
        np.save(f'training/{int(time())}.npy', np.array((bot.parameters, bot.states)))