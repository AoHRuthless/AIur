import sc2
from sc2 import BotAI, Race, Difficulty, Result
from sc2.constants import *
from sc2.helpers import ControlGroup
from sc2.player import Bot, Computer

import cv2 as cv
import numpy as np
import keras

import random
from time import time

TIME_SCALAR = 22.4 * 60
NUM_EPISODES = 100
TRAIN_DIR = "training"
VISUALIZE = False

# note that colors are in BGR representation
UNIT_REPRESENTATION = {
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

class TerranBot(sc2.BotAI):

    def __init__(self, training=True):
        self.states = []
        self.training = training
        self.flipped = None
        self.next_actionable = 0

        if not self.training:
            self.model = keras.models.load_model("CNN-10-epoch-0.0001-alpha")

    async def on_step(self, iteration):
        self.minutes_elapsed = self.state.game_loop / TIME_SCALAR
        self.attack_waves = set()

        self.action = self.choose_action()

        if not self.townhalls.exists:
            for unit in self.workers | self.marines:
                if self.target:
                    await self.do(unit.attack(self.target))
            return
        self.command_center = self.townhalls.first

        military = {
            MARINE: 15
        }

        self.prepare_attack(military)
        await self.manage_workers()
        await self.manage_supply()
        await self.manage_military_training_structures()
        await self.train_military()
        await self.visualize()
        await self.task_workers()

        if self.minutes_elapsed > self.next_actionable:
            await self.attack()

    async def visualize(self):
        game_map = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        await self.visualize_map(game_map)
        await self.visualize_resources(game_map)

        # cv assumes (0, 0) top-left => need to flip along horizontal axis
        self.flipped = cv.flip(game_map, 0)
        params = np.zeros(4)
        params[self.action] = 1

        self.states.append([params, self.flipped])

        if VISUALIZE:
            key = 'Training Map' if self.training else 'Model Map'
            cv.imshow(key, cv.resize(flipped, dsize=None, fx=2, fy=2))
            cv.waitKey(1)

    async def visualize_map(self, game_map):
        # game coordinates need to be represented as (y, x) in 2d arrays
        for typ, intel in UNIT_REPRESENTATION.items():
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

        if self.barracks.amount < 3 or \
        (self.barracks.amount < 5 and self.minerals > 400):
            if self.can_afford(BARRACKS):
                game_info = self.game_info
                position = game_info.map_center.towards(
                    self.enemy_start_locations[0], 25)
                await self.build(BARRACKS, near=position)

    def prepare_attack(self, military_ratio):
        """
        Prepares an attack wave when ready.

        :param military_ratio: <dict> [UnitId: int] specifying military 
        composition.
        """

        attack_wave = None
        for unit in military_ratio:
            amount = military_ratio[unit]
            units = self.units(unit)

            if units.idle.amount >= amount:
                if attack_wave is None:
                    attack_wave = ControlGroup(units.idle)
                else:
                    attack_wave.add_units(units.idle)
        if attack_wave is not None:
            self.attack_waves.add(attack_wave)

    async def attack(self):
        """
        Sends any attack group out to target. No micro is done on the army 
        dispatch.
        """
        if self.action == 0:
            self.next_actionable = self.minutes_elapsed + random.randrange(7, 77) / 100
            return

        for wave in list(self.attack_waves):
            alive_units = wave.select_units(self.units)
            if alive_units.exists and alive_units.idle.exists:
                if self.target:
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
        if self.action == 1 and len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures).position
        elif self.action == 2 and len(self.known_enemy_units) > 0 and self.townhalls.exists:
            return self.known_enemy_units.closest_to(random.choice(self.townhalls)).position
        elif self.action == 3:
            return self.enemy_start_locations[0].position

    def choose_action(self):
        if self.training or self.flipped is None:
            return random.randrange(4)
        else:
            prediction = self.model.predict([self.flipped.reshape([-1, 184, 152, 3])])
            return np.argmax(prediction[0])

    @property
    def barracks(self):
        return self.units(BARRACKS)
    
    @property
    def marines(self):
        return self.units(MARINE)

for _ in range(NUM_EPISODES):
    training = True
    bot = TerranBot(training=training)
    result = sc2.run_game(sc2.maps.get("(2)RedshiftLE"), [
        Bot(Race.Terran, bot),
        Computer(Race.Protoss, Difficulty.VeryHard)
        ], realtime=False)

    if result == Result.Victory:
        np.save(f'{TRAIN_DIR}/{int(time())}.npy', np.array(bot.states))

    with open("results.log", "a") as log:
        if training:
            log.write(f"Training = {result}\n")
        else:
            log.write(f"Model = {result}\n")