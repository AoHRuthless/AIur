import sc2
from sc2 import BotAI, Race, Difficulty, Result, position
from sc2.constants import *
from sc2.helpers import ControlGroup
from sc2.player import Bot, Computer

import cv2 as cv
import numpy as np
import keras

import random
from time import time

TIME_SCALAR = 22.4
SECONDS_PER_MIN = 60
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
        self.scout_locations = {}

        if not self.training:
            self.model = keras.models.load_model("CNN-10-epoch-0.0001-alpha")

        self.actions = [
            self.standby,
            self.attack,
            self.attack,
            self.attack,
            self.manage_supply,
            self.manage_barracks,
            self.train_workers,
            self.train_marines,
        ]
        self.num_actions = len(self.actions)

    async def on_step(self, iteration):
        self.seconds_elapsed = self.state.game_loop / TIME_SCALAR
        self.minutes_elapsed = self.seconds_elapsed / SECONDS_PER_MIN
        self.attack_waves = set()

        if not self.townhalls.exists:
            target = self.known_enemy_structures.random_or(self.enemy_start_locations[0]).position
            for unit in self.workers | self.marines:
                await self.do(unit.attack(target))
            return
        self.command_center = self.townhalls.first

        military = {
            MARINE: 15
        }

        self.action = self.make_action_selection()
        print(f"Action chosen == {self.action}")
        self.prepare_attack(military)

        await self.task_workers()
        await self.scout()
        await self.visualize()
        await self.take_action()

    async def visualize(self):
        game_map = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        await self.visualize_map(game_map)
        await self.visualize_resources(game_map)

        # cv assumes (0, 0) top-left => need to flip along horizontal axis
        self.flipped = cv.flip(game_map, 0)
        params = np.zeros(self.num_actions)
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

    async def train_workers(self):
        if self.can_afford(SCV) and self.workers.amount <= 15 \
        and self.command_center.noqueue:
            await self.do(self.command_center.train(SCV))

    async def scout(self):
        if self.seconds_elapsed % 2 == 0 or self.minutes_elapsed > 5:
            return
        expand_distances = {}

        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            expand_distances[distance_to_enemy_start] = el

        distance_keys = sorted(k for k in expand_distances)
        unit_tags = [unit.tag for unit in self.units]

        to_be_removed = []
        for s in self.scout_locations:
            if s not in unit_tags:
                to_be_removed.append(s)

        for scout in to_be_removed:
            del self.scout_locations[scout]

        assign_scout = True

        for unit in self.workers:
            if unit.tag in self.scout_locations:
                assign_scout = False

        if assign_scout:
            workers = self.workers.idle if len(self.workers.idle) > 0 else self.workers.gathering
            for worker in workers[:1]:
                if worker.tag not in self.scout_locations:
                    for dist in distance_keys:
                        try:
                            location = next(v for k, v in expand_distances.items() if k == dist)
                            active_locations = [self.scout_locations[k] for k in self.scout_locations]

                            if location not in active_locations:
                                await self.do(worker.move(location))
                                self.scout_locations[worker.tag] = location
                                break
                        except Exception as e:
                            pass

        for worker in self.workers:
            if worker.tag in self.scout_locations:
                await self.do(worker.move(self.vary_loc(self.scout_locations[worker.tag])))

    def vary_loc(self, location):
        x = location[0] + random.randrange(-10, 10)
        y = location[1] + random.randrange(-10, 10)

        x = min(self.game_info.map_size[0], max(x, 0))
        y = min(self.game_info.map_size[1], max(y, 0))

        return position.Point2(position.Pointlike((x,y)))

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

    async def manage_barracks(self): 
        if not self.units.of_type([
            SUPPLYDEPOT, 
            SUPPLYDEPOTLOWERED, 
            SUPPLYDEPOTDROP
            ]).ready.exists:
            return   

        # if self.barracks.amount < 3 or \
        # (self.barracks.amount < 5 and self.minerals > 400):
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

    async def standby(self):
        self.next_actionable = self.seconds_elapsed + random.randrange(1, 15)

    async def attack(self):
        """
        Sends any attack group out to target. No micro is done on the army 
        dispatch.
        """

        target = None 
        if self.action == 1:
            if len(self.known_enemy_structures) > 0:
                target = random.choice(self.known_enemy_structures).position
        elif self.action == 2:
            if len(self.known_enemy_units) > 0:
                target = self.known_enemy_units.closest_to(random.choice(self.townhalls)).position
        elif self.action == 3:
            target = self.enemy_start_locations[0].position
        else:
            return

        if target is None:
            return


        for wave in list(self.attack_waves):
            alive_units = wave.select_units(self.units)
            if alive_units.exists and alive_units.idle.exists:
                for unit in wave.select_units(self.units):
                    await self.do(unit.attack(target))
            else:
                self.attack_waves.remove(wave)

    async def task_workers(self):
        min_field = self.state.mineral_field.closest_to(self.command_center)
        for scv in self.workers.idle:
            await self.do(scv.gather(min_field))

    async def train_marines(self):
        for rax in self.barracks.ready.noqueue:
            if not self.can_afford(MARINE):
                break
            await self.do(rax.train(MARINE))

    def make_action_selection(self):
        if self.seconds_elapsed <= self.next_actionable:
            return

        if self.training or self.flipped is None:
            return random.randrange(self.num_actions)
        else:
            prediction = self.model.predict([self.flipped.reshape([-1, 184, 152, 3])])
            return np.argmax(prediction[0])

    async def take_action(self):
        if self.seconds_elapsed <= self.next_actionable:
            return

        try:
            await self.actions[self.action]()
        except Exception as err:
            print(str(err))

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