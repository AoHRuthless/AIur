import sc2
from sc2 import BotAI, Race, Difficulty, Result, position
from sc2.constants import *
from sc2.helpers import ControlGroup
from sc2.player import Bot, Computer

from model import DQNModel

import cv2 as cv
import numpy as np
import keras

import math
import random
from time import time

TIME_SCALAR = 22.4
SECONDS_PER_MIN = 60
NUM_EPISODES = 1000
TRAIN_DIR = "training"
VISUALIZE = False
REPLAY_BATCH_SIZE = 80
UPDATE_TARGET_FREQ = 1000

class TerranBot(sc2.BotAI):

    def __init__(self, epsilon=1.0):
        self.next_actionable = 0
        self.scout_locations = {}
        self.rewards = []

        weighted_actions = {
            self.no_op: 1,
            self.standby: 1,
            self.attack: 3,
            self.manage_supply: 5,
            self.adjust_refinery_assignment: 1,
            self.manage_refineries: 1,
            self.manage_barracks: 3,
            self.manage_barracks_tech_labs: 1,
            self.manage_barracks_reactors: 1,
            self.manage_factories: 1,
            self.manage_starports: 1,
            self.train_workers: 3,
            self.train_marines: 7,
            self.train_marauders: 4,
            self.train_hellions: 1,
            self.train_medivacs: 1,
            self.upgrade_cc: 1,
            self.expand: 4,
            self.scout: 1,
            self.calldown_mules: 2,
        }

        self.actions = []
        for action_fn, weight in weighted_actions.items():
            for _ in range(weight):
                self.actions.append(action_fn)

        self.curr_state = None
        self.num_actions = len(self.actions)
        self.dqn = DQNModel(self.actions, eps=epsilon)

        self.iteration = 0

        # <list> [UnitId] specifying military composition.
        self.military_distribution = [
            MARINE,
            MARAUDER,
            HELLION
        ]

        self.tl_tags = []
        self.techlab_research_options = [
            RESEARCH_COMBATSHIELD, 
            RESEARCH_CONCUSSIVESHELLS, 
            BARRACKSTECHLABRESEARCH_STIMPACK
        ]

    async def on_step(self, iteration):
        self.seconds_elapsed = self.state.game_loop / TIME_SCALAR
        self.minutes_elapsed = self.seconds_elapsed / SECONDS_PER_MIN
        self.attack_waves = set()
        self.iteration += 1
        self.num_troops_per_wave = min(14 + self.minutes_elapsed, 30)

        if self.curr_state is not None:
            self.prev_state = self.curr_state
            self.remember()
            if self.iteration % REPLAY_BATCH_SIZE == 0:
                self.dqn.replay(REPLAY_BATCH_SIZE)
            if self.iteration % UPDATE_TARGET_FREQ == 0:
                self.dqn.train_target_model()

        await self.visualize()

        if not self.townhalls.exists:
            target = self.known_enemy_structures.random_or(self.enemy_start_locations[0]).position
            for unit in self.workers | self.military_units:
                await self.do(unit.attack(target))
            return

        ready_techlabs = self.units(BARRACKSTECHLAB).ready
        if len(ready_techlabs) != self.tl_tags:
            self.tl_tags = []
            for techlab in ready_techlabs:
                self.tl_tags.append(techlab.tag)

        if len(self.techlab_research_options) > 0:
            for techlab in ready_techlabs:
                try:
                    to_research = random.choice(self.techlab_research_options)
                    if self.can_afford(to_research):
                        await self.do(techlab(to_research))
                        self.techlab_research_options = \
                        self.techlab_research_options.filter(lambda x: x != to_research)
                except Exception as err:
                    pass

        for cc in self.townhalls:
            enemies = self.known_enemy_units.closer_than(25.0, cc).filter(
                lambda x: x.name.lower() not in ["scv", "drone", "probe"])
            if len(enemies) > 0:
                target = random.choice(enemies)
                for unit in self.military_units:
                    await self.do(unit.attack(target))
                break

        self.action = self.make_action_selection()

        # print(f"action chosen == {self.action}")
        self.prepare_attack()
        if len(list(self.attack_waves)) > 0 and self.units(MEDIVAC).idle.amount > 0:
            alive_units = list(self.attack_waves)[0].select_units(self.units)
            for med in self.units(MEDIVAC).idle:
                await self.do(med.attack(alive_units.first.position))

        await self.distribute_workers()
        await self.lower_depots()
        await self.take_action()

    async def no_op(self):
        pass

    async def standby(self):
        self.next_actionable = self.seconds_elapsed + random.randrange(1, 37)

    async def take_action(self):
        if self.seconds_elapsed <= self.next_actionable:
            return

        try:
            await self.actions[self.action]()
        except Exception as err:
            print(str(err))

    def make_action_selection(self):
        if self.seconds_elapsed <= self.next_actionable or self.curr_state is None:
            return 0

        return self.dqn.choose_action(self.curr_state)

    def remember(self, reward=None, done=False):
        reward_value = reward if reward else (self.state.score.score / (200 * self.seconds_elapsed))
        self.rewards.append(reward_value)
        self.dqn.remember(self.prev_state, self.action, reward_value, self.curr_state, done)

    #### WORKERS ####
    #################

    async def train_workers(self):
        if not self.can_afford(SCV):
            return

        for cc in self.townhalls.ready.filter(lambda x: len(x.orders) < 3):
            if len(self.workers) < 18 * len(self.townhalls):
                await self.do(cc.train(SCV))

    async def manage_supply(self):
        if self.can_afford(SUPPLYDEPOT) \
        and self.supply_left < 10 and self.already_pending(SUPPLYDEPOT) < 2:
            position = self.townhalls.ready.random.position.towards(
                self.game_info.map_center, 5)
            await self.build(SUPPLYDEPOT, position)

    async def lower_depots(self):
        for sd in self.units(SUPPLYDEPOT).ready:
            await self.do(sd(MORPH_SUPPLYDEPOT_LOWER))

    async def upgrade_cc(self):
        for cc in self.units(COMMANDCENTER).idle:
            if self.barracks.ready.exists and self.can_afford(ORBITALCOMMAND):
                await self.do(cc(UPGRADETOORBITAL_ORBITALCOMMAND))

    async def calldown_mules(self):
        for oc in self.units(ORBITALCOMMAND).filter(lambda x: x.energy >= 50):
            mfs = self.state.mineral_field.closer_than(10, oc)
            if mfs:
                mf = max(mfs, key=lambda x: x.mineral_contents)
                await self.do(oc(CALLDOWNMULE_CALLDOWNMULE, mf))

    async def expand(self):
        try:
            if self.can_afford(COMMANDCENTER):
                await self.expand_now(max_distance=100)
        except Exception as err:
            print(str(err))

    async def manage_refineries(self):
        for cc in self.units(COMMANDCENTER).ready:
            vgs = self.state.vespene_geyser.closer_than(16.0, cc)
            for vg in vgs:
                if not self.can_afford(REFINERY):
                    break
                worker = self.select_build_worker(vg.position)
                if worker is None:
                    break
                if not self.units(REFINERY).closer_than(2.0, vg).exists:
                    await self.do(worker.build(REFINERY, vg))

    async def adjust_refinery_assignment(self):
        r = self.units(REFINERY).ready.random
        if r.assigned_harvesters < r.ideal_harvesters:
            w = self.workers.closer_than(16.0, r)
            if w.exists:
                await self.do(w.random.gather(r))

    #### MILITARY ####
    ##################

    async def attack(self):
        """
        Sends any attack group out to target. No micro is done on the army 
        dispatch.
        """

        if len(self.known_enemy_structures) > 0:
            target = random.choice(self.known_enemy_structures).position
        elif len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.townhalls)).position
        else:
            target = self.enemy_start_locations[0].position

        for wave in list(self.attack_waves):
            alive_units = wave.select_units(self.units)
            if alive_units.exists and alive_units.idle.exists:
                for unit in wave.select_units(self.units):
                    await self.do(unit.attack(target))
            else:
                self.attack_waves.remove(wave)

    async def manage_barracks(self): 
        if not self.depots.ready.exists:
            return

        if self.can_afford(BARRACKS) and self.barracks.amount < 1 + self.minutes_elapsed:
            depot = self.depots.ready.random
            await self.build(BARRACKS, near=depot)

    async def manage_barracks_tech_labs(self):
        rax = self.barracks.ready.noqueue.random
        if rax.add_on_tag == 0:
            await self.do(rax.build(BARRACKSTECHLAB))

    async def manage_barracks_reactors(self):
        rax = self.barracks.ready.noqueue.random
        if rax.add_on_tag == 0:
            await self.do(rax.build(BARRACKSREACTOR))

    async def manage_factories(self): 
        if not self.depots.ready.exists:
            return
        if not self.barracks.ready.exists:
            return

        if self.can_afford(FACTORY) and self.units(FACTORY).amount < 3:
            depot = self.depots.ready.random
            await self.build(FACTORY, near=depot)

    async def manage_starports(self): 
        if not self.depots.ready.exists:
            return
        if not self.barracks.ready.exists:
            return
        if not self.units(FACTORY).ready.exists:
            return

        if self.can_afford(STARPORT) and self.units(STARPORT).amount < 2:
            depot = self.depots.ready.random
            await self.build(STARPORT, near=depot)

    async def train_marines(self):
        for rax in self.barracks.ready.filter(lambda x: x.add_on_tag not in self.tl_tags and len(x.orders) < 3):
            if not self.can_afford(MARINE):
                break
            await self.do(rax.train(MARINE))

    async def train_marauders(self):
        for rax in self.barracks.ready.filter(lambda x: x.add_on_tag in self.tl_tags and len(x.orders) < 3):
            if not self.can_afford(MARAUDER):
                break
            await self.do(rax.train(MARAUDER))

    async def train_hellions(self):
        for f in self.units(FACTORY).ready.filter(lambda x: len(x.orders) < 3):
            if not self.can_afford(HELLION):
                break
            await self.do(f.train(HELLION))

    async def train_medivacs(self):
        for sp in self.units(STARPORT).ready.filter(lambda x: len(x.orders) < 3):
            if not self.can_afford(MEDIVAC):
                break
            await self.do(sp.train(MEDIVAC))

    def prepare_attack(self):
        """
        Prepares an attack wave when ready.
        """
        total = 0
        for unit in self.military_distribution:
            units = self.units(unit)
            total += units.idle.amount

        if total >= self.num_troops_per_wave:
            attack_wave = None

            for unit in self.military_distribution:
                units = self.units(unit)

                if attack_wave is None:
                    attack_wave = ControlGroup(units.idle)
                else:
                    attack_wave.add_units(units.idle)

            self.attack_waves.add(attack_wave)

    #### VISUALIZATION ####
    #######################

    async def visualize(self):
        game_map = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)
        await self.visualize_map(game_map)
        await self.visualize_resources(game_map)

        # cv assumes (0, 0) top-left => need to flip along horizontal axis
        curr_state = cv.flip(game_map, 0)

        if VISUALIZE:
            cv.imshow('Map', cv.resize(curr_state, dsize=None, fx=2, fy=2))
            cv.waitKey(1)
        self.curr_state = curr_state.reshape([-1, 184, 152, 3])

    async def visualize_map(self, game_map):
        # game coordinates need to be represented as (y, x) in 2d arrays

        for unit in self.units().ready:
            posn = unit.position
            cv.circle(game_map, (int(posn[0]), int(posn[1])), int(unit.radius*8), (0, 0, 255), math.ceil(int(unit.radius*0.5)))

        for unit in self.known_enemy_units:
            posn = unit.position
            cv.circle(game_map, (int(posn[0]), int(posn[1])), int(unit.radius*8), (255, 0, 0), math.ceil(int(unit.radius*0.5)))

    async def visualize_resources(self, game_map):
        line_scalar = 40
        minerals = min(1.0, self.minerals / 1200)
        vespene = min(1.0, self.vespene / 1200)
        pop_space = min(1.0, self.supply_left / max(1.0, self.supply_cap))
        supply_usage = self.supply_cap / 200
        military = (self.supply_cap - self.supply_left - self.workers.amount) \
        / max(1, self.supply_cap - self.supply_left)


        cv.line(game_map, (0, 16), (int(line_scalar*minerals), 16), (255, 40, 37), 2)  
        cv.line(game_map, (0, 12), (int(line_scalar*vespene), 12), (25, 240, 20), 2)
        cv.line(game_map, (0, 8),  (int(line_scalar*pop_space), 8), (150, 150, 150), 2)
        cv.line(game_map, (0, 4),  (int(line_scalar*supply_usage), 4), (64, 64, 64), 2)
        cv.line(game_map, (0, 0),  (int(line_scalar*military), 0), (0, 0, 255), 2)

    #### SCOUTING ####
    ##################

    async def scout(self):
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

    #### HELPERS ####
    #################

    @property
    def depots(self):
        return self.units.of_type([
            SUPPLYDEPOT, 
            SUPPLYDEPOTLOWERED, 
            SUPPLYDEPOTDROP
            ])

    @property
    def barracks(self):
        return self.units(BARRACKS)

    @property
    def military_units(self):
        return self.marines | self.marauders | self.medivacs | self.hellions
    
    @property
    def marines(self):
        return self.units(MARINE)

    @property
    def marauders(self):
        return self.units(MARAUDER)

    @property
    def medivacs(self):
        return self.units(MEDIVAC)

    @property
    def hellions(self):
        return self.units(HELLION)

epsilon = 1.0
gamma = 0.99
try:
    for episode in range(NUM_EPISODES):
        bot = TerranBot(epsilon=epsilon)
        result = sc2.run_game(sc2.maps.get("(2)RedshiftLE"), [
            Bot(Race.Terran, bot),
            Computer(Race.Protoss, Difficulty.MediumHard)
            ], realtime=False)

        if result == Result.Victory:
            bot.remember(reward=1000, done=True)
        else:
            bot.remember(reward=-1000, done=True)

        reward = 0
        for idx, r in enumerate(reversed(bot.rewards)):
            reward += gamma**idx + r

        bot.dqn.save(f"{TRAIN_DIR}/terran-dqn.h5")
        epsilon = bot.dqn.epsilon

        with open("results.log", "a") as log:
            log.write(f"episode: {episode + 1}/{NUM_EPISODES}, epsilon: {epsilon:.3}, reward: {reward:.4f}, result: {result}\n")
except KeyboardInterrupt as err:
    pass
