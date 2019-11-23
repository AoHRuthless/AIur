from abstract_bot import AbstractBot

import sc2
from sc2 import Race, Difficulty
from sc2.constants import *
from sc2.player import Bot, Computer

class ProxyRaxRushBot(AbstractBot):
    async def on_step(self, iteration):
        self.iteration = iteration
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
        await self.handle_military()
        await self.task_workers()

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

    async def manage_military_add_ons(self):
        pass

    async def manage_military_research_structures(self):
        pass

    async def task_workers(self):
        min_field = self.state.mineral_field.closest_to(self.command_center)
        for scv in self.units(SCV).idle:
            await self.do(scv.gather(min_field))

    async def train_military(self):
        for rax in self.units(BARRACKS).ready.noqueue:
            if not self.can_afford(MARINE):
                break
            await self.do(rax.train(MARINE))

    @property
    def barracks(self):
        return self.units(BARRACKS)
    
    @property
    def marines(self):
        return self.units(MARINE)


# Can beat elite protoss and terran AI with ease
# Loses occasionally to elite early zergling/roach push
result = sc2.run_game(sc2.maps.get("(2)RedshiftLE"), [
    Bot(Race.Terran, ProxyRaxRushBot()),
    Computer(Race.Protoss, Difficulty.VeryHard)
    ], realtime=False)

print("----")
print(result)