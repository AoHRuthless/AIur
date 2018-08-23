import sc2
from sc2 import BotAI

ITERATIONS_PER_MINUTE = 165

class AbstractBot(sc2.BotAI):
    def __init__(self):
        self.attack_waves = set()
        self.max_workers = 65

    async def on_step(self, iteration):
        self.iteration = iteration
        if not self.townhalls.exists:
            for unit in self.units:
                await self.do(unit.attack(self.target))
            return

        await self.distribute_workers()

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
        return self.iteration / ITERATIONS_PER_MINUTE

    @property
    def seconds_elapsed(self):
        """
        Grabs the seconds currently elapsed.
        """
        return self.iteration / (ITERATIONS_PER_MINUTE / 60)