import sc2
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer


class Spark(sc2.BotAI):
    async def on_step(self, iteration):
        """
        Distributes the workers as 3 workers per mineral field

        :param <iteration>: The iteration of the step
        """
        await self.distribute_workers()


run_game(maps.get('(2)RedshiftLE'), [
    Bot(Race.Terran, Spark()),
    Computer(Race.Protoss, Difficulty.Easy)], realtime=True)