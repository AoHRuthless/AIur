from protocols import military_protocol

import sc2
from sc2 import BotAI
from sc2.constants import *
from sc2.helpers import ControlGroup

ITERATIONS_PER_MINUTE = 165

class AbstractBot(sc2.BotAI, 
    military_protocol.MilitaryProtocol):
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

        added_non_medic_units = False
        attack_wave = None
        for unit in military_ratio:
            if not added_non_medic_units and unit == MEDIVAC:
                continue

            amount = military_ratio[unit]
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
        return self.iteration / ITERATIONS_PER_MINUTE

    @property
    def seconds_elapsed(self):
        """
        Grabs the seconds currently elapsed.
        """
        return self.iteration / (ITERATIONS_PER_MINUTE / 60)