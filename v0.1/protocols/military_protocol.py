from abc import ABC, abstractmethod

class MilitaryProtocol(ABC):

    async def handle_military(self):
        await self.manage_military_training_structures()
        await self.manage_military_add_ons()
        await self.manage_military_research_structures()
        await self.train_military()
        await self.attack()

    @abstractmethod
    def prepare_attack(self, military_ratio, interval=10):
        pass

    @abstractmethod
    async def manage_military_training_structures(self):
        pass

    @abstractmethod
    async def manage_military_add_ons(self):
        pass

    @abstractmethod
    async def manage_military_research_structures(self):
        pass

    @abstractmethod
    async def train_military(self):
        pass

    @abstractmethod
    async def attack(self):
        pass