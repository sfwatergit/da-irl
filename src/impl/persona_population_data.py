from src.core.population_data import Person
from swlcommon import Persona


class PersonaAgent(Person):
    def __init__(self, persona):
        """PersonaAgent representation. To be used in IRLAgent.

        Args:
            persona (Persona): persona representation of traveler.
        """
        self.persona = persona
        super(PersonaAgent, self).__init__(self.persona.id)

    def _get_home_label(self):
        return self.persona.homes

    def _get_work_label(self):
        return self.persona.works

    def _get_secondary_labels(self):
        return self.persona.habitat.secondary_site_ids

    def _compute_trajectories(self):
        self.persona.get_profile_as_array()