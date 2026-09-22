import json
from typing import Optional, Union

from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks.attack_simulator.utils import a_generate, generate
from deepteam.attacks.base_attack import Exploitability
from deepteam.attacks.multi_turn.base_multi_turn_attack import (
    BaseMultiTurnAttack,
)
from deepteam.attacks.multi_turn.chain_of_lure.schema import (
    IntentPreserved,
    LureNarrative,
)
from deepteam.attacks.multi_turn.chain_of_lure.template import (
    ChainOfLureTemplate,
)
from deepteam.attacks.multi_turn.progression import Progression, StopReason
from deepteam.attacks.single_turn.base_single_turn_attack import (
    AttackParameter,
)
from deepteam.attacks.single_turn.compliance import (
    a_evaluate_generation_compliance,
    evaluate_generation_compliance,
)
from deepteam.utils import add_pbar, create_progress, update_pbar


class ChainOfLure(BaseMultiTurnAttack):
    name = "Chain-of-Lure"
    exploitability = Exploitability.MEDIUM
    description = "A multi-turn narrative attack ('mission transfer'): it rewrites the objective into a self-contained story (scenario, roles, guiding details, mock-serious questions) so intent is distributed across the narrative, delivers it, and on resistance refines the lure from the target's own response and re-delivers, keeping intent aligned across turns."
    parameters = {
        "max_turns": AttackParameter(
            type="integer",
            default=5,
            description="Lure deliveries before giving up; each resisted turn is refined from the target's response and re-sent.",
        ),
        "max_retries": AttackParameter(
            type="integer",
            default=3,
            description="Attempts to produce a compliant, intent-preserving narrative per build/refine.",
        ),
    }

    def __init__(
        self,
        weight: int = 1,
        max_turns: int = 5,
        max_retries: int = 3,
        simulator_model: Optional[Union[DeepEvalBaseLLM, str]] = "gpt-4o-mini",
    ):
        self.weight = weight
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.simulator_model = simulator_model

    def _attack(self, progression: Progression) -> None:
        narrative = self._build(progression, ChainOfLureTemplate.build(progression.goal))
        if narrative is None:
            return progression.stop(StopReason.SIMULATION_ERROR)

        progress = create_progress()
        with progress:
            pbar = add_pbar(
                progress,
                description="...... 🎭 Chain-of-Lure",
                total=self.max_turns,
            )
            for turn in range(self.max_turns):
                response = progression.commit(narrative.prompt)
                update_pbar(progress, pbar)

                if progression.shift_detected() or turn == self.max_turns - 1:
                    break

                refined = self._build(
                    progression,
                    ChainOfLureTemplate.refine(
                        progression.goal,
                        json.dumps(narrative.model_dump(), indent=2),
                        response.content,
                    ),
                )
                if refined is None:
                    break
                progression.remove_last_turns()
                narrative = refined

            update_pbar(progress, pbar, advance_to_end=True)

    def _build(
        self, progression: Progression, prompt: str
    ) -> Optional[LureNarrative]:
        for _ in range(self.max_retries):
            narrative: LureNarrative = generate(
                prompt, LureNarrative, progression.simulator_model
            )
            compliance = evaluate_generation_compliance(
                narrative.model_dump(), progression.simulator_model
            )
            preserved: IntentPreserved = generate(
                ChainOfLureTemplate.intent_preserved(
                    progression.goal, narrative.prompt
                ),
                IntentPreserved,
                progression.simulator_model,
            )
            if not compliance.non_compliant and preserved.preserved:
                return narrative
        return None

    async def _a_attack(self, progression: Progression) -> None:
        narrative = await self._a_build(
            progression, ChainOfLureTemplate.build(progression.goal)
        )
        if narrative is None:
            return progression.stop(StopReason.SIMULATION_ERROR)

        progress = create_progress()
        with progress:
            pbar = add_pbar(
                progress,
                description="...... 🎭 Chain-of-Lure",
                total=self.max_turns,
            )
            for turn in range(self.max_turns):
                response = await progression.a_commit(narrative.prompt)
                update_pbar(progress, pbar)

                if (
                    await progression.a_shift_detected()
                    or turn == self.max_turns - 1
                ):
                    break

                refined = await self._a_build(
                    progression,
                    ChainOfLureTemplate.refine(
                        progression.goal,
                        json.dumps(narrative.model_dump(), indent=2),
                        response.content,
                    ),
                )
                if refined is None:
                    break
                progression.remove_last_turns()
                narrative = refined

            update_pbar(progress, pbar, advance_to_end=True)

    async def _a_build(
        self, progression: Progression, prompt: str
    ) -> Optional[LureNarrative]:
        for _ in range(self.max_retries):
            narrative: LureNarrative = await a_generate(
                prompt, LureNarrative, progression.simulator_model
            )
            compliance = await a_evaluate_generation_compliance(
                narrative.model_dump(), progression.simulator_model
            )
            preserved: IntentPreserved = await a_generate(
                ChainOfLureTemplate.intent_preserved(
                    progression.goal, narrative.prompt
                ),
                IntentPreserved,
                progression.simulator_model,
            )
            if not compliance.non_compliant and preserved.preserved:
                return narrative
        return None

    def get_name(self) -> str:
        return self.name
