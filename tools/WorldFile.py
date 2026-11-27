"""Saving a whole world to a file and loading it back, to carry on exactly as it would have.

A world file is the same for the Python and C++ versions, so a world saved by either loads in both.
Format 2, written now:

    WORLD_MAGIC (16 bytes), the JSON part's length (8 bytes, little-endian), the JSON part (UTF-8),
    the count of numbers (8 bytes), the numbers part: 64-bit little-endian floats,
    the count of whole numbers (8 bytes), the whole-numbers part: 32-bit little-endian integers.

The JSON part holds the world's settings, its rules, and everything in it: the random generator's state,
the grass, the seeds waiting to sprout, every Bitzkey and every field of it, and the insights. What would
make it enormous goes in the other parts, and the JSON part names where:

- a brain is {"at": its first number, "depth": hidden layers, "width": neurons in each, "traced": whether
  learning traces follow}. Its numbers are each hidden layer's weights (width rows, each reading the senses
  or the layer before) and biases, then the output weights (a row of width for each action) and biases;
  then, if traced, its traces, laid out the same. A brain saved without a width, before brains grew wider
  with intellect, is OLD_WIDTH wide, and loads resized to its owner's shape, without its traces.
- a set of senses (its last, or a memory's) is the position of INPUT_SIZE numbers.
- the census's snapshots are [turn, the first whole number, how many]: the identifiers alive then.
- "lineage" {"at", "count", "first"}: the genome of every Bitzkey from identifier "first" on (0 if missing;
  those before it are off the graph and forgotten), three whole numbers each (the social background word,
  the ring, the dot, as unsigned words).
- "death_tally": every death since the world began, counted: "causes" and "activities" {label: four counts, by species},
  "whom" (four rows of four counts: the dead's species, by the killers'), "killers" (how many ever killed), and
  "by_killer" [[identifier, species, kills, {doing: kills}]...], the killers kept. Missing, as from versions before it,
  it is tallied from the deaths the file keeps whole.
- "death_records" {"at", "count"}: every death of the last RECENT_DEATH_TURNS turns, seven whole numbers each: the dead's identifier, its cause
  (an index into "death_strings"), its species (an index into SPECIES), the turn, the killer's identifier,
  its species and what it was doing (an index into "death_strings"); the last three -1 for none.

Elsewhere a genome is 24 hex digits: the social background word, then the ring, then the dot. Format 1 had no
counts and no whole-numbers part, the numbers running to the file's end, and kept the census, lineage and
deaths in the JSON part; it still loads.
"""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tools import Rules
from tools import Simulation as world
from tools.Brain import ACTIONS, INPUT_SIZE, MAX_WIDTH, MEMORY_NEURONS, OUTPUTS, OUTSIDE_SENSES, SENSES, VALUE_SCALE, Brain, fresh_bias
from tools.Genome import GENOME_VERSION, TEMPERAMENT, convert_v3
from tools.Insights import DeathTally
from tools.Simulation import GEAR_KINDS

BOW_PILE = 12  # a bow's pile, as the C++ version numbers it, after the cooked food's
from tools.Simulation import SPECIES, Bitzkey, Death, Simulation, brain_shape

WORLD_MAGIC = b"BITZKEY WORLD 2\n"
WORLD_MAGIC_1 = b"BITZKEY WORLD 1\n"  # format 1's, still read
WORLD_FORMAT = 2
WORD = 0xFFFFFFFF
WORLD_SUFFIX = ".bzworld"
OLD_WIDTH = 34  # neurons in each hidden layer of a brain saved without a width
# Rules that say how this computer runs a world rather than how the world unfolds: loading keeps the user's own.
MACHINE_RULES = ("USE_GPU",)


def _optional_list(value) -> list | None:
    return None if value is None else list(value)


class _Numbers:
    """The numbers part, as it is written: each array's numbers appended, and where they start."""

    def __init__(self) -> None:
        self.parts: list[np.ndarray] = []
        self.count = 0

    def add(self, values) -> int:
        array = np.ascontiguousarray(values, dtype="<f8").ravel()
        self.parts.append(array)
        at = self.count
        self.count += array.size
        return at

    def brain(self, brain: Brain | None) -> dict | None:
        if brain is None:
            return None
        sets = [(brain.layers, brain.output_weights, brain.output_biases)]
        if brain.traces is not None:
            *hidden, (weights, biases) = brain.traces
            sets.append((hidden, weights, biases))
        at = self.count
        for layers, output_weights, output_biases in sets:
            for weights, biases in layers:
                self.add(weights)
                self.add(biases)
            self.add(output_weights)
            self.add(output_biases)
        return {"at": at, "depth": brain.depth, "width": brain.width, "traced": brain.traces is not None}

    def senses(self, senses) -> int | None:
        return None if senses is None else self.add(senses)

    def data(self) -> bytes:
        return b"".join(part.tobytes() for part in self.parts)


class _Reader:
    """The numbers part, as it is read back, and the whole numbers."""

    def __init__(self, data: bytes, ints: bytes = b"") -> None:
        if len(data) % 8:
            raise ValueError("The world file's numbers are cut short.")
        self.numbers = np.frombuffer(data, dtype="<f8")
        self.ints = np.frombuffer(ints, dtype="<i4")
        # For a world saved with other senses: how many it saved, and where each sensed now lies among them, or -1.
        self.saved_inputs = INPUT_SIZE
        self.sense_from: list[int] | None = None
        # For a world saved with other outputs (actions, calls): how many it saved, where each output now lies among
        # them, or -1, and what each action it saved is now, or -1 for one no longer an action.
        self.saved_outputs = len(OUTPUTS)
        self.output_from: list[int] | None = None
        self.action_now: list[int] | None = None
        self.uncentred = False  # saved before brains read their senses centred: each is made to, choosing as before
        self.unscaled = False  # saved before brains held their values in hundredths: each is made to

    def whole(self, at: int, count: int) -> np.ndarray:
        if at < 0 or at + count > self.ints.size:
            raise ValueError("The world file's whole numbers are cut short.")
        return self.ints[at:at + count]

    def take(self, at: int, count: int) -> np.ndarray:
        if at < 0 or at + count > self.numbers.size:
            raise ValueError("The world file's numbers are cut short.")
        return self.numbers[at:at + count].astype(float)

    def brain(self, record: Mapping | None, dna: int) -> Brain | None:
        """The brain saved at `record`, resized to fit the genome `dna` if it was saved another shape."""
        if record is None:
            return None
        at, depth, width = int(record["at"]), int(record["depth"]), int(record.get("width", OLD_WIDTH))
        if depth < 1 or not 0 < width <= MAX_WIDTH:
            raise ValueError("The world file holds a brain of no sensible shape.")
        sets = []
        for _ in range(2 if record["traced"] else 1):
            layers = []
            for index in range(depth):
                if index == 0:  # a row of senses for each neuron
                    weights = np.array([self._as_senses(self.take(at + neuron * self.saved_inputs, self.saved_inputs))
                                        for neuron in range(width)])
                    at += width * self.saved_inputs
                else:
                    weights = self.take(at, width * width).reshape(width, width)
                    at += width * width
                biases = self.take(at, width)
                at += width
                layers.append((weights, biases))
            # An output added since starts unweighted, and one no longer had is let go.
            saved = self.saved_outputs
            saved_weights = self.take(at, saved * width).reshape(saved, width)
            at += saved * width
            saved_biases = self.take(at, saved)
            at += saved
            output_from = self.output_from if self.output_from is not None else range(len(OUTPUTS))
            output_weights, output_biases = np.zeros((len(OUTPUTS), width)), np.zeros(len(OUTPUTS))
            for output, source in enumerate(output_from):
                if source >= 0:
                    output_weights[output], output_biases[output] = saved_weights[source], saved_biases[source]
                elif not sets:  # an output it never had, as a new brain holds it
                    output_biases[output] = fresh_bias(output) * (1.0 if self.unscaled else VALUE_SCALE)
            sets.append((layers, output_weights, output_biases))
        brain = Brain(*sets[0])
        if len(sets) > 1:
            layers, output_weights, output_biases = sets[1]
            brain.traces = [(weights.copy(), biases.copy()) for weights, biases in layers]
            brain.traces.append((output_weights.copy(), output_biases.copy()))
        if self.uncentred:
            brain.centre_inputs()
        if self.unscaled:
            brain.scale_values()
        shape = brain_shape(dna)
        return brain if (brain.depth, brain.width) == shape else brain.resized(*shape)

    def action(self, saved: int) -> int:
        """What an action it saved is now, or -1 for one no longer an action."""
        if self.action_now is None:
            return saved
        return self.action_now[saved] if 0 <= saved < len(self.action_now) else -1

    def senses(self, at: int | None) -> list[float] | None:
        return None if at is None else self._as_senses(self.take(int(at), self.saved_inputs)).tolist()

    def _as_senses(self, saved: np.ndarray) -> np.ndarray:
        """One set of saved senses, or one neuron's weights from them, as this version's senses."""
        if self.sense_from is None:
            return saved
        return np.array([0.0 if index < 0 else saved[index] for index in self.sense_from])


def _genome(dna: int) -> str:
    return f"{dna:024X}"


def _mind(mind: Mapping[str, float]) -> list[float]:
    return [float(mind[trait]) for trait in TEMPERAMENT]


def _mind_from(values) -> dict[str, float]:
    return {trait: float(value) for trait, value in zip(TEMPERAMENT, values)}


def desire_record(agent) -> dict:
    """What it likes, its sparks and faded pleasures (see DESIRE): only what is not 0. As the C++ version's."""
    things = [[thing, float(agent.want_things[thing]), float(agent.spark_things[thing]), float(agent.fade_things[thing]), agent.spark_from_things[thing]]
              for thing in range(21) if agent.want_things[thing] or agent.spark_things[thing] or agent.fade_things[thing]]
    deeds = [[deed, float(agent.want_deeds[deed]), float(agent.spark_deeds[deed]), float(agent.fade_deeds[deed]), agent.spark_from_deeds[deed]]
             for deed in range(len(agent.want_deeds)) if agent.want_deeds[deed] or agent.spark_deeds[deed] or agent.fade_deeds[deed]]
    return {"head": [float(agent.good_drift), float(agent.joy_mark), int(agent.goal_desire), agent.desires_formed,
                     agent.desires_reached, agent.desire_reach_turns, *agent.desires_by_cue], "things": things, "deeds": deeds,
            "places": [[thing, agent.place_x[thing], agent.place_y[thing], agent.place_turn[thing]] for thing in range(21) if agent.place_turn[thing] >= 0]}


def ingenuity_record(agent) -> dict:
    """What it learnt opens what, its steps (see OPENS_RATE): a node saved as a thing's index, or -1 less a deed's. As the
    C++ version's."""
    def node(n: int) -> int:
        return n if n < 21 else -1 - (n - 21)
    return {"head": [agent.step_turn, agent.chain_goal_turn, 0, agent.chains_begun, agent.steps_reached,
                     agent.steps_given_up, agent.links_first_followed, int(agent.has_open), agent.chains_unfinished],
            "failed": [[node(a), node(b)] for a, b in agent.failed_links],
            "links": [[node(l[0]), node(l[1]), float(l[2]), int(l[3])] for l in agent.opens],
            "steps": [node(step) for step in agent.goal_steps],
            "open": [node(n) for n, was in enumerate(agent.was_open) if was]}


def project_record(agent) -> dict:
    """Its long aims and project (see PROJECT_PULL). As the C++ version's."""
    return {"values": [float(v) for v in agent.aim_value], "last": [float(v) for v in agent.aim_last], "agree": [float(v) for v in agent.aim_agree],
            "asking": [agent.project_stuck_turns, agent.project_stuck_calls, agent.project_unstuck_after_call, agent.project_unstuck_silent,
                       agent.instructed_by_sign, agent.instructed_by_word],
            "head": [int(agent.aim_sampled), agent.project_aim, agent.project_dir, agent.project_turn, agent.project_progress_turn,
                     agent.project_x, agent.project_y, agent.project_group, agent.project_concept, agent.project_from, agent.project_waited,
                     int(agent.project_confided), int(agent.goal_project), int(agent.goal_instructed), agent.instructed_by,
                     agent.shown_step, agent.shown_turn, int(agent.plan_open), agent.projects_taken, agent.projects_given_up,
                     agent.project_progress_turns, agent.instructions_taken, agent.confided, agent.confided_to],
            "good": [float(agent.aim_good), float(agent.aim_good_drift), float(agent.project_last)],
            "deeds": [[deed, float(v)] for deed, v in enumerate(agent.project_deeds) if v != 0.0],
            "things": [[thing, float(v)] for thing, v in enumerate(agent.project_things) if v != 0.0]}


def goal_record(agent) -> dict:
    """Its needs, what eases them and its goal (see GOAL_JOY): only reliefs learnt. As the C++ version's."""
    needs = [float(v) for pair in zip(agent.needs, agent.need_drift) for v in pair]
    things = [[n, thing, float(v)] for n in range(6) for thing, v in enumerate(agent.relief_things[n]) if v != 0.0]
    deeds = [[n, deed, float(v)] for n in range(6) for deed, v in enumerate(agent.relief_deeds[n]) if v != 0.0]
    ones = [[one[0], float(one[1]), *(float(v) for v in one[2]), float(one[3]), float(one[4]), float(one[5]), int(one[6]), int(one[7]), int(one[8]), int(one[9])] for one in agent.relief_ones]
    return {"head": [agent.goal_kind, agent.goal_target, agent.goal_need, agent.goal_turn, float(agent.goal_urgency), float(agent.goal_close),
                     agent.goals_formed, agent.goals_reached, agent.goals_abandoned, agent.goals_switched, agent.goals_dropped,
                     agent.goal_reach_turns, int(agent.has_needs)],
            "needs": needs, "things": things, "deeds": deeds, "ones": ones, "desire": desire_record(agent), "ingenuity": ingenuity_record(agent), "project": project_record(agent),
            "vendetta": {"enemies": [[e[0], float(e[1]), float(e[2]), e[3], e[4], e[5], int(e[6])] for e in agent.enemies],
                         "head": [int(agent.goal_vendetta), agent.humbled_until, agent.vendettas_formed, agent.vendettas_settled, agent.vendettas_ended,
                                  agent.challenges_made, agent.challenges_won, agent.challenges_faced, agent.challenges_held]},
            "asking": [int(agent.stuck), agent.stuck_goal_turn, agent.help_call_turn, agent.stuck_turns, agent.free_turns, agent.stuck_calls,
                       agent.free_calls, agent.unstuck_after_call, agent.unstuck_silent, agent.helped, agent.helping, agent.helped_turn, agent.help_taught_turn]}


def _agent_record(agent: Bitzkey, numbers: _Numbers) -> dict:
    return {
        "identifier": agent.identifier, "dna": _genome(agent.dna), "x": agent.x, "y": agent.y,
        "hunger": float(agent.hunger), "health": float(agent.health), "stamina": float(agent.stamina), "age": agent.age,
        "cooldown": agent.cooldown, "fruit": agent.fruit, "picked_by": agent.picked_by,
        "last_action": agent.last_action, "brain": numbers.brain(agent.brain), "generation": agent.generation,
        "children": agent.children, "mother": agent.mother, "father": agent.father, "taught_by": agent.taught_by,
        "carried_meat": agent.carried_meat, "carried_fruit": agent.carried_fruit, "weapon": agent.weapon, "armour": agent.armour_worn, "weapon_uses": agent.weapon_uses, "armour_uses": agent.armour_uses, "bow": agent.bow, "bow_uses": agent.bow_uses, "archery": agent.archery, "desire": agent.desire, "wagon": agent.wagon, "tutored_turn": agent.tutored_turn,
        "infant_taught": agent.infant_taught, "crafts": agent.crafts, "craft_tries": agent.craft_tries, "cooked_meat": agent.cooked_meat, "cooked_fruit": agent.cooked_fruit,
        "carried_stone": agent.carried_stone,
        "carried_bone": agent.carried_bone, "carried_wood": agent.carried_wood, "carried_seed": agent.carried_seed, "boredom": float(agent.boredom), "shelter_known": float(agent.shelter_known), "road_known": float(agent.road_known), "cooked_known": float(agent.cooked_known), "plenty": float(agent.plenty), "plenty_known": float(agent.plenty_known), "hide": agent.hide, "hide_armour": float(agent.hide_armour), "spare_tools": [list(tool) for tool in agent.spare_tools], "spare_hides": [[hide[0], float(hide[1])] for hide in agent.spare_hides], "skinned": agent.skinned, "thefts": agent.thefts, "mark": agent.mark, "realm": agent.realm, "standout": agent.standout, "follows": agent.follows, "followers": agent.followers, "leader": agent.leader, "rank": agent.rank, "emperor": agent.emperor, "trespass": agent.trespass, "meat_lean": float(agent.meat_lean), "expected": agent.expected, "last_tried": [agent.last_tried.get(action, 0) for action in ACTIONS], "died_turn": agent.died_turn, "assailant": agent.assailant,
        "panicked": agent.panicked, "primal_fear": agent.primal_fear, "rivalry": float(agent.rivalry),
        "own_rivalry": float(agent.own_rivalry), "hunts_grazers": agent.hunts_grazers,
        "hunts_omnivores": agent.hunts_omnivores, "hunts_predators": agent.hunts_predators, "guarded": agent.guarded,
        "alarm": agent.alarm, "kill_alert": _optional_list(agent.kill_alert),
        "seen_success": float(agent.seen_success), "seen_failure": float(agent.seen_failure),
        "weathered": float(agent.weathered),
        "herd_assailant": agent.herd_assailant, "food_memory": _optional_list(agent.food_memory),
        "danger_memory": _optional_list(agent.danger_memory), "meat": float(agent.meat),
        "meat_fill": float(agent.meat_fill), "population": agent.population, "cost_share": float(agent.cost_share),
        "glut": float(agent.glut), "rot": float(agent.rot), "portions": agent.portions,
        "trail": [list(cell) for cell in agent.trail],
        "memories": [[numbers.senses(senses), int(choice), float(target), float(weight)]
                     for senses, choice, target, weight in agent.memories],
        "diary": [[numbers.senses(senses), int(choice), float(target), float(weight), float(vividness), int(turn)]
                  for (senses, choice, target, weight), vividness, turn in agent.diary],
        "womb": [[_genome(dna), _mind(mind), numbers.brain(brain), float(energy), int(generation), int(father)]
                 for dna, mind, brain, energy, generation, father in agent.womb],
        "due_turn": agent.due_turn, "mind": _mind(agent.mind), "last_senses": numbers.senses(agent.last_senses),
        "help_tried": ([agent.help_to, agent.help_deed, agent.help_kind, agent.help_turn, agent.help_goal_turn, int(agent.help_pending), numbers.senses(agent.help_senses)]
                       if agent.help_to >= 0 else None),
        "help_given": list(agent.help_given),
        "values": [[float(v) for v in agent.values], [float(v) for v in agent.taught], [float(v) for v in agent.liked_by_cue], agent.value_lessons, float(agent.liked)],
        "hawking": [agent.hawk_call_turn, agent.hawk_to, agent.hawk_deed if agent.hawk_to >= 0 else -1, agent.hawk_move_turn, agent.hawk_far,
                    agent.sold_turn, agent.bought_turn, agent.sold_taught_turn, agent.bought_taught_turn, float(agent.sold_gain), float(agent.bought_gain),
                    int(agent.sold_call is not None), agent.hawked_sales, agent.hawked_buys],
        "hawk_senses": numbers.senses(agent.hawk_senses) if agent.hawk_to >= 0 and agent.hawk_senses is not None else None,
        "sold_call": ([[int(v) for v in agent.sold_call[0]], [int(v) for v in agent.sold_call[1]]] if agent.sold_call is not None else None),
        "wooing": [agent.woo_call_turn, agent.woo_to, agent.woo_deed if agent.woo_to >= 0 else -1, agent.woo_move_turn, agent.woo_far,
                   agent.wooed_turn, agent.came_turn, agent.wooed_taught_turn, agent.came_taught_turn, int(agent.wooed_call is not None),
                   agent.matings, agent.wooed_count, agent.came_count, agent.wooed_far_total, [int(v) for v in agent.woo_calls]],
        "woo_senses": (numbers.senses(agent.woo_senses) if agent.woo_senses is not None
                       and (agent.woo_to >= 0 or agent.came_turn > agent.came_taught_turn) else None),
        "wooed_call": ([[int(v) for v in agent.wooed_call[0]], [int(v) for v in agent.wooed_call[1]]] if agent.wooed_call is not None else None),
        "last_choice": int(agent.last_choice), "last_forced": bool(agent.last_forced), "last_panicked": bool(agent.last_panicked), "last_wellbeing": float(agent.last_wellbeing),
        "reward": float(agent.reward), "last_comfort": float(agent.last_comfort),
        "odds": {action: float(chance) for action, chance in agent.odds.items()},
        "call": agent.call, "call_turn": agent.call_turn, "heard_from": agent.heard_from, "heard_call": agent.heard_call, "heard_turn": agent.heard_turn, "heard_hunger": float(agent.heard_hunger), "heard_danger": agent.heard_danger, "struck_turn": agent.struck_turn, "last_call": agent.last_call, "cues": list(agent.cues), "cry": agent.cry, "call_kind": agent.call_kind, "itch": float(agent.itch), "named": agent.named, "named_turn": agent.named_turn, "relieved": agent.relieved, "has_foresaw": agent.has_foresaw, "foresaw": [float(v) for v in agent.foresaw], "used_to": [float(v) for v in agent.used_to], "voiced_turn": agent.voiced_turn, "address_turn": agent.address_turn, "address_to": agent.address_to, "learn_turn": agent.learn_turn, "learn_from": agent.learn_from, "learn_correct": agent.learn_correct, "learn_gain": agent.learn_gain, "learn_call": agent.learn_call, "word_turn": agent.word_turn, "word_call": agent.word_call, "word_sounds": list(agent.word_sounds), "word_point": agent.word_point, "word_tries": agent.word_tries, "learn_thing": agent.learn_thing, "learn_sounds": list(agent.learn_sounds), "lexicon_spoke_turn": agent.lexicon_spoke_turn, "word_worth": [[float(v) for v in agent.nearer_comfort], [float(v) for v in agent.farther_comfort]], "word_heed": [agent.close_turn, agent.meant_turn, agent.meant_thing, float(agent.meant_close), agent.meant_deed, agent.judged_turn, int(agent.heeded), int(agent.acted_good), int(agent.acted_same), float(agent.word_reward), agent.judged_thing, agent.judged_deed, agent.unknown_turn, agent.unknown_deed, agent.unknown_judged_turn, int(agent.unknown_same), int(agent.meant_liked), int(agent.judged_liked)], "was_close": [float(v) for v in agent.was_close], "moved": list(agent.moved), "lexicon": [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in agent.lexicon], "names": [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in agent.names], "deeds": [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in agent.deeds], "influence": [float(agent.influence_given), agent.influence_turn, agent.influence_kind, agent.influence_sound, float(agent.influence_life), agent.influence_moved], "concepts": [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in agent.concepts], "talk": [agent.concept_spoke_turn, agent.concept_heard_turn, agent.replied_to, agent.replied_turn, agent.talk_calls, agent.idle_calls, agent.said_things, agent.said_deeds, agent.said_names, agent.said_concepts, agent.heard_concepts, agent.exchanges], "deed_heed": [agent.deed_spoke_turn, int(agent.meant_by_deed), int(agent.judged_by_deed), agent.echo_turn, agent.prior_choice], "second_word": [agent.call2, *agent.sounds2, agent.last_call2, *agent.last_sounds2, int(agent.meant_for_me), int(agent.meant_for_other), int(agent.judged_for_me)], "own_name": [agent.own_call, *agent.own_sounds], "name_heed": [agent.name_spoke_turn, agent.called_turn, agent.called_one, float(agent.called_close), agent.called_me_turn, agent.name_judged_turn, int(agent.name_heeded), float(agent.name_reward)], "word_one": agent.word_one, "chatter": [float(agent.chatter), agent.repeat_from, agent.repeat_call, *agent.repeat_sounds, agent.word_from, agent.answered_turn, agent.answered_count], "learn_one": agent.learn_one, "goal": goal_record(agent), "harm_taught_turn": agent.harm_taught_turn, "quiet_wish": float(agent.quiet_wish), "surprised": [float(v) for v in agent.surprised], "outside_surprise": float(agent.outside_surprise), "told_by": agent.told_by, "told_turn": agent.told_turn, "told_surprise": float(agent.told_surprise), "call_urgency": float(agent.call_urgency), "echo_of": [list(row) for row in agent.echo_of],
        "held": [float(held) for held in agent.held], "last_wrote": bool(agent.last_wrote),
        "past": [numbers.senses(senses) for senses in agent.past],
        "sounds": list(agent.sounds), "last_sounds": list(agent.last_sounds),
        **({"habits": [[base64.b64encode(habit_senses.tobytes()).decode("ascii"), *map(int, said)] for habit_senses, said in agent.habits],
            "habit_pull": float(agent.habit_pull)} if agent.habits else {}),
        "chosen": [float(agent.chosen_value), agent.chosen_turn, agent.call_heeded],
        **({"call_pleasure": float(agent.call_pleasure)} if agent.call_pleasure != 0 else {}),
        **({"call_values": [float(agent.call_whole), float(agent.call_part)]} if agent.call_whole != 0 or agent.call_part != 0 else {}),
        **({"held_call": {"senses": numbers.senses(agent.call_senses), "target": float(agent.held_call_target),
                          "outputs": [OUTPUTS[output] if output >= 0 else None for output in agent.call_outputs],
                          "outputs2": [OUTPUTS[output] if output >= 0 else None for output in agent.call2_outputs],
                          **({"intent": agent.held_intent} if agent.held_intent >= 0 else {}),
                          **({"point": agent.held_point} if agent.held_point >= 0 else {}),
                          **({"learn": agent.held_learn} if agent.held_learn != 0.0 else {})}}
           if agent.call_held and agent.call_senses is not None else {}),
        **({"intent": agent.intent} if agent.intent >= 0 else {}),
        **({"point": [agent.point, agent.point_turn, agent.prior_point, agent.prior_point_turn, agent.point_one, agent.prior_point_one]}
           if agent.point >= 0 or agent.prior_point >= 0 else {}),
        **({"pointed": [agent.pointed_by, agent.pointed_thing, agent.pointed_turn, float(agent.pointed_off)]}
           if agent.pointed_turn >= 0 else {}),
        **({"attend": [agent.attending, agent.attend_turn, agent.attend_choice]}
           if agent.attend_turn >= 0 or agent.attend_choice >= 0 else {}),
        # each [column, row, last, then for each season: seen, food, danger, comfort]
        "places": [[place.column, place.row, place.last,
                    *(value for quarter in range(4) for value in (place.seen[quarter], float(place.food[quarter]),
                                                                  float(place.danger[quarter]), float(place.comfort[quarter])))]
                   for place in agent.places],
        "factions": [[one.ring, float(one.regard), one.last, float(one.danger), float(one.skill)] for one in agent.factions],  # each [body word, regard, last]
        "known": [[one.identifier, float(one.regard), one.last, float(one.danger), float(one.skill), float(one.garb), one.liar] for one in agent.known],  # each [identifier, regard, last]
    }


def _agent_from(record: Mapping, numbers: _Reader, before_terrain: bool = False) -> Bitzkey:
    """A Bitzkey as saved; one saved before the terrain gene is terrestrial, as are the young it carries."""
    genome = (lambda text: convert_v3(int(text, 16))) if before_terrain else (lambda text: int(text, 16))  # noqa: E731
    agent = Bitzkey(int(record["identifier"]), genome(record["dna"]), int(record["x"]), int(record["y"]),
                    mind=_mind_from(record["mind"]))
    agent.stamina = float(record.get("stamina", world.FULL_STAMINA))  # fresh, in a world saved before stamina
    for name in ("hunger", "health", "rivalry", "own_rivalry", "seen_success", "seen_failure", "meat", "meat_fill",
                 "cost_share", "glut", "rot", "last_wellbeing", "reward", "last_comfort"):
        setattr(agent, name, float(record[name]))
    for name in ("age", "cooldown", "generation", "children", "population", "portions", "last_choice"):
        setattr(agent, name, int(record[name]))
    agent.fruit = int(record.get("fruit", -1))  # full, in an older world
    agent.picked_by = int(record.get("picked_by", -1))
    agent.weathered = float(record.get("weathered", 0.0))  # worlds saved before the weather's grace
    agent.last_forced = bool(record.get("last_forced", False))  # chosen, in a world saved before it was kept
    agent.last_panicked = bool(record.get("last_panicked", False))
    for name in ("panicked", "primal_fear", "hunts_grazers", "hunts_omnivores", "hunts_predators", "guarded"):
        setattr(agent, name, bool(record[name]))
    for name in ("died_turn", "assailant", "alarm", "herd_assailant", "due_turn"):
        setattr(agent, name, None if record[name] is None else int(record[name]))
    for name in ("kill_alert", "food_memory", "danger_memory"):
        setattr(agent, name, None if record[name] is None else tuple(int(value) for value in record[name]))
    agent.last_action = str(record["last_action"])
    agent.brain = numbers.brain(record["brain"], agent.dna)
    agent.trail = [(int(x), int(y)) for x, y in record["trail"]]
    agent.memories = [(numbers.senses(senses), numbers.action(int(choice)), float(target), float(weight))  # one of an action
                      for senses, choice, target, weight in record["memories"] if numbers.action(int(choice)) >= 0]  # no longer had is forgotten
    agent.diary = [[(numbers.senses(entry[0]), numbers.action(int(entry[1])), float(entry[2]), float(entry[3])),
                    float(entry[4]), int(entry[5])]
                   for entry in record.get("diary", []) if numbers.action(int(entry[1])) >= 0]
    agent.womb = [(genome(young[0]), _mind_from(young[1]), numbers.brain(young[2], genome(young[0])), float(young[3]),
                   int(young[4]), int(young[5]) if len(young) > 5 else -1)
                  for young in record["womb"]]
    agent.mother = int(record.get("mother", -1))  # unknown in a world saved before parents were kept
    agent.father = int(record.get("father", -1))
    agent.infant_taught = bool(record.get("infant_taught", False))
    agent.taught_by = int(record.get("taught_by", -1))  # none in a world saved before teaching
    agent.carried_meat = max(0, int(record.get("carried_meat", 0)))  # nothing in a world saved before carrying; none below nothing
    agent.carried_fruit = max(0, int(record.get("carried_fruit", 0)))
    agent.weapon = int(record.get("weapon", 0))
    agent.armour_worn = int(record.get("armour", 0))
    agent.weapon_uses = int(record.get("weapon_uses", 0))
    agent.bow_uses = int(record.get("bow_uses", 0))  # none, in a world saved before archery
    agent.bow = bool(record.get("bow", agent.bow_uses > 0))  # before empty bows were kept, a bow was its shots
    agent.archery = float(record.get("archery", 0.0))
    agent.wagon = int(record.get("wagon", -1))
    agent.emperor = bool(record.get("emperor", False))
    agent.rank = int(record.get("rank", 0))  # none till the next count, in a world saved before ranks
    agent.desire = [float(value) for value in record.get("desire", [0.0, 0.0, 0.0])]  # none, in a world saved before desire
    agent.tutored_turn = int(record.get("tutored_turn", -1))
    agent.armour_uses = int(record.get("armour_uses", 0))
    agent.crafts = int(record.get("crafts", 0))
    agent.craft_tries = (list(record.get("craft_tries", ())) + [0] * 7)[:7]  # five before the bow, six before the wagon
    agent.cooked_meat = max(0, int(record.get("cooked_meat", 0)))
    agent.cooked_fruit = max(0, int(record.get("cooked_fruit", 0)))
    agent.carried_stone = max(0, int(record.get("carried_stone", 0)))  # none below nothing, as an old bug left some
    agent.carried_bone = max(0, int(record.get("carried_bone", 0)))  # none below nothing, as an old bug left some
    agent.carried_wood = max(0, int(record.get("carried_wood", 0)))  # none below nothing, as an old bug left some
    agent.carried_seed = int(record.get("carried_seed", 0))
    agent.boredom = float(record.get("boredom", 0.0))  # none in an older world
    agent.shelter_known = float(record.get("shelter_known", 0.0))
    agent.road_known = float(record.get("road_known", 0.0))  # none in an older world
    agent.cooked_known = float(record.get("cooked_known", 0.0))
    agent.plenty = float(record.get("plenty", 0.0))
    agent.plenty_known = float(record.get("plenty_known", -1.0))
    agent.hide = int(record.get("hide", 0))
    agent.hide_armour = float(record.get("hide_armour", 0.0))
    agent.spare_tools = [[int(tool[0]), int(tool[1])] for tool in record.get("spare_tools", [])]  # none, in a world saved before
    agent.spare_hides = [[int(hide[0]), float(hide[1])] for hide in record.get("spare_hides", [])]
    agent.skinned = bool(record.get("skinned", False))
    agent.thefts = int(record.get("thefts", 0))
    agent.mark = int(record.get("mark", 0))  # none, in an older world
    agent.standout = int(record.get("standout", -1))  # nor these
    agent.follows = int(record.get("follows", -1))
    agent.followers = int(record.get("followers", 0))
    agent.leader = bool(record.get("leader", False))
    agent.realm = int(record.get("realm", -1))
    agent.trespass = int(record.get("trespass", 0))
    agent.meat_lean = float(record.get("meat_lean", 0.0))
    agent.expected = record.get("expected")  # nor this
    if len(record.get("last_tried", ())) == len(ACTIONS):
        agent.last_tried = {action: int(turn) for action, turn in zip(ACTIONS, record["last_tried"]) if turn}
    agent.last_senses = numbers.senses(record["last_senses"])
    agent.last_choice = numbers.action(agent.last_choice)
    if agent.last_choice < 0:  # an action no longer had: nothing to learn from it
        agent.last_choice, agent.last_senses = ACTIONS.index("rest"), None
    agent.call = int(record.get("call", 0))  # silent, in a world saved before calls
    agent.call_turn = int(record.get("call_turn", -1))
    agent.heard_from = int(record.get("heard_from", -1))  # none, in a world saved before calls paid
    agent.heard_call = int(record.get("heard_call", -1))
    agent.heard_turn = int(record.get("heard_turn", -1))
    agent.heard_hunger = float(record.get("heard_hunger", 0.0))
    agent.heard_danger = bool(record.get("heard_danger", False))
    agent.struck_turn = int(record.get("struck_turn", -1))
    agent.voiced_turn = int(record.get("voiced_turn", -1))
    agent.address_turn = int(record.get("address_turn", world.NEVER_ADDRESSED))
    agent.address_to = int(record.get("address_to", -1))
    agent.learn_turn = int(record.get("learn_turn", -1))
    agent.learn_from = int(record.get("learn_from", -1))
    agent.learn_correct = bool(record.get("learn_correct", False))
    agent.learn_gain = float(record.get("learn_gain", 0.0))
    agent.learn_call = int(record.get("learn_call", -1))
    agent.word_turn = int(record.get("word_turn", -1))
    agent.word_call = int(record.get("word_call", -1))
    agent.word_sounds = [int(sound) for sound in record.get("word_sounds", [-1, -1])]
    agent.word_point = int(record.get("word_point", -1))
    agent.word_tries = int(record.get("word_tries", 0))
    agent.learn_thing = int(record.get("learn_thing", -1))
    agent.learn_sounds = [int(s) for s in record.get("learn_sounds", [-1, -1])]
    agent.lexicon_spoke_turn = int(record.get("lexicon_spoke_turn", -1))
    if "word_worth" in record:  # see LEXICON_ACT
        agent.nearer_comfort = [float(v) for v in record["word_worth"][0]]
        agent.farther_comfort = [float(v) for v in record["word_worth"][1]]
    if "was_close" in record:
        agent.was_close = [float(v) for v in record["was_close"]]
    if "moved" in record:
        agent.moved = [int(v) for v in record["moved"]]
    if "word_heed" in record:
        h = record["word_heed"]
        agent.close_turn, agent.meant_turn, agent.meant_thing, agent.meant_close = int(h[0]), int(h[1]), int(h[2]), float(h[3])
        agent.meant_deed, agent.judged_turn = int(h[4]), int(h[5])
        agent.heeded, agent.acted_good, agent.acted_same, agent.word_reward = bool(h[6]), bool(h[7]), bool(h[8]), float(h[9])
        if len(h) > 11:
            agent.judged_thing, agent.judged_deed = int(h[10]), int(h[11])
        if len(h) > 15:
            agent.unknown_turn, agent.unknown_deed, agent.unknown_judged_turn, agent.unknown_same = int(h[12]), int(h[13]), int(h[14]), bool(h[15])
        if len(h) > 17:
            agent.meant_liked, agent.judged_liked = bool(h[16]), bool(h[17])
    agent.lexicon = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in record.get("lexicon", [])]
    agent.names = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in record.get("names", [])]
    agent.deeds = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in record.get("deeds", [])]
    agent.concepts = [[int(w[0]), int(w[1]), int(w[2]), int(w[3]), float(w[4]), int(w[5])] for w in record.get("concepts", [])]
    if record.get("influence") is not None:  # see INFLUENCE_JOY
        f = record["influence"]
        agent.influence_given, agent.influence_turn, agent.influence_kind = float(f[0]), int(f[1]), int(f[2])
        agent.influence_sound, agent.influence_life, agent.influence_moved = int(f[3]), float(f[4]), int(f[5])
    if record.get("talk") is not None:  # see CONVERSE_JOY
        (agent.concept_spoke_turn, agent.concept_heard_turn, agent.replied_to, agent.replied_turn, agent.talk_calls, agent.idle_calls,
         agent.said_things, agent.said_deeds, agent.said_names, agent.said_concepts, agent.heard_concepts, agent.exchanges) = [int(v) for v in record["talk"]]
    if "second_word" in record:  # see CALL_WORDS
        w = [int(v) for v in record["second_word"]]
        agent.call2, agent.sounds2, agent.last_call2, agent.last_sounds2 = w[0], w[1:3], w[3], w[4:6]
        agent.meant_for_me, agent.meant_for_other, agent.judged_for_me = bool(w[6]), bool(w[7]), bool(w[8])
    if "deed_heed" in record:
        d = record["deed_heed"]
        agent.deed_spoke_turn, agent.meant_by_deed, agent.judged_by_deed, agent.echo_turn = int(d[0]), bool(d[1]), bool(d[2]), int(d[3])
        if len(d) > 4:
            agent.prior_choice = max(0, numbers.action(int(d[4])))
    if "chatter" in record:  # see HOARSE and ANNOYANCE
        c = record["chatter"]
        agent.chatter, agent.repeat_from, agent.repeat_call, agent.repeat_sounds = float(c[0]), int(c[1]), int(c[2]), [int(c[3]), int(c[4])]
        if len(c) > 7:
            agent.word_from, agent.answered_turn, agent.answered_count = int(c[5]), int(c[6]), int(c[7])
    if "own_name" in record:
        agent.own_call, agent.own_sounds = int(record["own_name"][0]), [int(record["own_name"][1]), int(record["own_name"][2])]
    if "name_heed" in record:
        n = record["name_heed"]
        agent.name_spoke_turn, agent.called_turn, agent.called_one, agent.called_close = int(n[0]), int(n[1]), int(n[2]), float(n[3])
        agent.called_me_turn, agent.name_judged_turn, agent.name_heeded, agent.name_reward = int(n[4]), int(n[5]), bool(n[6]), float(n[7])
    if record.get("help_tried") is not None:  # help tried, to be judged (see HELPED_JOY)
        h = record["help_tried"]
        agent.help_to, agent.help_deed, agent.help_kind, agent.help_turn, agent.help_goal_turn = int(h[0]), numbers.action(int(h[1])), int(h[2]), int(h[3]), int(h[4])
        agent.help_pending, agent.help_senses = bool(h[5]), numbers.senses(h[6])
        if agent.help_deed < 0 or not 0 <= agent.help_kind < 4:
            agent.help_to = -1
    agent.help_given = [int(v) for v in record.get("help_given", [0, 0, 0, 0])]
    if record.get("values") is not None:  # see VALUE_KINDS
        v = record["values"]
        agent.values = [max(0.0, min(3.0, float(x))) for x in v[0]][:8] + [1.0] * max(0, 8 - len(v[0]))
        agent.taught = [max(-1.0, min(3.0, float(x))) for x in v[1]][:8] + [0.0] * max(0, 8 - len(v[1]))
        agent.liked_by_cue = [float(x) for x in v[2]][:5] + [0.0] * max(0, 5 - len(v[2]))
        agent.value_lessons, agent.liked = int(v[3]), float(v[4])
    if record.get("hawking") is not None:  # see HAWK_JOY
        h = record["hawking"]
        agent.hawk_call_turn, agent.hawk_to = int(h[0]), int(h[1])
        agent.hawk_deed = numbers.action(int(h[2])) if int(h[2]) >= 0 else -1
        agent.hawk_move_turn, agent.hawk_far = int(h[3]), int(h[4])
        agent.sold_turn, agent.bought_turn, agent.sold_taught_turn, agent.bought_taught_turn = int(h[5]), int(h[6]), int(h[7]), int(h[8])
        agent.sold_gain, agent.bought_gain = float(h[9]), float(h[10])
        agent.hawked_sales, agent.hawked_buys = int(h[12]), int(h[13])
        if record.get("hawk_senses") is not None:
            agent.hawk_senses = numbers.senses(record["hawk_senses"])
        if agent.hawk_to >= 0 and agent.hawk_deed < 0:
            agent.hawk_to = -1
        c = record.get("sold_call")
        agent.sold_call = (np.asarray(c[0], dtype=np.int8), [int(v) for v in c[1]]) if int(h[11]) and c is not None else None
    if record.get("wooing") is not None:  # see WOO_JOY
        w = record["wooing"]
        agent.woo_call_turn, agent.woo_to = int(w[0]), int(w[1])
        agent.woo_deed = numbers.action(int(w[2])) if int(w[2]) >= 0 else -1
        agent.woo_move_turn, agent.woo_far = int(w[3]), int(w[4])
        agent.wooed_turn, agent.came_turn, agent.wooed_taught_turn, agent.came_taught_turn = int(w[5]), int(w[6]), int(w[7]), int(w[8])
        agent.matings, agent.wooed_count, agent.came_count, agent.wooed_far_total = int(w[10]), int(w[11]), int(w[12]), int(w[13])
        agent.woo_calls = [int(v) for v in w[14]][:5] + [0] * max(0, 5 - len(w[14]))
        if record.get("woo_senses") is not None:
            agent.woo_senses = numbers.senses(record["woo_senses"])
        else:
            agent.came_taught_turn = max(agent.came_taught_turn, agent.came_turn)
        if agent.woo_deed < 0:
            agent.woo_to = -1
        c = record.get("wooed_call")
        agent.wooed_call = (np.asarray(c[0], dtype=np.int8), [int(v) for v in c[1]]) if int(w[9]) and c is not None else None
    agent.word_one = int(record.get("word_one", -1))
    agent.learn_one = int(record.get("learn_one", -1))
    if "goal" in record:  # see GOAL_JOY
        g = record["goal"]
        head = g["head"]
        agent.goal_kind, agent.goal_target, agent.goal_need, agent.goal_turn = int(head[0]), int(head[1]), int(head[2]), int(head[3])
        agent.goal_urgency, agent.goal_close = float(head[4]), float(head[5])
        (agent.goals_formed, agent.goals_reached, agent.goals_abandoned, agent.goals_switched, agent.goals_dropped,
         agent.goal_reach_turns) = (int(v) for v in head[6:12])
        agent.has_needs = bool(head[12])
        if agent.goal_kind == 3:  # a deed no longer had: no goal
            agent.goal_target = numbers.action(agent.goal_target)
            if agent.goal_target < 0:
                agent.goal_kind = 0
        if agent.goal_kind == 1 and not 0 <= agent.goal_target < 21:
            agent.goal_kind = 0
        if "desire" in g:  # see DESIRE
            d = g["desire"]
            top = d["head"]
            agent.good_drift, agent.joy_mark, agent.goal_desire = float(top[0]), float(top[1]), bool(top[2])
            agent.desires_formed, agent.desires_reached, agent.desire_reach_turns = int(top[3]), int(top[4]), int(top[5])
            for cue in range(min(5, len(top) - 6)):
                agent.desires_by_cue[cue] = int(top[6 + cue])
            for thing, want, spark, fade, *source in d["things"]:
                if 0 <= int(thing) < 21:
                    agent.want_things[int(thing)], agent.spark_things[int(thing)], agent.fade_things[int(thing)] = float(want), float(spark), float(fade)
                    agent.spark_from_things[int(thing)] = int(source[0]) if source else 0
            for deed, want, spark, fade, *source in d["deeds"]:
                deed = numbers.action(int(deed))
                if deed >= 0:
                    agent.want_deeds[deed], agent.spark_deeds[deed], agent.fade_deeds[deed] = float(want), float(spark), float(fade)
                    agent.spark_from_deeds[deed] = int(source[0]) if source else 0
            for thing, x, y, seen in d.get("places", []):
                if 0 <= int(thing) < 21:
                    agent.place_x[int(thing)], agent.place_y[int(thing)], agent.place_turn[int(thing)] = int(x), int(y), int(seen)
        if "vendetta" in g:  # see VENDETTA
            v = g["vendetta"]
            agent.enemies = [[int(e[0]), float(e[1]), float(e[2]), int(e[3]), int(e[4]), int(e[5]), bool(e[6])] for e in v["enemies"]][:4]
            h = [int(x) for x in v["head"]]
            agent.goal_vendetta, agent.humbled_until = bool(h[0]), h[1]
            (agent.vendettas_formed, agent.vendettas_settled, agent.vendettas_ended, agent.challenges_made, agent.challenges_won,
             agent.challenges_faced, agent.challenges_held) = h[2:9]
        if "project" in g:  # see PROJECT_PULL
            p = g["project"]
            agent.aim_value, agent.aim_last = [float(v) for v in p["values"]], [float(v) for v in p["last"]]
            if "agree" in p:
                agent.aim_agree = [float(v) for v in p["agree"]]
            if "asking" in p:
                (agent.project_stuck_turns, agent.project_stuck_calls, agent.project_unstuck_after_call, agent.project_unstuck_silent,
                 agent.instructed_by_sign, agent.instructed_by_word) = (int(v) for v in p["asking"])
            h = [int(v) for v in p["head"]]
            agent.aim_sampled = bool(h[0])
            (agent.project_aim, agent.project_dir, agent.project_turn, agent.project_progress_turn, agent.project_x, agent.project_y,
             agent.project_group, agent.project_concept, agent.project_from, agent.project_waited) = h[1:11]
            agent.project_confided, agent.goal_project, agent.goal_instructed = bool(h[11]), bool(h[12]), bool(h[13])
            agent.instructed_by = h[14]
            agent.shown_step = numbers.action(h[15]) if h[15] >= 0 else -1
            agent.shown_turn, agent.plan_open = h[16], bool(h[17])
            (agent.projects_taken, agent.projects_given_up, agent.project_progress_turns, agent.instructions_taken, agent.confided,
             agent.confided_to) = h[18:24]
            agent.aim_good, agent.aim_good_drift, agent.project_last = (float(v) for v in p["good"])
            for deed, value in p["deeds"]:
                deed = numbers.action(int(deed))
                if deed >= 0:
                    agent.project_deeds[deed] = float(value)
            for thing, value in p["things"]:
                if 0 <= int(thing) < 21:
                    agent.project_things[int(thing)] = float(value)
            if not 0 <= agent.project_aim < 15:
                agent.project_aim = -1
        if "asking" in g:  # see HELPED_JOY
            a = g["asking"]
            agent.stuck, agent.stuck_goal_turn, agent.help_call_turn = bool(a[0]), int(a[1]), int(a[2])
            (agent.stuck_turns, agent.free_turns, agent.stuck_calls, agent.free_calls, agent.unstuck_after_call, agent.unstuck_silent,
             agent.helped, agent.helping) = (int(v) for v in a[3:11])
            if len(a) > 12:
                agent.helped_turn, agent.help_taught_turn = int(a[11]), int(a[12])
        if "ingenuity" in g:  # see OPENS_RATE
            i = g["ingenuity"]
            top = i["head"]
            agent.step_turn, agent.chain_goal_turn = int(top[0]), int(top[1])
            agent.chains_begun, agent.steps_reached, agent.steps_given_up, agent.links_first_followed = (int(v) for v in top[3:7])
            agent.chains_unfinished = int(top[8]) if len(top) > 8 else 0
            agent.has_open = len(top) > 7 and bool(top[7])

            def node(saved: int) -> int:  # by the deed's saved number
                saved = int(saved)
                if saved >= 0:
                    return saved if saved < 21 else -1
                deed = numbers.action(-1 - saved)
                return -1 if deed < 0 else 21 + deed
            agent.failed_links = [[node(a), node(b)] for a, b in i.get("failed", []) if node(a) >= 0 and node(b) >= 0]
            for n in i.get("open", []):
                if node(n) >= 0:
                    agent.was_open[node(n)] = True
            for source, to, strength, used in i["links"]:
                if node(source) >= 0 and node(to) >= 0 and len(agent.opens) < 24:
                    agent.opens.append([node(source), node(to), float(strength), bool(used)])
            steps = [node(step) for step in i["steps"]]
            agent.goal_steps = [] if any(step < 0 for step in steps) else steps
        if not agent.goal_desire and not agent.goal_project and not agent.goal_instructed and not agent.goal_vendetta and agent.goal_kind != 0 and not 0 <= agent.goal_need < 6:
            agent.goal_kind = 0
        if agent.goal_kind == 0:
            agent.goal_target = agent.goal_need = -1
            agent.goal_desire = False
        needs = g["needs"]
        agent.needs = [float(needs[2 * n]) for n in range(6)]
        agent.need_drift = [float(needs[2 * n + 1]) for n in range(6)]
        for n, thing, value in g["things"]:
            if 0 <= int(n) < 6 and 0 <= int(thing) < 21:
                agent.relief_things[int(n)][int(thing)] = float(value)
        for n, deed, value in g["deeds"]:
            deed = numbers.action(int(deed))
            if 0 <= int(n) < 6 and deed >= 0:
                agent.relief_deeds[int(n)][deed] = float(value)
        agent.relief_ones = [[int(row[0]), float(row[1]), [float(v) for v in row[2:8]],
                              *((float(row[8]), float(row[9]), float(row[10])) if len(row) >= 11 else (0.0, 0.0, 0.0)),
                              *((int(row[11]), int(row[12]), int(row[13])) if len(row) >= 14 else (-1, -1, -1)),
                              int(row[14]) if len(row) >= 15 else 0] for row in g["ones"]]
    agent.harm_taught_turn = int(record.get("harm_taught_turn", -1))
    agent.quiet_wish = float(record.get("quiet_wish", 0.0))
    agent.last_call = int(record.get("last_call", -1))
    for action, cue in enumerate(record.get("cues", [])[:len(agent.cues)]):
        agent.cues[action] = int(cue)
    agent.cry = int(record.get("cry", -1))
    agent.call_kind = int(record.get("call_kind", 0))
    agent.itch = float(record.get("itch", 0.0))
    agent.named = int(record.get("named", -1))
    agent.named_turn = int(record.get("named_turn", -1))
    agent.relieved = bool(record.get("relieved", False))
    if agent.named >= len(OUTSIDE_SENSES):  # before 3.212, an action told of
        agent.named = -1
    agent.has_foresaw = bool(record.get("has_foresaw", False))
    if "foresaw" in record:
        agent.foresaw = [float(v) for v in record["foresaw"]]
    if "used_to" in record:
        agent.used_to = [float(v) for v in record["used_to"]]
    if "surprised" in record:
        agent.surprised = [float(v) for v in record["surprised"]]
    agent.outside_surprise = float(record.get("outside_surprise", 0.0))
    agent.told_by = int(record.get("told_by", -1))
    agent.told_turn = int(record.get("told_turn", -1))
    agent.told_surprise = float(record.get("told_surprise", 0.0))
    agent.call_urgency = float(record.get("call_urgency", 0.0))
    agent.intent = int(record.get("intent", -1))
    if "point" in record:  # what it pointed at as it last called, and the call before, with their turns
        point = [int(value) for value in record["point"]]
        agent.point, agent.point_turn, agent.prior_point, agent.prior_point_turn = point[:4]
        if len(point) > 5:
            agent.point_one, agent.prior_point_one = point[4], point[5]
    if "pointed" in record:  # the finger it last read, and how surprised by that thing it then was
        by, thing, turn, off = record["pointed"]
        agent.pointed_by, agent.pointed_thing, agent.pointed_turn, agent.pointed_off = int(by), int(thing), int(turn), float(off)
    if "attend" in record:  # whom it attends to and since when, and whether it chose to, to be taught
        agent.attending, agent.attend_turn, agent.attend_choice = (int(value) for value in record["attend"])
    if "echo_of" in record:
        agent.echo_of = [[int(sound) for sound in row] for row in record["echo_of"]]
    agent.sounds = [int(sound) for sound in record.get("sounds", [-1, -1])]  # one sound, before calls ran on
    agent.last_sounds = [int(sound) for sound in record.get("last_sounds", [-1, -1])]
    if "habits" in record and numbers.sense_from is None:  # kept only while its senses are the same
        for entry in record["habits"]:
            habit_senses = np.frombuffer(base64.b64decode(entry[0]), dtype=np.int8).copy()
            if habit_senses.size == INPUT_SIZE:
                agent.habits.append((habit_senses, tuple(int(place) for place in entry[1:4])))
        agent.habit_pull = float(record.get("habit_pull", 0.0))
    agent.call_pleasure = float(record.get("call_pleasure", 0.0))
    if "call_values" in record:  # what its last call was taught, whole and its own part (see CALL_BASELINE)
        agent.call_whole, agent.call_part = float(record["call_values"][0]), float(record["call_values"][1])
    if "chosen" in record:
        agent.chosen_value, agent.chosen_turn, agent.call_heeded = float(record["chosen"][0]), int(record["chosen"][1]), int(record["chosen"][2])
    if "held_call" in record:  # its call's lesson, held till its listeners' turn is judged
        held_call = record["held_call"]
        agent.call_senses = numbers.senses(held_call["senses"])
        agent.held_call_target = float(held_call["target"])
        agent.call_outputs = [OUTPUTS.index(name) if name in OUTPUTS else -1 for name in held_call["outputs"]]
        agent.call2_outputs = [OUTPUTS.index(name) if name in OUTPUTS else -1 for name in held_call.get("outputs2", [None] * 3)]
        agent.held_intent = int(held_call.get("intent", -1))
        agent.held_point = int(held_call.get("point", -1))
        agent.held_learn = float(held_call.get("learn", 0.0))
        agent.call_held = True
    if "held" in record:  # blank, in a world saved before memory
        saved_held = [float(held) for held in record["held"]][:MEMORY_NEURONS]  # 8 in a world saved before 1.148
        agent.held = saved_held + [0.0] * (MEMORY_NEURONS - len(saved_held))
        agent.last_wrote = bool(record["last_wrote"])
        agent.past = [numbers.senses(senses) for senses in record["past"]]
    agent.places = [world.MapPlace(int(entry[0]), int(entry[1]), int(entry[2]),  # none, in a world saved before the map
                                   [bool(entry[3 + 4 * quarter]) for quarter in range(4)],
                                   [float(entry[4 + 4 * quarter]) for quarter in range(4)],
                                   [float(entry[5 + 4 * quarter]) for quarter in range(4)],
                                   [float(entry[6 + 4 * quarter]) for quarter in range(4)])
                    for entry in record.get("places", [])]
    agent.factions = [world.Faction(int(entry[0]), float(entry[1]), int(entry[2]), *map(float, entry[3:5]))  # none, in a world saved before
                      for entry in record.get("factions", [])]
    agent.known = [world.Acquaintance(int(entry[0]), float(entry[1]), int(entry[2]), *map(float, entry[3:6]), *map(bool, entry[6:7]))  # none, in a world saved before
                   for entry in record.get("known", [])]
    agent.odds = {str(action): float(chance) for action, chance in record["odds"].items() if action in ACTIONS}
    return agent


def world_bytes(simulation: Simulation) -> bytes:
    """The whole world, as a world file holds it."""
    numbers = _Numbers()
    ints: list[np.ndarray] = []  # the census, the lineage and the deaths
    held = 0

    def whole(values) -> int:
        nonlocal held
        array = np.ascontiguousarray(values, dtype="<i4").ravel()
        ints.append(array)
        at, held = held, held + array.size
        return at

    snapshots = [[turn, whole(alive), len(alive)] for turn, alive in simulation.census.snapshots]
    first = simulation.lineage_first
    lineage = [simulation.lineage.get(identifier, 0)
               for identifier in range(first, max(simulation.lineage, default=first - 1) + 1)]
    lineage_at = whole(np.array([[dna >> 64 & WORD, dna >> 32 & WORD, dna & WORD] for dna in lineage],
                                dtype="<u4").reshape(-1, 3).view("<i4"))
    strings: dict[str, int] = {}  # the causes and doings, each once
    code = lambda text: strings.setdefault(text, len(strings))  # noqa: E731
    deaths = sorted(simulation.death_records.items())
    rows = [[identifier, code(death.cause), SPECIES.index(death.species), death.turn,
             -1 if death.killer is None else death.killer,
             -1 if death.killer_species is None else SPECIES.index(death.killer_species),
             -1 if death.activity is None else code(death.activity)] for identifier, death in deaths]
    deaths_at = whole(np.array(rows, dtype="<i4").reshape(-1, 7))
    version, internal, gauss_next = simulation.random.getstate()
    sea: list[int] = []  # the sea's runs, row by row: land, then sea, then land, and so on
    if simulation.has_sea:
        under, run = 0, 0
        for square in simulation.sea_map:
            if square != under:
                sea.append(run)
                under, run = square, 0
            run += 1
        sea.append(run)
    record = {
        "format": WORLD_FORMAT, "genome_version": GENOME_VERSION, "senses": list(SENSES), "centred_senses": True, "value_scale": VALUE_SCALE, "actions": list(OUTPUTS),
        "temperament": list(TEMPERAMENT),
        "settings": {"width": simulation.width, "height": simulation.height,
                     "population_cap": simulation.population_cap, "mind_mutation": simulation.mind_mutation,
                     "body_mutation": simulation.body_mutation, "year_length": simulation.year_length,
                     "initial_grass": simulation._initial_grass, "ocean": simulation.ocean},
        "sea": sea,
        "rules": Rules.current(),
        "turn": simulation.turn, "deaths": simulation.deaths, "next_identifier": simulation.next_identifier,
        "death_causes": {species: dict(causes) for species, causes in simulation.death_causes.items()},
        "death_tally": _tally_record(simulation.death_tally),
        "random": {"mt": list(internal[:624]), "index": internal[624], "gauss_next": gauss_next},
        "grass": [simulation.grass.get((x, y), 0) for y in range(simulation.height) for x in range(simulation.width)],
        "materials": [[x, y, kind + 2, piles[(x, y)], simulation.pile_turns.get((x, y, ("bone", "wood")[kind - 1]), 0) if kind else 0] for y in range(simulation.height) for x in range(simulation.width)
                      for kind, piles in enumerate((simulation.stone_piles, simulation.bone_piles, simulation.wood_piles))
                      if (x, y) in piles]
                     + [[x, y, BOW_PILE if name == "bow" else kind, count, simulation.gear_wear.get((x, y, name), 0)] for kind, name in enumerate(GEAR_KINDS, 5)
                        for (x, y), count in sorted(simulation.gear_piles[name].items())]
                     + [[x, y, kind, count, 0] for kind, piles in ((10, simulation.cooked_meat_piles), (11, simulation.cooked_fruit_piles))
                        for (x, y), count in sorted(piles.items())],
        "arable": sorted(y * simulation.width + x for x, y in simulation.arable),
        "tended": sorted([y * simulation.width + x, simulation.tended.get((x, y), 0)] for x, y in simulation.arable),
        "works": sorted([y * simulation.width + x, *work] for (x, y), work in simulation.works.items()),  # cell, craft, stone, bone, wood, hide, last worked
        "fertilized": sorted(y * simulation.width + x for x, y in simulation.fertilized),
        "fights": [list(fight) for fight in simulation.fights],
        "lands": {"patch": simulation.land_patch, "columns": simulation.land_columns, "squares": len(simulation.lands), "next_realm": simulation.next_realm,
                  "realms": [[identifier, SPECIES.index(realm.species), realm.ring, realm.members] for identifier, realm in sorted(simulation.realms.items())],
                  "claims": [[patch, realm, hold] for patch, land in enumerate(simulation.lands) for realm, hold in land if realm >= 0],
                  "wars": [list(war) for war in simulation.wars], "welcomes": [list(pair) for pair in sorted(simulation.welcome)]},
        "roads": [[y * simulation.width + x, simulation.road_work.get((x, y), 0), int((x, y) in simulation.roads)] for x, y in sorted(simulation.roads | set(simulation.road_work), key=lambda c: (c[1], c[0]))],
        "wagons": [[wagon[0], wagon[1], wagon[2], list(wagon[3])] for wagon in simulation.wagons],  # x, y, its hitcher or -1, what it holds
        "shelters": [[x, y, simulation.sites.get((x, y), 0), int((x, y) in simulation.shelters), simulation.site_turns.get((x, y), 0), simulation.site_stone.get((x, y), 0)] for y in range(simulation.height) for x in range(simulation.width)
                     if (x, y) in simulation.sites or (x, y) in simulation.shelters],
        "fires": [[x, y, until, lighter, simulation.cooked_on.get((x, y), -world.DAY_LENGTH - 1)] for (x, y), (until, lighter) in sorted(simulation.fires.items(), key=lambda fire: (fire[0][1], fire[0][0])) if until > simulation.turn],
        "stones_scattered": simulation.stones_scattered,
        "piles": [[x, y, simulation.meat_piles.get((x, y), 0), simulation.fruit_piles.get((x, y), 0),
                   simulation.pile_turns.get((x, y, True), 0) if (x, y) in simulation.meat_piles else 0,
                   simulation.pile_turns.get((x, y, False), 0) if (x, y) in simulation.fruit_piles else 0]
                  for y in range(simulation.height) for x in range(simulation.width)
                  if (x, y) in simulation.meat_piles or (x, y) in simulation.fruit_piles],
        "seeds": [[x, y, _genome(dna), generation, sown] + ([simulation.sowers[(x, y, sown)]] if (x, y, sown) in simulation.sowers else [])
                  for x, y, dna, generation, sown in simulation.seeds],
        "insights": [[*key, offered, simulation.insights.chosen.get(key, 0.0)]
                     for key, offered in simulation.insights.offered.items()]
                    + [[*key, 0.0, chosen] for key, chosen in simulation.insights.chosen.items()
                       if key not in simulation.insights.offered],
        "census": {"stride": simulation.census.stride, "snapshots": snapshots},
        "lineage": {"at": lineage_at, "count": len(lineage), "first": first},
        "death_strings": list(strings), "death_records": {"at": deaths_at, "count": len(rows)},
        "agents": [_agent_record(agent, numbers) for agent in simulation.agents],
    }
    text = json.dumps(record, separators=(",", ":")).encode("utf-8")
    whole_part = b"".join(array.tobytes() for array in ints)
    return (WORLD_MAGIC + struct.pack("<Q", len(text)) + text + struct.pack("<Q", numbers.count) + numbers.data()
            + struct.pack("<Q", held) + whole_part)


def world_from_bytes(data: bytes) -> Simulation:
    """The world a world file holds, ready to carry on as it would have; its rules become the rules, all but
    those saying how this computer runs it. Raises ValueError, naming the problem, if it can't be read."""
    first_format = data.startswith(WORLD_MAGIC_1)
    if not first_format and not data.startswith(WORLD_MAGIC):
        raise ValueError("This isn't a Bitzkey world file.")
    at = len(WORLD_MAGIC)

    def counted(width: int) -> int:  # a count, then the room for that many values `width` bytes each
        nonlocal at
        if len(data) < at + 8:
            raise ValueError("The world file is cut short.")
        (count,) = struct.unpack("<Q", data[at:at + 8])
        at += 8
        if count > (len(data) - at) // width:
            raise ValueError("The world file is cut short.")
        return count

    length = counted(1)
    text, at = data[at:at + length], at + length
    if first_format:  # the numbers run to the end
        reader = _Reader(data[at:])
    else:
        count = counted(8)
        numbers, at = data[at:at + 8 * count], at + 8 * count
        count = counted(4)
        reader = _Reader(numbers, data[at:at + 4 * count])
    try:
        record = json.loads(text.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"The world file is damaged ({error}).") from None
    try:
        return _world_from(record, reader, 1 if first_format else WORLD_FORMAT)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"The world file is incomplete ({error}).") from None


def _world_from(record: Mapping, numbers: _Reader, format_: int) -> Simulation:
    if record.get("format") != format_:
        raise ValueError(f"This world file is format {record.get('format')}; this version reads format {WORLD_FORMAT}.")
    if not isinstance(record["actions"], list) or record["temperament"] != list(TEMPERAMENT):
        raise ValueError("This world was saved by a version of Bitzkey whose brains sense or act differently.")
    # Brains' outputs saved otherwise are read by name, as their senses are; so are the actions chosen.
    if record["actions"] != list(OUTPUTS):
        saved_at = {name: index for index, name in enumerate(record["actions"])}
        numbers.saved_outputs = len(record["actions"])
        numbers.output_from = [saved_at.get(name, -1) for name in OUTPUTS]
        if "teach" not in saved_at:  # a brain from before teaching values it as it does feeding its young, or it would never try it
            numbers.output_from[OUTPUTS.index("teach")] = saved_at.get("feed_young", -1)
        if "eat_pile" not in saved_at:  # so too with carrying; and eating from a pile as it does scavenging
            for name in ("take_meat", "take_fruit", "drop_meat", "drop_fruit"):
                numbers.output_from[OUTPUTS.index(name)] = saved_at.get("feed_young", -1)
            numbers.output_from[OUTPUTS.index("eat_pile")] = saved_at.get("scavenge", -1)
        if "eat_carried" not in saved_at:
            numbers.output_from[OUTPUTS.index("eat_carried")] = saved_at.get("scavenge", -1)
        numbers.action_now = [ACTIONS.index(name) if name in ACTIONS else -1 for name in record["actions"]]
    numbers.uncentred = not record.get("centred_senses", False)
    numbers.unscaled = "value_scale" not in record
    # Brains saved with other senses are read by name: a sense they lacked starts with no weight, and one no longer
    # sensed is let go.
    if record["senses"] != list(SENSES):
        saved_at = {name: index for index, name in enumerate(record["senses"])}
        numbers.saved_inputs = len(record["senses"])
        numbers.sense_from = [saved_at.get(name, -1) for name in SENSES]
    rules = {key: value for key, value in record["rules"].items() if key in Rules.BY_KEY}
    rules.update({key: Rules.get(key) for key in MACHINE_RULES})
    Rules.apply(rules)
    settings = record["settings"]
    simulation = Simulation(settings["width"], settings["height"], population=0, seed=0, grass_density=0,
                            population_cap=settings["population_cap"], mind_mutation=100 * settings["mind_mutation"],
                            body_mutation=100 * settings["body_mutation"], year_length=settings["year_length"])
    simulation.mind_mutation, simulation.body_mutation = settings["mind_mutation"], settings["body_mutation"]
    simulation._initial_grass = settings["initial_grass"]
    simulation.grass_regrowth = simulation._initial_grass * world.GRASS_REGROWTH
    simulation.ocean = settings.get("ocean", 0.0)  # none in a world saved before the sea
    if record.get("sea"):
        squares = bytearray()
        for index, run in enumerate(record["sea"]):
            squares.extend((index % 2,) * run)
        if len(squares) != simulation.width * simulation.height:
            raise ValueError("The world file's sea doesn't fit its map.")
        simulation.sea_map = squares
    # Genomes saved before the terrain gene had nothing but spare bits there: they are terrestrial.
    before_terrain = record.get("genome_version", 3) < 4
    simulation.turn = record["turn"]
    simulation.deaths = record["deaths"]
    simulation.next_identifier = record["next_identifier"]
    simulation.death_causes = {species: dict(causes) for species, causes in record["death_causes"].items()}
    state = record["random"]
    simulation.random.setstate((3, (*state["mt"], state["index"]), state["gauss_next"]))
    width = simulation.width
    simulation.grass = {(index % width, index // width): tufts for index, tufts in enumerate(record["grass"]) if tufts}
    piles = record.get("piles", ())  # none in a world saved before piles
    simulation.meat_piles = {(pile[0], pile[1]): pile[2] for pile in piles if pile[2] > 0}  # none below nothing, as an old bug left some
    simulation.fruit_piles = {(pile[0], pile[1]): pile[3] for pile in piles if pile[3] > 0}
    for pile in piles:  # dropped now, in a world saved before they spoiled
        simulation.pile_turns[(pile[0], pile[1], True)] = pile[4] if len(pile) > 4 else record["turn"]
        simulation.pile_turns[(pile[0], pile[1], False)] = pile[5] if len(pile) > 4 else record["turn"]
    simulation.stone_piles, simulation.bone_piles, simulation.wood_piles = {}, {}, {}  # the blank world's own stones gone
    for x, y, kind, count, dropped in record.get("materials", ()):
        if kind == BOW_PILE:
            simulation.gear_piles["bow"][(x, y)] = count
            simulation.gear_wear[(x, y, "bow")] = dropped
            continue
        if kind >= 10:
            (simulation.cooked_meat_piles if kind == 10 else simulation.cooked_fruit_piles)[(x, y)] = count
            continue
        if kind >= 5:
            simulation.gear_piles[GEAR_KINDS[kind - 5]][(x, y)] = count
            simulation.gear_wear[(x, y, GEAR_KINDS[kind - 5])] = dropped
            continue
        (simulation.stone_piles, simulation.bone_piles, simulation.wood_piles)[kind - 2][(x, y)] = count
        if kind >= 3:
            simulation.pile_turns[(x, y, "bone" if kind == 3 else "wood")] = dropped
    simulation.arable = {(cell % simulation.width, cell // simulation.width) for cell in record.get("arable", ())}  # none in an older world
    simulation.tended = {square: simulation.turn for square in simulation.arable}  # in an older world, tended as it is loaded
    simulation.tended.update(((cell % simulation.width, cell // simulation.width), turn) for cell, turn in record.get("tended", ()))
    simulation.works = {(work[0] % simulation.width, work[0] // simulation.width): [int(value) for value in work[1:7]] for work in record.get("works", ())}  # none in an older world
    simulation.fights = [[int(a), int(b), int(c), bool(d)] for a, b, c, d in record.get("fights", ())]  # none in an older world
    simulation.realms, simulation.lands, simulation.wars, simulation.welcome = {}, [], [], set()
    if "lands" in record:  # none, in a world saved before lands: counted afresh
        lands = record["lands"]
        simulation.land_patch, simulation.land_columns, simulation.next_realm = int(lands["patch"]), int(lands["columns"]), int(lands["next_realm"])
        simulation.lands = [[[-1, 0.0], [-1, 0.0]] for _ in range(int(lands["squares"]))]
        for identifier, species, ring, members in lands["realms"]:
            simulation.realms[int(identifier)] = world.Realm(int(identifier), SPECIES[int(species)], int(ring), int(members))
        for patch, realm, hold in lands["claims"]:
            land = simulation.lands[int(patch)]
            land[0 if land[0][0] < 0 else 1] = [int(realm), float(hold)]
        simulation.wars = [[int(value) for value in war] for war in lands["wars"]]
        simulation.welcome = {(int(owner), int(guest)) for owner, guest in lands.get("welcomes", ())}  # none before
        simulation._index_realm_patches()
    simulation.fertilized = {(cell % simulation.width, cell // simulation.width) for cell in record.get("fertilized", ())}  # none in an older world
    simulation.roads, simulation.road_work = set(), {}
    for cell, work, standing in record.get("roads", ()):  # none in an older world
        if work:
            simulation.road_work[(cell % simulation.width, cell // simulation.width)] = work
        if standing:
            simulation.roads.add((cell % simulation.width, cell // simulation.width))
    for x, y, units, standing, *built in record.get("shelters", ()):
        if units:
            simulation.sites[(x, y)] = units
            simulation.site_turns[(x, y)] = built[0] if built else record["turn"]  # built now, in a world saved before they fell
            simulation.site_stone[(x, y)] = built[1] if len(built) > 1 else 0  # all wood, in a world saved before
        if standing:
            simulation.shelters.add((x, y))
    for x, y, hitched_to, goods in record.get("wagons", ()):  # none in an older world
        simulation.wagons.append([x, y, hitched_to, list(goods)])
        if hitched_to < 0:
            simulation.parked.setdefault((x, y), []).append(len(simulation.wagons) - 1)
    simulation.fires = {(fire[0], fire[1]): (fire[2], fire[3]) for fire in record.get("fires", ())}
    simulation.cooked_on = {(fire[0], fire[1]): fire[4] for fire in record.get("fires", ()) if len(fire) > 4}  # none in an older world
    simulation.stones_scattered = record.get("stones_scattered", False)
    simulation.scatter_stones()  # a world saved before stones has them now, the once
    simulation.seeds = [(x, y, convert_v3(int(dna, 16)) if before_terrain else int(dna, 16), generation, sown)
                        for x, y, dna, generation, sown, *_ in record["seeds"]]
    simulation.sowers = {(seed[0], seed[1], seed[4]): seed[5] for seed in record["seeds"] if len(seed) > 5}  # none in an older world
    simulation.insights.offered = {}
    simulation.insights.chosen = {}
    for species, trait, level, action, offered, chosen in record["insights"]:
        if action not in ACTIONS:  # no longer an action
            continue
        key = (species, trait, level, action)
        if offered:
            simulation.insights.offered[key] = offered
        if chosen:
            simulation.insights.chosen[key] = chosen
    simulation.census.stride = record["census"]["stride"]
    if format_ == 1:
        simulation.census.snapshots = [(turn, tuple(alive)) for turn, alive in record["census"]["snapshots"]]
        simulation.lineage = {identifier: int(dna, 16) for identifier, dna in enumerate(record["lineage"])}
        simulation.death_records = {identifier: Death(*death) for identifier, *death in record["death_records"]}
    else:
        simulation.census.snapshots = [(turn, tuple(numbers.whole(first, count).tolist()))
                                       for turn, first, count in record["census"]["snapshots"]]
        lineage = record["lineage"]
        words = numbers.whole(lineage["at"], 3 * lineage["count"]).view("<u4").reshape(-1, 3).tolist()
        simulation.lineage_first = lineage.get("first", 0)
        simulation.lineage = {identifier: social << 64 | ring << 32 | dot
                              for identifier, (social, ring, dot) in enumerate(words, start=simulation.lineage_first)}
        strings, deaths = record["death_strings"], record["death_records"]
        species = lambda index: None if index < 0 else SPECIES[index]  # noqa: E731
        simulation.death_records = {
            identifier: Death(strings[cause], SPECIES[kind], turn, None if killer < 0 else killer, species(killer_kind),
                              None if doing < 0 else strings[doing])
            for identifier, cause, kind, turn, killer, killer_kind, doing
            in numbers.whole(deaths["at"], 7 * deaths["count"]).reshape(-1, 7).tolist()}
    if before_terrain:
        simulation.lineage = {identifier: convert_v3(dna) for identifier, dna in simulation.lineage.items()}
    simulation._order_deaths()
    untallied = "death_tally" not in record  # saved before deaths were tallied
    if untallied:  # tallied from those it kept whole
        simulation.death_tally = DeathTally()
        for _, identifier in simulation._death_order:
            simulation.death_tally.add(simulation.death_records[identifier])
    else:
        simulation.death_tally = _tally_from(record["death_tally"])
    simulation.agents = [_agent_from(agent, numbers, before_terrain) for agent in record["agents"]]
    simulation._leaders = sum(agent.leader and agent.alive for agent in simulation.agents)
    simulation._update_climate()
    simulation._index_cells()
    # What each knew as of its turn's start, which indexing the cells afresh would count again as things now stand.
    for agent, saved in zip(simulation.agents, record["agents"]):
        agent.population, agent.cost_share = int(saved["population"]), float(saved["cost_share"])
        agent.glut, agent.guarded = float(saved["glut"]), bool(saved["guarded"])
    if untallied:  # and then, as now, only the latest deaths kept whole, and only the killers still alive and the deadliest
        simulation._forget_old_deaths()
        simulation.death_tally.let_go(simulation._alive)
    return simulation


def _tally_record(tally: DeathTally) -> dict:
    counts = lambda rows: {label: [row.get(species, 0) for species in SPECIES] for label, row in sorted(rows.items())}  # noqa: E731
    return {"causes": counts(tally.causes), "whom": [[tally.whom.get(dead, {}).get(killer, 0) for killer in SPECIES] for dead in SPECIES],
            "activities": counts(tally.activities), "killers": tally.killers,
            "by_killer": [[identifier, SPECIES.index(kind), kills, dict(sorted(doing.items()))]
                          for identifier, (kind, kills, doing) in sorted(tally.by_killer.items())],
            "ages": [[band.get(species, 0) for species in SPECIES] for band in tally.ages]}


def _tally_from(record: Mapping) -> DeathTally:
    tally = DeathTally()
    rows = lambda table: {label: {species: count for species, count in zip(SPECIES, counts) if count}  # noqa: E731
                          for label, counts in table.items()}
    tally.causes, tally.activities = rows(record["causes"]), rows(record["activities"])
    tally.whom = {dead: {killer: count for killer, count in zip(SPECIES, counts) if count}
                  for dead, counts in zip(SPECIES, record["whom"]) if any(counts)}
    tally.killers = int(record["killers"])
    tally.by_killer = {identifier: [SPECIES[kind], kills, dict(doing)] for identifier, kind, kills, doing in record["by_killer"]}
    if "ages" in record:  # counted since
        tally.ages = [{species: count for species, count in zip(SPECIES, counts) if count} for counts in record["ages"]]
    return tally


def save_world(simulation: Simulation, path: str | Path) -> None:
    Path(path).write_bytes(world_bytes(simulation))


def load_world(path: str | Path) -> Simulation:
    return world_from_bytes(Path(path).read_bytes())
