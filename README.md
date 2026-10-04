# Bitzkey

Bitzkey is a small artificial-ecology experiment. Each Bitzkey carries a 32-bit
CMYK genome, shown as its eight-digit hexadecimal code and converted to RGB for
display. The world advances in turns on a grid (100x100 by default).

## Run

Install the dependency and start a world:

```powershell
python -m pip install -r requirements.txt
python Main.py
```

Choose a different world size, population, or repeatable random seed at launch:

```powershell
python Main.py --width 80 --height 60 --population 120 --seed 42
```

## Controls

- `Space`: pause or resume
- `Left Arrow`: rewind one recorded turn
- `Right Arrow`: advance one turn while paused, or replay a recorded turn
- `R`: restart with the same launch settings
- `I`: toggle the behaviour insights panel
- `B`: save a brain report to `reports/`
- Click a Bitzkey: inspect its genome, current state and the odds of its last choice
- `Escape`: quit

History retains up to 500 world snapshots. Rewinding and stepping forward replays
the recorded future exactly; advancing from an earlier point replaces that
future with a new branch.

## Genome, From Least-Significant Bits

Each successive four-bit word encodes metabolism, aggression, constitution,
species, fertility, sight, social behavior, and learning. The four species are
plant, grazer, omnivore, and predator. The four bytes of the genome, from most
to least significant, are cyan, magenta, yellow, and black; together they
determine the displayed color. The inspector shows the raw genome, its binary
words, decoded traits, hunger, health, age, generation, children, last action,
and the odds its brain gave each option.

Plants stay in place and photosynthesize. Every animal carries its own small
neural network, and that brain chooses everything it does. Each turn it senses
its hunger, health, food underfoot or in sight, a willing mate, nearby prey,
hunters and kin, and its age. It then picks one of the actions open to it:
eat, forage, rest, breed, attack, flee or explore. Better-valued actions are
more likely, but no choice is certain.

The genes set what each animal wants, not what it does:

- metabolism: how fast hunger rises and how much it hurts
- fertility: the pleasure of breeding
- aggression: the pleasure of a fight, and more of it for a kill
- social: the pleasure of company from its own species
- learning: how fast its brain learns; at 0 it acts only on instinct

Death is heavily punished. The brain learns online, from how each choice turned
out, to favour actions that lead somewhere it likes. A child inherits each
brain neuron whole from one parent or the other, with small mutations, so
learned habits pass down the generations.

Breeding costs each parent 20 hunger, and the newborn holds exactly that energy.
A newborn needs a free neighbouring cell, and the world holds at most 600 living
Bitzkeys. Reaching 100% hunger or zero health is fatal. Corpses fade after 30
turns. Brains, learning and statistics are all part of turn history, so rewind
and replay are exact.

## What the Bitzkeys Have Learned

Press `I` to swap the inspector for the behaviour panel. For each species, it
shows how often each action was taken when it was possible, over recent turns.
It also names the trait groups least and most keen to breed and least keen to
fight, e.g. "high-metabolism grazers 4%".

Press `B` to save a brain report to `reports/`. The report puts every living
brain in the same standard situations, such as a fed animal with a willing mate
beside it or a wounded animal beside a hunter. It averages the chance of each
action by species and trait group and ranks who is least and most keen.
