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
- Click a Bitzkey: inspect its genome and current state
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
words, decoded traits, hunger, health, age, and last action.

Animals forage, explore, rest, flee, defend, hunt, and attack according to their
traits and nearby Bitzkeys. Plants stay in place and photosynthesize. Compatible
animals of the same species can mate; offspring combine parental genomes and
may mutate. Hunger rises with metabolism, health recovers when conditions
allow, and reaching 100% hunger or zero health removes a Bitzkey from the living
population.

Aggression is selected by a shared, one-hidden-layer neural policy. Once per
turn, it evaluates all living animals with a learning trait of at least 4 in
one batch using their genome traits, hunger, health, nearby prey and threats,
and species. Its output adjusts the
genetic aggression value by up to five points, without changing DNA or color.
When a learned animal kills or witnesses a kill, the policy records a positive
or negative training example respectively. It trains with minibatches of 32
examples; animals below the learning threshold use their genome aggression
unchanged and do not contribute examples.
The policy weights and pending examples are part of turn history, so rewind
and replay preserve the learned behavior.