# Bitzkey

Bitzkey is a small artificial-ecology experiment. Each Bitzkey carries a 64-bit
genome made of two CMYK colour words: an inner dot for its mind and an outer
ring for its body. The world advances in turns on a grid (100x100 by default).

## Run

Install the dependency and start a world:

```powershell
python -m pip install -r requirements.txt
python Main.py
```

A configuration panel opens first. Set:

- the grid width and height (10 to 250 cells)
- the number of Bitzkeys spawned (up to 600). They appear in groups of 3 to 5
  sharing one genome, each group within two cells of a random spot.
- the grass density: the percent of cells with grass at the start (default 5).
  Each turn, 0.4% of that starting amount grows back, so density means the
  same at any grid size.
- an optional random seed. The same seed replays the same world; leave it blank
  for a new one. Click a row
or use Up/Down to choose a setting. Adjust it with its - and + buttons or
Left/Right (hold Shift for steps of ten), or type a number. Press Enter or
click START to begin, or Esc to quit.

Launch options fill in the panel's starting values:

```powershell
python Main.py --width 80 --height 60 --population 120 --grass-density 8 --seed 42
```

The panel can also import Bitzkeys, to add to the random ones. Paste exported
genomes into its text box with Ctrl+V or PASTE. Alternatively, load a file with
LOAD FILE, or drag the file onto the window. Several pasted blocks, one after
another, are all read. The line under the box says how many genomes of each
species are ready, or what can't be read; START waits until the text reads
cleanly or is cleared. Each imported genome spawns as a group of 3 to
5 Bitzkeys at a random spot, with fresh brains, so it arrives with mates. Up to
120 genomes are used, keeping their groups within the 600 population cap. Set
"Bitzkeys" to 0 to start with only your imports.

To add a setting, add a `Setting` to `SETTINGS` in `display/Setup.py` and a
parameter with the same name to `BitzkeyApp`.

## Build a Windows Executable

```powershell
.\build.ps1
```

This installs PyInstaller (from `requirements-build.txt`) and builds
`dist\Bitzkey.exe`, a single file that runs on Windows without Python and opens
no console window. The script uses the project's `.venv` when there is one. If
Windows blocks scripts, run
`powershell -ExecutionPolicy Bypass -File .\build.ps1` instead.

The executable saves `reports\` and `exports\` beside itself, so keep it in a
folder you can write to. The first launch takes a few seconds while it unpacks.

## Controls

- `Space`: pause or resume
- `Left Arrow`: rewind one recorded turn
- `Right Arrow`: advance one turn while paused, or replay a recorded turn
- `R`: restart with the same launch settings
- `I`: toggle the behaviour insights panel
- `B`: save a brain report to `reports/`
- `G`: open or close the colour graph; `Tab` switches it between rings and dots
- `C`: copy the selected Bitzkey's genome to the clipboard as JSON
- Click a Bitzkey: inspect its genome, current state and the odds of its last choice
- `Escape`: close the colour graph, or quit

History retains up to 500 world snapshots. Rewinding and stepping forward replays
the recorded future exactly; advancing from an earlier point replaces that
future with a new branch.

## Genome

Each Bitzkey is drawn as an inner dot inside a 3px ring. Its 64-bit genome is
the two colour words together. The low 32 bits colour the dot and shape the
mind; the high 32 bits colour the ring and shape the body. In each word, the
four bytes from most to least significant are cyan, magenta, yellow and black.
Each word also holds eight four-bit traits, read from its least-significant
bits.

**Inner dot: mind, emotion and social life.** These traits start at their
gene values and are then reshaped by experience:

| Trait | Effect | Experience that shifts it |
|---|---|---|
| aggression | pleasure of a fight, more for a kill | kills raise it; wounds and stings lower it |
| libido | pleasure of breeding | mating raises it; breeding into starvation sharply lowers it |
| sociability | pleasure of company from its own species | mating raises it; being wounded alone raises it |
| curiosity | pleasure of exploring | finding grass where none was seen raises it; walking into a hunter lowers it |
| fear | dread when hunters are adjacent | wounds, stings and ambushes raise it; kills lower it |
| appetite | how much hunger hurts | near-starvation raises it |
| patience | how much it values the future (its learning discount) | meals earned by foraging raise it; starving lowers it |
| intellect | how fast its brain learns, and how much experience can reshape the mind | fixed |

A learned shift fades slowly back toward the inborn value. Children inherit
genes and brains, not learned temperament.

**Outer ring: the body.**

| Trait | Effect |
|---|---|
| species | plant, grazer, omnivore or predator |
| size | health pool (60 + 6 × size), dot size and some striking force |
| strength | striking force |
| armour | reduces damage taken by up to 60% |
| weapons | sharpen blows by up to 75%, and spines hurt anyone who attacks |
| speed | cells moved per step of foraging, fleeing or exploring (1 to 4) |
| sight | how far away grass can be seen |
| fertility | high: cheap, frequent breeding and weak, hungry young; low: costly, rare breeding and well-fed young |

There is no metabolism gene. A bigger, stronger, faster or better-armed body
burns more food each turn.

Only the outer rings decide who can mate: two animals must be the same species,
and their body words may differ in at most 14 of 32 bits. A child's ring is a
single-point crossover of its parents' rings. A child's dot is the average of
its parents' dot colours, CMYK byte by byte; an odd half rounds up or down at
random. Then each word has a 20% chance of one flipped bit. Because minds
blend, a lineage's temperament converges, and mutation is the only source of
new mental variety.

The inspector shows both words in hex and every trait. Mind traits appear
inborn and as they are now, coloured green where experience has raised them
and amber where it has lowered them. It also shows the body's energy burn,
hunger, health, age, generation, children, last action, and the odds its
brain gave each option.

## Behaviour

Grass grows at random: 500 tufts at the start and 2 more each turn. Grazers
and omnivores eat grass; predators cannot, and live only by hunting.

Plants stay in place and photosynthesize. Every animal carries its own small
neural network, and that brain chooses everything it does. Each turn it senses:

- its hunger and health
- grass underfoot or in sight (grazers and omnivores only)
- a willing mate beside it, and how close the nearest one in sight is
- prey in sight, how close the nearest is, and how hard it could hit prey beside it
- hunters in sight, how close the nearest is, and how dangerous one beside it is
- nearby kin and its own age
- its current temperament

It then picks one of the actions open to it: eat, forage, rest, court, breed,
hunt, attack, flee or explore. Foraging walks toward the nearest grass in sight.
Courting walks toward the nearest willing mate in sight, but breeding is still a
separate choice once they are side by side.
Hunting chases the nearest prey in sight. A hunter that reaches its prey
pounces, striking in the same turn, and attacking strikes prey alongside. A kill
feeds the hunter 25 hunger plus 3 per point of the prey's size; a blow that
doesn't kill feeds it 5. Fleeing runs from the nearest hunter in sight, and
speed decides who wins a chase. Resting halves the hunger a Bitzkey gains that
turn and heals 1.5% of its health pool, three times the usual rate.
Better-valued actions are more likely, but no choice is certain. Mind traits set
what it enjoys and dreads; the brain learns online which actions lead there.
Death is heavily punished. A child inherits each brain neuron whole from one
parent or the other, with small mutations, so learned habits pass down the
generations.

Each parent pays 25 − fertility hunger to breed, and the newborn holds
exactly the energy both parents gave. A newborn needs a free neighbouring
cell, and the world holds at most 600 living Bitzkeys. Reaching 100% hunger or
zero health is fatal. Corpses fade after 30 turns. Brains, temperament,
learning and statistics are all part of turn history, so rewind and replay are
exact.

## Colour Graph

Press `G` for the colour census. Every living Bitzkey, plants included, is sorted
into one of 15 colour families: twelve hues, plus white, grey and black. The
graph shows time across the bottom, as stacked bars of how many Bitzkeys of each
family were alive. Each bar averages a run of consecutive turns, and its height
is the total population. `Tab` switches between ring (body) and dot (mind)
colours. The census covers the whole run, not just the 500 turns that can be
rewound.

The graph also exports Bitzkeys in bulk. Click legend colours to choose
families; with none chosen, every living Bitzkey is included. Then press `C`
or COPY to put them on the clipboard, or `S` or SAVE to write a `.json` file
into `exports/`.

## Genome Export Format

A genome is written as its two colour words in hex, ring then dot:

```json
{"ring": "7421FF41", "dot": "637E4B01"}
```

Several genomes form a JSON array, one per line. Import accepts single objects,
arrays, several blocks pasted one after another, and `#`-prefixed or lowercase
hex.

## What the Bitzkeys Have Learned

Press `I` to swap the inspector for the behaviour panel. For each species, it
shows how often each action was taken when it was possible, over recent turns.
It also names the trait groups least and most keen to breed and least keen to
fight, e.g. "high-metabolism grazers 4%".

Press `B` to save a brain report to `reports/`. The report puts every living
brain in the same standard situations, such as a fed animal with a willing mate
beside it or a wounded animal beside a hunter, each with its own current
temperament. It averages the chance of each action by species and trait group
and ranks who is least and most keen. It also shows how far experience has
moved each species' temperament from its genes.
