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

Bitzkey opens fullscreen at your desktop's resolution. Press `F11` to switch to
a resizable window and back, or start windowed with `python Main.py --windowed`.
Text and controls scale with the screen. They are designed for 1200x820 and
enlarged to fit, so a 1080p screen shows them about a third larger and a 4K
screen about two and a half times larger. Text is drawn at full resolution, not
stretched.

A configuration panel opens first. Set:

- the grid width and height (10 to 250 cells)
- the number of Bitzkeys spawned (up to 600). They appear in groups of 3 to 5
  sharing one genome, each group within two cells of a random spot.
- the grass density: the percent of cells with grass at the start (default 50).
  Each turn, 0.4% of that starting amount grows back, so density means the
  same at any grid size.
- an optional random seed. The same seed replays the same world; leave it blank
  for a new one.
- an optional population cap: the most animals alive at once. Leave it blank,
  the default, for no limit, and crowding alone decides.
- the mind mutation chance (0 to 100, default 50). A birth has three chances,
  each this percent, to change one of the child's mind traits by one step: about
  1.5 mind mutations a birth at 50, and three at 100. At 0 minds are
  genetically stable.
- the body mutation chance (0 to 100, default 50): the percent chance that a
  birth flips one bit of the child's ring. At 100 every child carries one body
  mutation.

Click a row or use Up/Down to choose a setting. Adjust it with its - and +
buttons or Left/Right (hold Shift for steps of ten), or type a number. Press
Enter or click START to begin, or QUIT (or Esc) to close the application, after confirming.

Launch options fill in the panel's starting values:

```powershell
python Main.py --width 80 --height 60 --population 120 --grass-density 8 --seed 42 --population-cap 500 --mind-mutation 80 --body-mutation 20
```

The panel can also import Bitzkeys, to add to the random ones. Paste exported
genomes into its text box with Ctrl+V or PASTE. Alternatively, load a file with
LOAD FILE, or drag the file onto the window. Several pasted blocks, one after
another, are all read. The line under the box says how many genomes of each
species are ready, or what can't be read; START waits until the text reads
cleanly or is cleared. DESIGN opens the Bitzkey designer. It shows all 20 traits
of a genome, mind and body, each adjustable with its - and + buttons, the arrow
keys (`Shift` for steps of five), or by typing a digit from 0 to 7. The ring and
dot hex codes can also be typed whole. A large preview shows the Bitzkey in its
true colours, at sixteen times its in-world size and proportions, beside it at
actual size. Below them are the stats that follow from the genome: health,
metabolism, stomach, energy burn, speed, sight, breeding cost, litter size, how much carrion feeds
it, and its lifespan. Species is stored as a value from 0 to 7; each value picks
a species (value mod 4) and also shades the ring. APPLY (`Enter`) adds the
design to the bottom of the import box, and CANCEL (`Esc`) leaves it out. RANDOM
rolls a new genome. Reopening the designer starts from the last design applied,
so variations are quick to build. A designed Bitzkey starts with a fresh brain
and its genes' temperament.

Each imported genome spawns as a group of 3 to
5 Bitzkeys at a random spot, with fresh brains, so it arrives with mates. Up to
120 genomes are used, keeping their groups within the 600 population cap. Set
"Bitzkeys" to 0 to start with only your imports.

To add a setting, add a `Setting` to `SETTINGS` in `display/Setup.py` and a
parameter with the same name to `BitzkeyApp`.

## When a Simulation Ends

A run ends by itself when every animal has died, leaving only plants and grass,
or when the world is full, with every cell taken by a living Bitzkey. The world
then stops at that turn and the colour graph opens, saying when and why it
ended. Space won't carry it on, but you can still rewind and scrub through the
whole run, and replaying up to the end stops there again. R restarts the same
world, and Esc returns to the configuration panel.

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
- `C`: copy the selected Bitzkey's genome and brain to the clipboard as JSON
- `A` or ADD TO EXPORT (under the selected Bitzkey): add it to, or remove it from,
  the export list. Bitzkeys on the list wear an amber ring in the world.
- `E` or EXPORT ALL SELECTED (bottom right): save the export list to a file in
  `exports/` or copy it to the clipboard. Living Bitzkeys export as they are now,
  with what they've learned since being added. Any that have died export as they
  were when added, brain included. The dialog can also clear the list.
- Click a Bitzkey: inspect its genome, current state and the odds of its last choice
- `F11`: switch between fullscreen and a window
- `Escape`: close the colour graph; otherwise ask whether to leave the simulation.
  Leaving greys out the button ("Leaving, please be patient") while the world's
  recorded history and brains are freed, which can take a few seconds for a long run
  and return to the configuration panel. The panel keeps its settings and import
  list, ready for a new world. Bitzkeys on the export list are lost unless
  exported first, and the dialog warns about them. Closing the window quits.

You can rewind to any turn of the run, back to turn 0. Every one of the latest
500 turns is kept as a snapshot of the world. Older turns are kept as up to 200
evenly spaced checkpoints, spread further apart as the run grows. Reaching a turn
between checkpoints restores the one before it and replays forward. The
simulation is deterministic, so the replay repeats the recorded run exactly.
Stepping or playing forward from the past replays the recorded future.

## Genome

Each Bitzkey is drawn as an inner dot inside a 2px ring. Its 64-bit genome is
the two colour words together. The low 32 bits colour the dot and shape the
mind; the high 32 bits colour the ring and shape the body. In each word, the
four bytes from most to least significant are cyan, magenta, yellow and black.
Each word also holds ten three-bit traits, each 0 to 7, read from its
least-significant bits. The top two bits of each word hold no trait, but they
still shade the colour.

**Inner dot: mind, emotion and social life.** These traits start at their
gene values and are then reshaped by experience:

| Trait | Effect | Experience that shifts it |
|---|---|---|
| aggression | pleasure of a fight, more for a kill | kills raise it; wounds and stings lower it |
| libido | pleasure of breeding | mating raises it; breeding into starvation sharply lowers it |
| sociability | pleasure of company | mating raises it; being wounded alone raises it |
| curiosity | pleasure of exploring | finding grass where none was seen raises it; walking into a hunter lowers it |
| fear | dread when hunters are adjacent | wounds, stings and ambushes raise it; kills lower it |
| appetite | how much hunger hurts | near-starvation raises it |
| patience | how much it values the future (its learning discount) | meals earned by foraging raise it; starving lowers it |
| intellect | how fast its brain learns, and how much experience can reshape the mind | fixed |
| empathy | pain at seeing an animal wounded or killed nearby; more for its own kind, double for a death. An empathetic hunter feels some of its victim's pain and enjoys fighting less. | fixed |
| tolerance | ease among other species: they count as company, hunters are less frightening, and attacking gives less pleasure. The intolerant are uneasy whenever other species are near. | fixed |

A learned shift fades slowly back toward the inborn value. Children inherit
learned temperament as well as genes and brains: each child's traits start at
the average of its parents' current traits, as experience has left them, not
their inborn ones. Then three times, each with the world's mind mutation
chance (50% by default), a random trait mutates a step up or down, within 0
to 7. The child's dot is written from that starting mind, each
fraction rounding up with a chance equal to it, so its colour shows the
temperament it was born with, and what its parents lived through shows in
their young.

**Outer ring: the body.**

| Trait | Effect |
|---|---|
| species | plant, grazer, omnivore or predator (value mod 4) |
| size | health pool (60 to 150), dot radius (1px to 5px), half its striking force and half its toughness, metabolism, stomach, lifespan, and the meat its corpse leaves |
| strength | the other half of its striking force |
| armour | the other half of its toughness, which reduces damage taken by up to 60% |
| weapons | sharpen blows by up to 75%, and spines hurt anyone who attacks |
| speed | cells moved per step of foraging, fleeing or exploring (1 to 4; grazers move half as far again, rounded up: 2, 3, 5 or 6) |
| sight | how far it sees (2 to 9 cells) |
| fertility | high: cheap, frequent breeding and weak, hungry young; low: costly, rare breeding and well-fed young |
| litter | young per birth: 1 to 6. A high litter trait and a small body both make litters bigger, half each, averaged over both parents, so small animals tend to have large litters and large animals small ones. Predators have half as many young, rounded up (1 to 3). Each young needs a free cell around the parent (sides first, then corners), and the litter shares the energy its parents give. |
| carrion | how much of carrion's meat nourishes it: 20% at 0, all of it at 7 |

There is no metabolism gene: size sets it. The smallest bodies run at 1.6 and
the largest at 0.6. A faster metabolism empties the stomach sooner, heals
faster, and makes for a shorter life. Size also sets the stomach, from 0.5
(smallest) to 2.0 (largest). Hunger is how empty the stomach is, so the same
meal fills a small animal four times as much as a large one, and a large
animal needs more food in all. Its slower metabolism offsets part of that: a
bare large body eats about 1.5 times what a bare small one does each turn, not four
times. Strength, speed, weapons and armour all add to the burn.

Only the outer rings decide who can mate: two animals must be the same species,
and their body words may differ in at most 14 of 32 bits. A child's ring is a
single-point crossover of its parents' rings. A child's dot comes from its
starting mind (see above), its two spare bits from either parent. With the
world's body mutation chance (50% by default), one bit of the ring flips.
Minds blend, but experience and mind mutations keep them varied.

This is genome version 2. Version 1 held eight four-bit traits (0 to 15) per
word, before empathy, tolerance, litter and carrion existed. Older exports,
which carry no version, are converted when imported: each trait is scaled to 0
to 7, and empathy starts at 0, tolerance at 3, litter at 0 and carrion at 0.
Their colours change, because the bits move.

The inspector shows both words in hex and every trait. Mind traits appear
inborn and as they are now, coloured green where experience has raised them
and amber where it has lowered them. It also shows the body's energy burn,
hunger, health, age against expected lifespan, generation, children, last
action, the odds its brain gave each option, and for a corpse, the meat left on
it.

## Behaviour

Grass is sown at random, and regrows each turn (see grass density above). On and around a corpse, the
cell it lies on and the eight cells next to it, grass grows twice as fast
while the body remains.
Grazers and omnivores eat grass, and browse plants; predators can do neither.
Any animal may eat carrion.

Plants stay in place and photosynthesize. Every animal carries its own small
neural network, and that brain chooses everything it does. Each turn it senses:

- its hunger and health
- grass underfoot or in sight, and plants in sight (grazers and omnivores only)
- how close the nearest carrion in sight is
- a willing mate beside it, and how close the nearest one in sight is
- prey in sight, how close the nearest is, and how hard it could hit prey beside it
- hunters in sight, how close the nearest is, and how dangerous one beside it is
- whether the animal that just struck it is still beside it, and how hard it could strike back
- its own kind and other species close by, and its own age
- its current temperament

It then picks one of the actions open to it: eat, forage, browse, scavenge,
rest, court, breed, hunt, attack, defend, flee or explore. Foraging walks toward the
nearest grass in sight; a tuft sates 35 hunger. Grazers get all the food in grass and plants; omnivores, half. Browsing walks to the nearest
plant in sight and, once alongside, takes a bite: 45 hunger, up to 70 from the
largest plants, far more than grass. Each bite costs the plant 15 health, and
plants regrow 2% of their health each turn. A bitten plant can't be bitten again
for 8 turns, so browsers move on, but enough bites in time kill one, and a dead
plant leaves no meat. Plants spread by seed: each turn a well-fed plant has a 3%
to 10% chance, by its fertility, of sprouting a seedling of its own kind, with
the usual chance of mutation, on a free cell up to five cells away. Seeding
costs it 20 hunger. A seed doesn't take where more than three plants already
grow within two cells, and plants are capped at 200. Scavenging walks to the
nearest carrion in sight and, once alongside, eats up to 25 meat; the carrion
trait decides how much of that feeds it. Courting walks toward the nearest
willing mate in sight, but breeding is still a separate choice once they are
side by side. Hunting chases the nearest prey in sight. Omnivores prey on grazers; predators prey on grazers, omnivores and other predators. Predators whose ring colours differ by more than 5% (the mean difference of the ring's four CMYK bytes) are different kinds, and hunt one another freely. Hunting their own kind is cannibalism: a predator only sees its own kind as prey once its hunger, as a share, exceeds its reluctance, the average of its current empathy, tolerance and patience. The ruthless will eat kin at any hunger; the gentlest never will. Killing its own kind pains an empathetic predator as much as its own species' pain, rather than the reduced pain it feels for others. Predators and omnivores are territorial. An animal's territory is the cells within six steps of it. A predator counts other predators and omnivores in its territory as rivals for the feeding ground, and how many it will share it with depends on its current empathy: one for the least empathetic, rising to five for the most. Each predator beyond that raises its rivalry by a third, up to full rivalry three past its limit. A predator senses its rivalry, finds an overcrowded territory uncomfortable (less so if tolerant), and feels relief, as well as any joy in fighting, at killing a rival there. Crowding by other predators, though not by omnivores, also wears down its reluctance to turn on its own kind: in a territory fully overcrowded with predators, even the gentlest will fight kin for the feeding ground. Omnivores are territorial at half the strength. They count other omnivores as rivals, and predators too, each counting less the more empathetic the omnivore is, and not at all at the greatest empathy. They tolerate two rivals for the least empathetic, rising to ten for the most, and their rivalry tops out at a half. An omnivore only turns on another omnivore when its territory is overcrowded with omnivores, and then only once its hunger exceeds its reluctance as rivalry wears it down, never by more than half. A hunter that reaches
its prey pounces, striking in the same turn, and attacking strikes prey
alongside. A kill feeds the hunter 25, plus up to 45 more for the largest prey; a blow
that doesn't kill feeds it 5. Striking is hard work: each blow burns 0.2 food per point of damage it deals, so a heavy hitter pays more for each blow. Hunger figures for meals are for a middle-sized
stomach: a small animal gains twice as much, a large one half. Defending is open only to an animal struck since its last turn whose assailant is still beside it: it strikes back with the usual force of its size, strength and weapons against the assailant's size and armour. It costs effort like any blow and feeds the defender nothing, but it can kill. Fight or flight: whenever an animal is struck, by an attack or a counterblow, it checks its morale against its current fear. A fully fearful animal breaks 30% of the time when barely hurt, rising to certainty near death; the fearless never break. One that breaks loses its chance to defend: on its next turn it flees its assailant, whatever its brain would have chosen, and grows a little more fearful. Plants can't flee, so their only answer is to fight: a plant with any weapons strikes back at each animal that browses it. Its thorns come mostly from its weapons (80%) and a little from its size (20%), never its strength: at most 15 damage, less against a large or armoured browser. Fleeing runs from the nearest hunter
in sight, and speed decides who wins a chase. Resting halves the hunger a
Bitzkey gains that turn and heals 1.5% of its health pool, three times the usual
rate. Both rates are scaled by metabolism, so small animals heal faster. Better-valued actions are more likely, but no choice is certain. Mind
traits set what it enjoys and dreads; the brain learns online which actions lead
there. Death is heavily punished. A child inherits each brain neuron whole from
one parent or the other, with small mutations, and its parents' learned shifts
in temperament, so learned habits and moods pass down the generations.

Each parent pays 10 to 25% of its stomach to breed, less the more fertile it is. Animals
only breed, or court, when both partners can afford it and still have at least
10 hunger to spare afterwards, so nothing breeds itself to death. The litter
shares exactly the energy its parents gave, so a large parent gives more food
than a small one, and the same food fills a small newborn more. Each newborn needs a free
neighbouring cell, and if a population cap is set, the world holds at most that
many living animals (plants are limited separately, to 200). Crowding also
limits breeding: for each other animal within four cells of a would-be parent,
besides its partner, its litter shrinks by a tenth, and with ten or more around,
it does not breed. Reaching 100% hunger or zero health is fatal, and so, in
time, is age. Plants live about 4,000 turns, and animals from about 250 (the
smallest, fastest metabolism) to 900 (the largest, slowest). From 70% of its lifespan, a Bitzkey's chance of
dying of old age grows each turn: about half reach their lifespan, and almost
none live past 1.3 times it. Dying of old age is not held against the brain's
last choice. A dead Bitzkey is dimmed and crossed through in red until its
corpse fades. A corpse holds the energy its body had left: up to 30 meat, plus
up to 50 more for the largest. A starved body leaves nothing, and a killed one
half, as the killer has already eaten. The meat rots away over the 30 turns
before the corpse fades. Brains, temperament, learning and statistics are all
part of turn history, so rewind and replay are exact.

## Colour Graph

Press `G` for the colour graph. Time runs across the bottom. Each bar is a moment
in the run, and every Bitzkey alive then (plants included) is a slice of it in its
own exact colour. Slices are sorted by hue, so similar colours flow together from
bar to bar, and a bar's height is the population. `Tab` switches between ring
(body) and dot (mind) colours. The graph covers the whole run. It samples who was
alive at evenly spaced turns, spacing them further apart as the run grows.

The graph is also a time scrubber. A playhead marks the present. Drag along the
timeline beneath the bars, or right-click anywhere on the graph, to move the
world to that turn; it pauses there. Space plays on, replaying the recorded
future, which stays visible but faded. On long runs, a far jump may take a
moment while it replays, and says so. The mouse wheel zooms the time axis around
the cursor, down to 10 turns. `Shift`+wheel, or a sideways wheel, pans the
zoomed view, and zooming all the way out shows the whole run again.

Pick Bitzkeys straight from the graph:

- click a slice to select that Bitzkey; click empty space to clear
- drag a box to select every Bitzkey it touches, in every bar it spans
- hold `Shift` to add to the selection (`Shift`-click toggles one), or `Ctrl` to
  remove from it

A selected Bitzkey stays lit in every bar it appears in while the others fade.
Hovering over a slice names the Bitzkey: its species, whether it is alive, and
its ring and dot codes.

Then press `C` or COPY to put the selection on the clipboard, or `S` or SAVE to
write a `.json` file into `exports/`. CLEAR empties the selection. Living
Bitzkeys export with their temperament and brain. Dead ones have no brain, and
those no longer on the board export as their genome only. With nothing
selected, COPY and SAVE take everyone alive.

## Export Format

A Bitzkey is written as its genome version and its two colour words in hex,
ring then dot. Animals also carry their current temperament and their brain, so
they pick up in the next simulation where they left off:

```json
{"version": 2, "ring": "7421FF41", "dot": "637E4B01",
 "mind": {"aggression": 9.412, "libido": 3.0, ...},
 "brain": {"senses": ["hunger", ...], "actions": ["eat", ...],
           "hidden_weights": [[...], ...], "hidden_biases": [...],
           "output_weights": [[...], ...], "output_biases": [...]}}
```

Plants have no brain or changing temperament, so they export only their colours. Several Bitzkeys form a
JSON array, one per line; an animal with its brain takes about 5 KB.

Import accepts single objects, arrays, several blocks pasted one after another,
and `#`-prefixed or lowercase hex. A record without a `"version"` is read as
genome version 1 and converted, so hand-written records for this version need
`"version": 2`. A record without a brain gets a fresh one.
Brain weights are matched to senses and actions by name, so exports from an
earlier version still load: senses or actions they lack start neutral, and ones
this version no longer has are dropped. Every member of an imported group starts
with its own copy of the saved brain and the saved temperament, and their
children inherit both as usual. Temperament is read for the known traits only,
kept within 0 to 7; a trait it lacks starts at the gene's value. As in life, a
restored temperament slowly drifts back toward the genes unless experience
keeps it shifted.

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
