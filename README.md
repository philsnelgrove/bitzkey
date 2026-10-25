# Bitzkey

The Bitzkey project was started years ago, to teach my best friend programming. He wanted to learn about Python and AI, so it seemed like a natural project. Since then, my friend has passed away. So I finished this to honour him. I open-sourced it for the same reason. So, Sam Bunn, wherever you are, may you live on through Bitzkeys!

This is the original Bitzkey, the research build, written in Python. It runs very slowly, but every rule is plain to read, so it is the place to learn how Bitzkeys work: how they sense, think, learn, breed and evolve. It is too slow to run a world for long, though. For that, use the much faster C++ versions: the [Windows desktop app](https://github.com/philsnelgrove/bitzkey_cpp), or the [Linux daemon](https://github.com/philsnelgrove/bitzkey_daemon), which runs larger worlds as a service and shows them as a web page.

Download it for Windows from the [latest release](https://github.com/philsnelgrove/bitzkey/releases/latest): `Bitzkey.exe` runs as it is, with nothing to install.

Bitzkey is a small artificial-ecology experiment. Each Bitzkey carries a 96-bit
genome made of three CMYK colour words: an inner dot for its mind, an outer
ring for its body, and a social background, behind its species icon, for how it
treats others. The world advances in turns on a grid (100x100 by default).

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

- the grid width and height (10 to 1,000 cells)
- the number of Bitzkeys spawned (up to 20,000; worlds that big want the C++ version thinking on the GPU). They appear in groups of 3 to 5
  sharing one genome, each group within two cells of a random spot. Every
  climate zone gets at least one group of plants: the plant groups take a zone
  each, from the top down, and if too few groups come up plants, the last animal
  groups become plants instead, though at least one animal group is kept.
- the grass density: the percent of cells with grass at the start (default 50).
  Each turn, 3.2% of that starting amount grows back, so density means the
  same at any grid size. Grass grows only where it is medium (see Seasons and
  Climate).
- an optional random seed. The same seed replays the same world; leave it blank
  for a new one.
- an optional population cap: the most animals alive at once. Leave it blank,
  the default, for no limit, and crowding alone decides.
- the mind mutation chance (0 to 100, default 25). A birth has three chances,
  each this percent, to change one of the child's temperament traits, mind or
  social, by one step: about
  1.5 mind mutations a birth at 50, and three at 100. At 0 minds are
  genetically stable.
- the body mutation chance (0 to 100, default 25): the percent chance that a
  birth flips one bit of the child's ring. At 100 every child carries one body
  mutation. A flip that lands on the species is moved, half the time, to another bit rolled afresh,
  so the species changes half as often as the rest.
- the year length (20 to 3,600 turns, default 360): a year has four seasons
  (see Seasons and Climate), and lifespans are counted in years.

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
cleanly or is cleared. FROM CATALOG opens the catalog of your favourite
Bitzkeys (see The Catalog) and adds the one you pick, trained brain and all.
DESIGN opens the Bitzkey designer. It shows all 20 traits
of a genome, mind and body, each adjustable with its - and + buttons, the arrow
keys (`Shift` for steps of five), or by typing a digit from 0 to 7. Below each
column is a climate gene, from 0 to 3: hardiness under the mind (none, heat, cold
or both) and the home temperature under the body (hot, medium, cold or very cold). The ring and
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
120 genomes are used. Set
"Bitzkeys" to 0 to start with only your imports.

ADVANCED opens the advanced rules, beneath a word of warning that they are the world's
'natural law', carefully balanced, and that changing them can have unpredictable results: 227 numbers that shape how the world plays
out, in groups (World, Bodies, Breeding, Feelings, Mind, Fighting, Territory,
Herding, Food, Plants, Climate, Society and Experiences). Among them are lifespans, metabolism, litter
sizes, food-aware breeding, the rewards and pains a brain learns from, blows and
thorns, territory and herd instincts, meals, fresh kills, plant seeding, the climate
zones, and how each experience reshapes temperament. Each rule has a short note, limits and a step. Think on the GPU is the C++ version's alone: there, it moves every brain onto the GPUs, and is on by default wherever a GPU is found. The Python version always thinks on the CPU, so here it is greyed out. Choose and adjust
them as in the configuration panel; PgUp/PgDn jump ten rules, and the wheel scrolls.
Fractional rules take a typed decimal point, and the experience shifts take a minus
sign. Changed values show in amber. RESET puts the focused rule back to its default
and RESET ALL every rule. APPLY (`Enter`) keeps the changes, unless two rules
conflict (the smallest litter above the largest, equal metabolisms, or arctic
zones and an equator too wide to fit the map together); CANCEL
(`Esc`) drops them. The ADVANCED button turns amber and counts the changed rules.
They hold for every world started from the panel, `R` restarts included, until
changed again. A seed replays the same world only under the same rules.

ADVANCED in the map's footer, beside OPEN CATALOG, pauses the world and opens the same rules
as THIS WORLD / ADVANCED RULES, as the world is playing by them. Applied, they hold from its
next turn, and for its restarts; `SPACE` plays on. Everything the world settled from the rules
as it began is worked out afresh (the grass that grows back each turn, the climate bands), and
every rule is read as the world goes, so the change takes hold at once. Think on the GPU is
greyed out there: a running world thinks where it started. Turns already played are kept as
they were, so rewinding replays them, though a turn played again after a rewind plays by the
rules as they now stand. The button turns amber while the world plays by changed rules. A new
world from the configuration panel plays by the panel's rules again.

### Saving and loading worlds

SAVE WORLD, in the map's footer beside ADVANCED, pauses the world and saves the whole of it to a
file, in `worlds/` unless you choose another folder. `Esc` on the map asks whether to leave, and
SAVE AND LEAVE saves the world first. LOAD WORLD, on the configuration panel, carries on a saved
world: it comes back at the turn it was saved, paused, and plays on exactly as it would have.
Every Bitzkey comes back with its position, brain, learning traces, memories, temperament, the
young it carries and everything else about it, along with the grass, the seeds waiting to sprout,
the random generator's state, the insights, the colour graph's census and the record of every
death. The world brings its own rules, all but the three that say how this computer runs it:
Think on the GPU, History turns and History memory. Restarting a loaded world (`R`) starts afresh
by the panel's settings. The rewind history starts again at the turn loaded.

A world file is the same for the Python and C++ versions and the web daemon, so a world saved in one
loads in the others and carries on there turn for turn the same. Brains are most of a file, kept
exactly, at about 120 KB an animal: 8 MB for a world of 60 animals, 120 MB for a thousand. The
format is described at the top of `tools/WorldFile.py`. A long-lived world also carries its history: the
genome of every Bitzkey and the record of every death over the last 250,000 turns, 40 bytes for each, kept compact and
written straight to the file as it goes, so a world of 20 million deaths saves and loads in seconds. Worlds
saved in the first format, before that, still load.

To add a setting, add a `Setting` to `SETTINGS` in `display/Setup.py` and a
parameter with the same name to `BitzkeyApp`. To make another constant in
`tools/Simulation.py` or `tools/Brain.py` tunable, add a row for it to `_TABLE` in
`tools/Rules.py`. Code that uses it must read it from its module when it runs,
not copy it at import. Add the same rule to the C++ port.

## When a Simulation Ends

A run ends by itself when every animal has died, leaving only plants and grass,
or when the world is full, with every cell taken by a living Bitzkey. The world
then stops at that turn and the colour graph opens, saying when and why it
ended. Space won't carry it on, but you can still rewind and scrub through the
whole run the history keeps, and playing up to the end stops there again. RETURN TO MAP closes
the graph to look over the map as it ended; G opens the graph again. Closing the
graph with G or Esc returns to the configuration panel, first asking to leave if
Bitzkeys wait on the export list. R restarts the same world.

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

## Exemplars

Either side of the map, five cards a side, evenly spaced, show the living animals
that most stand out. Each card's title is a menu: click it to choose, from 50
statistics, what that card looks for, and the card shows the animal that is
highest by it (or lowest, for those that say so). By default the left column shows
the smartest, the largest, the oldest, the one with the most children and the
latest generation, and the right the strongest, the fastest, the best armoured,
the most fertile and the best fed. The choices last for the rest of the session,
through new worlds. They are:

- temperament, as born: the smartest, most aggressive, most amorous, most
  sociable, most curious, most fearful, greatest appetite, most patient, most
  empathetic, most tolerant, best memory, most resilient, moodiest, most loyal, most
  dominant, most trusting and most clannish (kinship); and the simplest, gentlest,
  bravest and most solitary
- body: the largest and smallest, strongest, best armoured, best armed, fastest
  and slowest, sharpest-eyed, most fertile, biggest litters and best scavenger
- climate: the hottest and the coldest home
- now: the oldest and youngest, most children, latest generation, best fed and
  hungriest, healthiest and most wounded (by share of full health), longest and
  shortest lifespan, fastest and slowest metabolism, biggest stomach, deepest
  brain, most herd-minded, and most changed by experience (how far its
  temperament has moved from what it was born with)

Plants never count. Ties go to the oldest, then the first born. Each card shows
the Bitzkey, its number, its species and the value it stands out by, and keeps up
as the world runs. Click a card to select that Bitzkey, as clicking its dot would,
ready to copy, export or catalog; the selected one's card is outlined in amber.
`Esc` or a click elsewhere closes an open menu.

## Controls

- `Space`: pause or resume
- `Left Arrow`: rewind one recorded turn
- `Right Arrow`: advance one turn while paused, or replay a recorded turn
- Mouse wheel over the map: zoom in and out around the pointer, without
  pausing; drag with the right or middle button to pan; `Z`: the whole map again.
  The Bitzkeys zoom with the map: the largest body's dot, ring and all, fills a
  cell, and the smallest, always big enough for its icon, grows with it in
  proportion
- Hover over a map square to list the Bitzkeys stacked in it, with the spaces
  each takes and the square's spaces in use; the list stays while the pointer is
  on it, and clicking a name selects that Bitzkey.
- Click a dot on the map to select that Bitzkey, or empty ground to select none;
  drag a box with the left button to select every living Bitzkey inside it.
  `Ctrl`-click adds a Bitzkey to the selection or takes it out again, and
  `Ctrl`-drag adds a box's worth. Each selected Bitzkey wears a target's sights, swelling slowly to twice their size and back every two seconds, paused or not:
  a white ring inside a black one, with a pip at each point of the compass
- `R`: restart with the same launch settings
- `I`: toggle the insights panel (scroll it with the wheel or its scroll bar)
- `B`: save a brain report to `reports/`
- `G`: open or close the colour graph; `Tab` switches it between rings and dots.
  Once the run has ended, closing it returns to the configuration panel.
- `C`: copy the selected Bitzkey's genome and brain to the clipboard as JSON
- `A` or ADD TO EXPORT (under the selected Bitzkey): add it to, or remove it from,
  the export list. Bitzkeys on the list wear an amber ring in the world.
- `K` or ADD TO CATALOG (beside it): add the selected Bitzkey to the catalog,
  as it is now, brain and all. On the colour graph, `K` or ADD TO CATALOG adds
  the living Bitzkeys picked there.
- `E` or EXPORT ALL SELECTED (bottom right): save the export list to a file in
  `exports/` or copy it to the clipboard. Living Bitzkeys export as they are now,
  with what they've learned since being added. Any that have died export as they
  were when added, brain included. The dialog can also add the whole list to the
  catalog, or clear it.
- Click a Bitzkey: inspect its genome, current state and the odds of its last choice
- `F11`: switch between fullscreen and a window
- `Escape`: close the colour graph; otherwise ask whether to leave the simulation.
  Leaving greys out the button ("Leaving, please be patient") while the world's
  recorded history is let go, and returns to the configuration panel. The panel keeps its settings and import
  list, ready for a new world. Bitzkeys on the export list are lost unless
  exported first, and the dialog warns about them. Closing the window quits.

You can rewind the map to any of the latest 100 turns. The history keeps the
world at each of them whole, in memory, so reaching one is instant, with nothing
to replay and nothing written to the disk. When a crowded world's brains make
them large, fewer turns are kept, so the history stays within about 1 GB; the
footer shows the earliest turn kept (HISTORY 8100/8100 FROM 8001). Older turns
live on only in the colour graph. Stepping or playing forward from the past steps
through the recorded future.

Both limits are advanced rules (the World group): History turns, from 1 up to ALL,
which keeps every turn so far as the memory allows, and History memory, in GB.
Both carry a warning, shown in amber, and in full when the rule is chosen: each
turn kept holds the whole world, about 200 KB a living brain (some 20 MB a turn
with 100 animals), so set the memory well below what your computer has free, or
Bitzkey will slow to a crawl or crash.

## Genome

Each Bitzkey is drawn as an inner dot inside a 2px ring. A dot big enough for
its species' icon is filled with its social background's colour, with the icon
in the dot's colour. Its 96-bit genome is the three colour words together. The
low 32 bits colour the dot and shape the mind; the next 32 bits colour the ring
and shape the body; the high 32 bits colour the social background and shape how
it treats others. In each word, the four bytes from most to least significant
are cyan, magenta, yellow and black. The dot and the ring each hold ten
three-bit traits, each 0 to 7, read from their least-significant bits. Their top
two bits hold the climate genes, and like the rest they shade the colour: in the
ring, the home temperature (0 hot, 1 medium, 2 cold, 3 very cold); in the dot,
the heat tolerance flag (bit 30) and the cold tolerance flag (bit 31). See
Seasons and Climate. The social background holds seven three-bit traits, then a
four-bit neurodivergence gene (0 to 15) that does nothing yet; its top seven bits
are spare, shading its colour only.

**Inner dot: mind and emotion.** These traits, and the social background's,
start at their gene values and are then reshaped by experience, drifting back
toward them only very slowly (0.05% of the gap a turn):

| Trait | Effect | Experience that shifts it |
|---|---|---|
| aggression | pleasure of a fight, more for a kill | kills raise it; wounds and stings lower it |
| libido | a quarter of the pleasure of breeding and of courting (the rest every animal feels) | mating raises it; breeding into starvation sharply lowers it |
| curiosity | pleasure of setting off in a direction | finding grass where none was seen raises it; walking into a hunter lowers it |
| fear | dread when hunters are adjacent; a predator feels half its fear's effects, and less the hungrier it is, none as it starves | wounds, stings and ambushes raise it; kills lower it |
| appetite | how much hunger hurts | near-starvation raises it |
| patience | how much it values the future (its learning discount) | meals earned by foraging raise it; starving lowers it |
| intellect | how big its brain is (a hidden layer for each point up to four, then five neurons more in every layer for each point beyond), how fast it learns (its learning rate, boosted for the bright, how many recent experiences it replays each turn, and how far back it traces an outcome), how much experience can reshape the mind, how much it learns from what it sees, and how rarely primal fear freezes it; but a bigger brain costs food: hunger burns up to 15% faster at the brightest, in proportion below, and a predator pays half that (Brain cost, Predator brain cost). A predator's intellect counts twice over in all of these, up to the full range, and its brain learns as fast as the Predator learning rule sets (by default no faster). | kills (twice as much as anything else), breeding, discoveries and successful defences raise it; starving, wounds and panic lower it |
| memory | how long it remembers where it last fed and where it was last struck: up to 100 turns at the best memory, fewer the poorer, none at 0 (Memory turns). It senses how fresh the memory of its last meal is, and how close it is to where it was struck, and while it remembers where it fed it may revisit, heading back there | fixed |
| resilience | how quickly it gets over fright and loss. A herd-mate's failure it saw fades half as fast again for the most resilient, and half as slowly for the least, and the fear a bad experience adds is half again for the least resilient and half for the most (Resilience) | fixed |
| moodiness | how far experience moves its temperament: half as far again for the moodiest, half as far for the steadiest (Moodiness) | fixed |

**Social background: how it treats others.** These are part of its
temperament, inherited and reshaped by experience as the dot's are:

| Trait | Effect | Experience that shifts it |
|---|---|---|
| sociability | pleasure of company | mating raises it; being wounded alone raises it |
| empathy | pain at seeing an animal wounded or killed nearby; more for its own kind, double for a death. An empathetic hunter feels some of its victim's pain and enjoys fighting less. | fixed |
| tolerance | ease among other species: they count as company, hunters are less frightening, and attacking gives less pleasure. The intolerant are uneasy whenever other species are near. The tolerant also accept mates, and herd-mates, that look less like themselves. | fixed |
| loyalty | keeping to its herd: whenever it can herd, it values herding more, a quarter of a point at full loyalty (Loyalty pull) | fixed |
| dominance | a ready animal breeds with the most dominant willing partner within reach; and a crowded territory troubles the most dominant half as much and the most submissive half as much again (Dominance ease) | fixed |
| trust | how much it takes in of what it sees its herd-mates do, half as much again for the most trusting and half for the least (Trust heed), and how far off it heeds a herd-mate's alert: up to twice its trust times its sight away (Trust reach) | fixed |
| kinship | how strongly it favours its own social background: a partner or herd-mate whose social background colour differs by more than 1 less 0.8 times its kinship, less its tolerance's share, is shunned (Kinship shun). At no kinship any social background will do; at full kinship and no tolerance, only one within 20% | fixed |
| neurodivergence | (four bits, 0 to 15) nothing yet; inherited whole from either parent | fixed |

A learned shift fades slowly back toward the inborn value. Children inherit
learned temperament as well as genes and brains: each child's traits start at
the average of its parents' current traits, as experience has left them, not
their inborn ones. Then three times, each with the world's mind mutation
chance (25% by default), a random trait mutates a step up or down, within 0
to 7. The child's dot and social background are written from that starting
temperament, each fraction rounding up with a chance equal to it, so their
colours show the temperament it was born with, and what its parents lived
through shows in their young. Its neurodivergence comes whole from either parent.

**Outer ring: the body.**

| Trait | Effect |
|---|---|
| species | plant, grazer, omnivore or predator (value mod 4) |
| size | health pool (60 to 150), dot radius (1px to 5px), half its striking force and half its toughness, metabolism, stomach, lifespan, and the meat its corpse leaves |
| strength | the other half of its striking force |
| armour | the other half of its toughness, which reduces damage taken by up to 60% |
| weapons | sharpen blows by up to 75%, and spines hurt anyone who attacks; a predator's count three times over, sharpening its blows by up to 225%; an omnivore's 2.25 times (up to 169%) and a grazer's 1.5 times (up to 113%); in defence, striking back or stinging, a grazer brings half its weapons and an omnivore three quarters, though an omnivore has no spines: struck, it never stings its attacker; a predator striking an armed animal takes its spines half as hard (Predator sting) |
| speed | cells moved per step of foraging, fleeing or going off in a direction (1 to 4; grazers move half as far again, rounded up: 2, 3, 5 or 6) |
| sight | how far it sees: 2 cells, and 3 more for each step of the trait (2 to 23 cells). Its sight also sets how far its herd, its crowding and its territory reach |
| fertility | high: cheap, frequent breeding and weak, hungry young; low: costly, rare breeding and well-fed young |
| litter | young per birth: 1 to 6, before the small-body boost. A high litter trait and a small body both make litters bigger, half each, averaged over both parents, so small animals tend to have large litters and large animals small ones. On top of that, the smallest bodies have twice as many young, falling evenly to the usual number for the largest (up to 12 in all). Each young is born on a free cell up to two cells from its mother (the sides first, then the corners, then the ring beyond), and the litter shares the energy its parents give. Parents conceive no more young than they could start on 15% of a stomach each (Newborn food): what they could give is a quarter of each stomach, the most any parent pays to breed, or what each has left if less, whatever they actually pay, so neither a fertile parent's cheap breeding nor a hunter's halved cost shrinks its litters, while a hungry pair has fewer young. Up to ten young; each parent pays at least its half of what they need, 5% of a stomach each, so every young starts fed and a large litter costs what it is worth. |
| carrion | how much of carrion's meat nourishes it: 20% at 0, all of it at 7. Above the middle of the range it makes a carrion-eater, more of one the higher it is, up to fully at 7: it craves the carrion it sees (Carrion craving), shies from hunting and attacking (Carrion aversion), and scales into a coward, breaking when struck at least as often as it is a carrion-eater, so always at 7, as an omnivore does (Carrion cowardice) |

There is no metabolism gene: size sets it. The smallest bodies run at 1.6 and
the largest at 0.6. A faster metabolism empties the stomach sooner, heals
faster, and makes for a shorter life. Size also sets the stomach, from 0.5
(smallest) to 2.0 (largest). Hunger is how empty the stomach is, so the same
meal fills a small animal four times as much as a large one, and a large
animal needs more food in all. Its slower metabolism offsets part of that: a
bare large body eats about 1.5 times what a bare small one does each turn, not four
times. Strength, speed, weapons and armour all add to the burn.

Two animals can mate only if they are the same species and each is willing to
breed with the other. An animal shuns a partner whose ring colour differs from
its own by more than 5% (25% for a predator), or whose dot colour differs by more than 20% (the mean
difference of the four CMYK bytes), or whose social background differs by more
than its kinship accepts (see above). Tolerance widens all three: at the greatest
tolerance it accepts any ring, dot and social background of its species. The same
test decides who counts as its herd. A child's ring is a
single-point crossover of its parents' rings; a predator's young are always predators, whatever the crossover or a mutation would make them. A child's dot comes from its
starting mind (see above); its two tolerance flags come together from either
parent. The home temperature travels with the ring. With the world's body
mutation chance (25% by default), one bit of the ring flips, which may change
the home temperature; the tolerance flags never mutate. A flip may change the species too, but only along
the chain of evolution, grazer, omnivore, predator. A flip that lands on predator makes any animal a
predator; otherwise a flip of the species' lower bit moves it a step up the chain (grazer to omnivore,
omnivore to predator) and of its upper bit a step down (predator to omnivore, omnivore to grazer). A step
off the chain is undone, so plants always stay plants and animals never become plants. A predator born of other parents brings what they learnt of eating and uprooting to hunting
and feasting: its brain starts out valuing hunting as its parents valued eating grass, and feasting as
they valued uprooting.
Minds blend, but experience and mind mutations keep them varied.

This is genome version 3. Version 2 had no social background: its dot held
sociability, empathy and tolerance in place of memory, resilience and moodiness.
Version 2 exports are converted when imported: their ring and climate genes are
kept, sociability, empathy and tolerance move to the social background, and the
traits version 2 lacked (memory, resilience, moodiness, loyalty, dominance, trust
and kinship) start at 3, with no neurodivergence. Their dot colours change,
because the bits move. Version 1 held eight four-bit traits (0 to 15) per
word, before empathy, tolerance, litter and carrion existed. Older exports,
which carry no version, are converted when imported: each trait is scaled to 0
to 7, and empathy starts at 0, tolerance at 3, litter at 0 and carrion at 0,
the version 3 traits at 3, with a medium home temperature and neither tolerance.

The inspector shows all three words in hex and every trait, the mind's beside
the social background's. Temperament traits appear
inborn and as they are now, coloured green where experience has raised them
and amber where it has lowered them. It also shows the body's energy burn,
its climate (home temperature, the temperature where it stands, coloured amber
one step from home and red two, and what it tolerates), hunger, health, age
against expected lifespan, generation, children, last action, the odds its brain
gave each option, and for a corpse, the meat left on it.

## Seasons and Climate

The map is banded into climate zones across its height. The top and bottom 20%
are arctic, cold, and their outermost 2.5% are polar caps, very cold all year
round whatever the season (Polar caps); a strip of 20% across the middle is the
equator, hot; the two bands between are temperate, medium. The top and bottom edges of the map are
barriers: nothing walks, flees or is born across them, and seeds that would fall
beyond them are lost. East and west still wrap round. The world is light grey, each row tinted
by its temperature now: reddish brown where hot, green where medium, white where
cold and light blue where very cold, and part way between two while the weather
turns.

A year is 360 turns by default (the year length setting), in four quarters,
starting mild. Through the second quarter (the northern winter) everything from
the top of the map down to and through the equator is a step colder: the arctic
very cold, the temperate zone cold, the equator medium but for a strip of 5% across
its middle, which stays hot through both winters (Winter equator). The third quarter is
mild again. Through the fourth quarter (the southern winter) the same happens
from the bottom up to and through the equator. The polar caps never change. The sidebar shows the year and season.

The weather doesn't change all at once: it comes in waves, one in each climate
zone, all setting off together as the season turns, so no zone waits on another
and an animal can always see where the weather is headed. In each zone a winter
washes in from the side toward its hemisphere's edge (the top of the map for the
north, the bottom for the south) across to the side toward the equator, and as it
ends ebbs back the way it came. Each wave crosses its zone in bands of two rows
over the first tenth of each season (9 turns of a 360-turn year), faster than an animal can
flee it, so only those that set off before it arrives escape it (the Season turning rule; it was
half a season, slow enough to outrun). At its front four bands stand part way
between the old weather and the new, each a quarter step further on than the
band ahead of it, so the change comes as a staircase an animal can see within
its sight: winter behind the front, the old season ahead of it. Each band turns
by a quarter step as the front moves on by one. The zones' colours turn with the
wave, and so do its effects: an animal half a step from home needs half the
extra food and loses half a step's health, and grass grows in a row, and dies
back in it, by how medium the row is. The world's first season starts settled.

Every Bitzkey has a home temperature. An animal a step or more from home needs
twice the food, and loses 2.5% of its full health a turn for each step it is from home:
2.5% one step away, 5% two and 7.5% three (the Climate damage rule; it was 1% until harsher
seasons were wanted to make leaving early worth learning, and once 10% a step,
which killed most animals a season's turn caught away from home before they could move). It may freeze or overheat to death. An animal that starves while out of its
temperature has died of exposure too: it froze or overheated. Heat tolerance
halves the penalties of heat, and cold tolerance those of cold: a cold-tolerant
animal two steps too cold needs half again as much food rather than double, and
loses 2.5% of its health a turn rather than 5%. Tolerance
is rare at the start: each random starting genome carries each flag only one
time in ten, so hardiness has to spread by breeding. Plants ignore the tolerance
flags. Nearer another temperature than its own, a plant neither regrows nor
seeds; a step from home it gives grazing animals no food, and too hot it needs
twice the food itself. One step from home doesn't hurt a plant, but two steps
cost it 20% of its health a turn and three 30%, rising steadily between one step
and two while the weather turns. The plants spawned at the start are at home in
their zone's usual temperature (an imported plant keeps its own). Every other starting or imported group appears where its home temperature is usual, so it starts at home: a very cold home, with no zone of its own outside the winters, starts in the arctic. The Spawn at home rule (World group) turns this off, so groups appear anywhere. A seed lies in the ground 10 turns
before it can sprout. One that falls where it is the wrong
temperature waits there, and sprouts when its cell is nearer its temperature than any other and
has room, for up to three quarters of a year; after that it dies. Grass grows
only where it is medium right now, so in a winter it grows on the equator and
stops in that hemisphere's temperate zone. Where it isn't medium, standing grass
dies back gradually: each cell loses a tuft every 20 turns, the cells taking
their turns in step.

Animals sense the temperature: how far it is from home, as they feel it, and how
close the nearest row in sight is where they would feel at home. They also sense
how far north or south they are, and how much warmer or colder than home it is
where they stand and at the edge of their sight to the north and to the south.
These read the zones' colours, so they change gradually as the weather turns,
and an animal can feel a winter setting in before it bites. Being out of their temperature makes them uneasy, strongly so, and the more so the further from
home (half as much for the hardy), so a step back toward home is felt at once and their brains soon learn to keep to it.
Away from home they are uneasier still the further the nearest row they would feel at home in lies, however far beyond
their sight (Homeward unease), so every step toward it is felt, not only the last.
They can head north or south, but there is no action that seeks home out for
them; any migration emerges from what their brains learn to do with those senses
and the time of year. Foraging never leads further from home: an animal ignores
grass in rows whose temperature is further from home than where it stands, and
doesn't herd toward a herd whose middle is. A predator's pack keeps to its temperature the same
way: it doesn't court a partner, or hunt prey, in a row whose temperature is further from its home
than where it stands, so chasing and courting no longer lead it off to freeze or overheat.

The zones' sizes, how much of each season a wave takes to pass and how wide its bands and front are, the food and health penalties, how much food a plant gives out
of its temperature, how fast grass dies back, how long seeds wait, how common
tolerance is at the start and how much it spares are
advanced rules (the Climate group).

## Behaviour

A turn runs in two halves. First every Bitzkey, in a shuffled order, lives
through it: it ages, grows hungrier, weathers the climate and may die of it,
gives birth when its young are due, and plants grow and seed. Then every animal
left thinks at once: all sense the world as it stands, all their brains weigh the
actions open to them and learn from how their last choices turned out, and all
choose. Last they act, one at a time in the same shuffled order, each finding the
world as those before it left it. So a plan can go stale: grass another ate
first is gone ("found it eaten"), prey that fell or fled is missed, and a mate
that died, moved off or bred with another first leaves its suitor spurned. What
herd-mates learn from watching each other is learnt at the end of the turn,
all together. Thinking at once is what lets every brain be worked as one batch.

Grass is sown at random, and regrows each turn (see grass density above). On and around a corpse, the
cell it lies on and the eight cells next to it, grass grows twice as fast
while the body remains.
Grazers and omnivores eat grass, and browse plants; predators can do neither.
The two have separate niches: grazers are the specialists, getting a quarter
more than the food in grass and plants (1.25 times), while omnivores get half the
food in grass and three quarters of the food in plants, and live on meat.
Omnivores and predators may eat carrion; grazers never do.

A dead animal isn't carrion straight away. Through the turn it died and the
three after it is a fresh kill: it doesn't rot yet, and an omnivore or predator may eat
it if it is something it lives on (grazers, for an omnivore; grazers and
omnivores, for a predator). A body is eaten in portions, one for each size class: from one for the
smallest to eight for the largest. Each portion of a fresh kill is a whole feast, three times the meal
of making the kill, so a large body feeds many, and feasting doesn't count as scavenging, so the carrion
trait plays no part. After that it rots as carrion, its meat eaten a portion at a time, each its share of what is
left. A body is gone once its last portion is eaten, whether feasting or scavenging, or once it has
rotted away.

Plants stay in place and photosynthesize. Every animal carries its own small
neural network, and that brain chooses everything it does. Each turn it senses:

- its hunger and health
- grass underfoot or in sight, and plants in sight (grazers and omnivores only)
- how close the nearest carrion in sight is (omnivores and predators), and the
  nearest fresh kill it would eat (likewise)
- a willing mate beside it, and how close the nearest one in sight is
- prey in sight, how close the nearest is, whether it carries young, and how hard it could hit prey beside it
- hunters in sight, how close the nearest is, and how dangerous one beside it is
- whether the animal that just struck it is still beside it, and how hard it could strike back
- its herd: how many herd-mates are close by, how near it is to the herd's middle,
  whether a herd-mate close by has just been struck or alerted it to a hunter, and
  whether a pack-mate has alerted it to a fresh kill
- whether it has lately seen a herd-mate succeed or fail (see below), fading
  turn by turn
- its own kind and other species close by, and its own age
- the time of year, as two hands of a clock a quarter of a year apart, so it can
  learn the seasons' rhythm and what each brings
- how far the temperature is from home, and how close home weather is (see
  Seasons and Climate)
- which way, north or south, the nearest row where it would feel at home lies,
  and how far, however far beyond its sight
- how far north or south it is, and how much warmer or colder than home it is
  here and at the edge of its sight to the north and to the south
- how fresh its memory of where it last fed is, and how close it is to where it
  was last struck, while it remembers
- how well fed its home band is: plants per grazer, prey per predator, or both, half each, for an omnivore
- how many of its own young, still infants, are within its sight
- its current temperament, mind and social traits
- its own genome: each of its dot's and social background's traits as born, each
  of its ring's traits (species among them) and its climate genes

That is 91 senses in all. Its brain has a hidden layer for each point of its
inborn intellect, at least one and up to four, each of 38 neurons; beyond four,
each point of intellect adds five neurons to every layer instead. So the dullest
think with one layer of 38 neurons, and the brightest with four layers of 53.
Plants have no brain.

The cleverer an animal, the faster it learns, three ways, while the least
intelligent learn just as they always have. Its learning rate rises with its
intellect, and then by up to as much again for the brightest (the Learning boost
rule); at the brightest it is 0.027 a choice before that boost (doubling it made
brains unsteady rather than quicker, and 0.045 left values overestimated, predators' most). A predator can be made to learn faster still (Predator learning, an omnivore
getting half the bonus), but by default learns no faster: faster, its values ran away. Food that takes a kill is harder to learn than grass: any
animal learns twice as much from a success choosing to hunt, attack or feast (Kill
food learning), and half as much from a success choosing to eat, forage, browse or uproot
(Plant food learning), whether its own choice, one it replays or one it sees. These weights,
and those for breeding and courting, make a success more, or less, memorable: a choice that
turned out badly is learnt from as any other, so a fruitless chase doesn't teach a hunter
five times over to give up hunting. Each turn it re-learns from its most recent experiences, two for each point
of inborn intellect, so the brightest go over their last fourteen while the
dullest replay none (Replays per intellect). And it traces an outcome further back
through the choices that led to it: its trace decay runs from 0.7 for the dullest
to 0.95 for the brightest (Trace decay, smartest). Replaying costs time: in a busy
world it can double the time each turn spends learning.

It then picks one of the actions open to it: eat, forage, browse, uproot,
scavenge, feast, rest, court, breed, hunt, attack, defend, flee, herd, go north,
go south, go east, go west, walk north, walk south, walk east, walk west, alert or
revisit. Revisiting heads back toward where it last fed, while it remembers where
that was (see memory, under Genome). Alerting tells its herd within its sight: a grazer that sees a hunter raises the alarm, so its
herd-mates may flee that hunter from up to six cells away; a predator that sees a
fresh kill points its pack to it, so they may come to feast though they don't see it themselves.
Alerting pleases it a little (Alert joy), but only when a herd-mate was near enough to take it in. Two instincts sit on top of what a brain learns: a
predator's bloodlust makes a fresh kill in sight hard to pass by (Bloodlust), and an animal with a
carrion trait above the middle craves carrion it sees, the more the higher the trait (Carrion craving). It
isn't told which actions are open; it picks among them by the values its brain
gives every action. Each go heads straight that way as far as its speed allows,
and each walk a single cell: north and south stop at the edge of the map, while
east and west wrap round it, so they are always open. Moving costs food: a walk
saves half the turn's burn, as resting does, while any move further than a cell, a go,
a chase, a flight, or a way to food, a mate, its herd or where it last fed, costs a quarter
of a turn's burn more for each cell beyond the first (Go effort), so a fast grazer going six
cells pays two and a quarter turns' burn. Setting off in a direction, going or
walking, is how an animal explores: it pleases the curious, though only if it got
somewhere (not into the map's edge or a full square), and finding grass
where none was seen, or walking into a hunter, shapes its curiosity. A new brain
starts out valuing the eight headings a little below its other actions, so an
untaught animal doesn't march away from its range, its herd and its food before
it has learned where to go. What a walk saves and what going far costs are
advanced rules (the Bodies group). Foraging walks toward the
nearest grass in sight; a tuft sates 18 hunger, about a tenth of a kill. Grazers get 1.25 times the food in grass and plants; omnivores half of grass and three quarters of plants. Browsing walks to the nearest
plant in sight and, once alongside, takes a bite: 70 hunger whatever the plant's size, far
more than grass. Each bite costs the plant 15 health, so a plant is eaten in portions in
proportion to its size, 4 bites from the smallest and 10 from the largest, and
plants regrow 2% of their health each turn. A bitten plant can't be bitten again
for 8 turns, so browsers move on, but enough bites in time kill one, and a dead
plant leaves no meat. A desperate grazer may uproot a plant beside it instead:
only once it is starving (80 hunger or more) and food is scarce about it (less
than the food it would need to breed), but whether the plant has recovered from
its last bite or not. Uprooting feeds it half a feast on a body of the plant's size and costs the plant
45 health, three bites at once, which often kills it; an armed plant still
strikes back. The hunger, the meal and the damage are advanced rules (the Food
group). Plants spread by seed: each turn a well-fed plant has a 3%
to 10% chance, by its fertility, of sowing a seed of its own kind, with
the usual chance of mutation, on a free cell up to five cells away. Seeding
costs it 20 hunger. A seed doesn't take where more than three plants already
grow within two cells. There is no limit on the number of plants, but they
compete for light: each plant growing within three cells cuts a plant's chance
of seeding by a third, and with three or more around it, it does not seed. Scavenging walks to the
nearest carrion in sight and, once alongside, eats a portion of it; the carrion
trait decides how much of that feeds it. Feasting walks to the nearest fresh
kill in sight and, once alongside, eats a portion of it. Courting walks toward the nearest
willing mate in sight, but breeding is still a separate choice once they are
within two cells of each other (Mate reach). A predator ready to breed feels a mating
call: it values breeding more when a partner is in reach, or else courting a mate it sees, on top of
what its brain has learnt (Mating call), and when it sees none it seeks a willing predator up to twice its
sight off (Mating call range). Omnivores feel a milder call (Mating call, omnivores: 0.25). Predators
and omnivores pay half the usual breeding cost (Breeding cost, hunters). Hunting chases the nearest prey in sight. Predators see prey three cells further than they see anything else, and chase it a step further each turn. They also have an instinct to hunt: whenever hunting or attacking is open to a predator, it values each more the hungrier it is, on top of whatever its brain has learnt, so even a newborn, or an animal whose genes have just made it a predator, goes after prey when hungry. Omnivores prey on grazers and predators; predators prey on grazers, omnivores and other predators. An omnivore hunts predators only as a last resort, while no grazers, plants or grassy cells are within its forage radius (Omnivores' last resort), and a predator's infant with a grown predator of its kind within two cells (Nursing reach) is guarded: no hunter sees it as prey. Before this, swarming omnivores ate most predator litters, holding predators to about a dozen however many prey there were. Predators whose ring colours differ by no more than 25% (the mean difference of the ring's four CMYK bytes) are one kind: they form packs together, and breed together as readily as other animals do within 5%. Hunting another predator, of any kind, is cannibalism: a predator only sees another predator as prey once its hunger, as a share, exceeds its reluctance, the average of its current empathy, tolerance and patience. The ruthless will eat other predators at any hunger; the gentlest never will, unless predators glut the world: above 600 alive (Predator glut), every predator's reluctance to hunt other kinds wears away, to none at 1,200, and past that even its own kind become prey, more readily as numbers climb, fully at 1,800. Predators turn on one another the same way as their prey die out: once grazers and omnivores together number under thirty for each predator alive (twice Rare prey), their reluctance to hunt other kinds wears away, to none at fifteen for each, and below that their own kind become prey too, fully with none left. Killing its own kind pains an empathetic predator as much as its own species' pain, rather than the reduced pain it feels for others. Predators and omnivores are territorial. An animal's territory is the cells within its sight. Rivalry is only within a species: a predator counts other predators in its territory as rivals for the feeding ground (omnivores are prey, not rivals), and how many it will share it with depends on its current empathy: one for the least empathetic, rising to five for the most. Each predator beyond that raises its rivalry by a third, up to full rivalry three past its limit. A predator senses its rivalry, finds an overcrowded territory uncomfortable (less so if tolerant), and feels relief, as well as any joy in fighting, at killing a rival there. Crowding by other predators also wears down its reluctance to turn on other predators: in a territory fully overcrowded with predators, even the gentlest will fight them for the feeding ground. Omnivores are territorial at half the strength. They count only other omnivores as rivals; predators are prey to them. They tolerate two rivals for the least empathetic, rising to ten for the most, and their rivalry tops out at a half. An omnivore only turns on another omnivore when its territory is overcrowded with omnivores, and then only once its hunger exceeds its reluctance as rivalry wears it down, never by more than half. A hunter that reaches
its prey pounces, striking in the same turn, and attacking strikes prey
alongside. A kill feeds the hunter 185, plus up to 200 more for the largest prey, which fills most stomachs; a blow
that doesn't kill feeds it 5. Blows are strong enough that the best-armed, mightiest predator kills the
largest, best-armoured grazer in three. A predator enjoys a little every blow that goes its way, dealing more than its victim's spines
deal back (Attack joy), and
feasting on a fresh kill more than the kill itself. Whatever eases an animal's hunger, eating, hunting, feasting or scavenging, also pleases it by as much again as the hunger it eases, beside what being less hungry is worth (the Sated joy rule), so every meal teaches more; resting and walking only burn less, and ease no hunger. Striking is hard work: each blow burns 0.2 food per point of damage it deals, so a heavy hitter pays more for each blow. Hunger figures for meals are for a middle-sized
stomach: a small animal gains twice as much, a large one half. Defending is open only to an animal struck since its last turn whose assailant is still beside it, and to a grazer only with strength above 5: it strikes back with the usual force of its size, strength and weapons against the assailant's size and armour. It costs effort like any blow and feeds the defender nothing, but it can kill. The aggressive
enjoy it as they enjoy attacking, the empathetic and the tolerant less. Fight or flight: whenever an animal is struck, by an attack or a counterblow, it checks its morale against its current fear. A fully fearful animal breaks 30% of the time when barely hurt, rising to certainty near death; the fearless never break. One that breaks loses its chance to defend: on its next turn it flees its assailant, whatever its brain would have chosen, and grows a little more fearful. Omnivores are cowards: an omnivore struck always breaks, whatever its fear (Cowardly omnivores; 0 gives them the morale check again). A predator feels only half of its fear's effects: half the dread of hunters, half the chance of breaking, and half the pull to keep close to its herd. Primal fear: an animal that sees a hunter stronger than itself (weighing the share of each other's health one blow would take, with size, strength, weapons and armour all counting) may be overcome. Each turn the chance is how lopsided the fight would be, so an overwhelmingly stronger hunter overcomes it every time; predators are never overcome (Primal fear, predators: 0), so they never panic or freeze. Sizing up the fight, a predator counts as two steps larger and two steps stronger than it is, whether it is the hunter feared or the animal judging, and its health as the larger health that larger body would have. Once overcome, it stays so while any hunter is in sight, growing a little more fearful: each turn it panics, fleeing the hunter whatever its brain wants, or freezes and does nothing at all (no rest, no healing), certainly at the lowest intellect, the chance falling to none at the highest. With no hunter in sight it is calm again. Plants can't flee, so their only answer is to fight: a plant with any weapons strikes back at each animal that browses it. Its thorns come mostly from its weapons (80%) and a little from its size (20%), never its strength: at most 15 damage, less against a large or armoured browser. Fleeing runs from the nearest hunter in sight, and speed decides who wins a
chase. It is only worthwhile when the hunter is within three steps (or a
herd-mate was just struck close by): a hunter further off is watched, as a
sense, but not fled. Among herd-mates the range shrinks, as the herd keeps watch
together: to half at full herd instinct with three herd-mates close by. Resting halves the hunger a
Bitzkey gains that turn and heals 1.5% of its health pool, three times the usual
rate. Both rates are scaled by metabolism, so small animals heal faster. Better-valued actions are more likely, but no choice is certain. Mind
traits set what it enjoys and dreads. The brain learns online from what each
choice brought about: whatever raised its health (fullness counting with it) or
its mood was a success, and whatever lowered them a failure, weighed by how much.
Both can happen at once, and the bigger counts for more. Its mood is the joys and
pains it feels, from breeding, fighting, company, its herd, fear, unease, rivals
and the pain of others; what counts is how much better or worse it feels, not
how good it feels. Death is heavily punished. What comes of a choice also reaches
back to the choices that led to it (TD(lambda) learning): each weight keeps a
trace of how recently and how much it shaped the brain's choices, fading each
turn by its discount (0.7 for the least patient, up to 0.99 for the most: Discount, most
patient) times its trace decay, so a death or a
meal is credited to the run of choices before it, not just the last one. How far
back it reaches is an advanced rule (the Mind group), as is how reluctant a new
brain is to set off in a direction. A child's brain is as deep and as wide as
its own inborn intellect makes it. Where both parents have a layer, the child takes
each of its neurons whole from one parent or the other, or from the one that has it
where only one parent's layer is that wide; where only one parent has the layer,
from that one; deeper still, its layers start out passing their input on. Neurons
neither parent has start idle, until mutation stirs them.
Each action's output comes from one parent or the other, all with small
mutations. A child also inherits its parents' learned shifts in temperament, so
learned habits and moods pass down the generations.

Animals also learn from their herd, or pack. An animal sees what happens to its
herd-mates within its sight: a herd-mate's kill is a success, and a sudden death,
from a blow or exhaustion, a failure. A slow death, of old age, starving, or freezing or
overheating, is no one choice's doing: the dying animal's last choice isn't blamed with the
death penalty, and its herd learns nothing from it, as the health it lost on the way has
taught its lessons already. The witness
learns from it as if it had made that herd-mate's last choice itself, in the
same situation, toward the death penalty for a death or the fullness the killer
gained for a kill. It feels it too: a kill lifts its mood and a death lowers it,
by how much the herd-mate gained or lost. How much it takes in is its empathy
times its intellect, more for the trusting and less for the wary, so the dull or
callous barely notice, while the bright and caring learn from every loss. It also senses what it saw, fading by a fifth each
turn. How strongly this teaches, how much it moves the mood and how long it is
remembered are advanced rules (the Society group).

Breeding is the greatest pleasure there is, so every species learns to seek it: 4 for each parent
(Breeding joy), three quarters of it whatever its libido. Courting is the next: 2.5 for a courting
move that brings a ready, willing partner within reach to breed (Courting joy), so it is learnt
for what it leads to and can't be farmed by chasing a mate it never reaches. Courting is open, and
pays, only toward a partner it could then breed with: both ready, room in the world, not too crowded
by its own kind, food enough about it where that limits breeding, and, for a grazer, plants enough in
its home band; only being close enough is left for the courting to bring about. The strongest pleasure
besides is a predator's feasting, at 1.5. Neither costs anything when it fails: a partner that has
gone, or bred with another first, leaves no penalty. Breeding and courting can also be learnt
from more strongly than other choices (Breeding learning, Courting learning), but not by default:
learning them harder unsettles a brain for everything else: over 40 seeds, 13 worlds died out
by turn 600 when they were learnt 8 and 6 times as strongly, and 22 with the pleasures too,
against 8 without. With these pleasures, predators lasted in twice as many worlds as before, ten
times as many of them, over 40 seeds, with no more worlds dying out.

Each parent pays 10 to 25% of its stomach to breed, less the more fertile it is, and every animal pays half of that: grazers (Breeding cost, grazers) as hunters do, so 5 to 12.5%. A species grown scarce breeds more cheaply still: below 100 living (Scarce population), its breeding cost falls evenly, to a third of the usual at 10 or fewer (Rare population, Rare breeding cost), counted at the start of each turn. Grazers, the prey everything else lives on, get three times the help: theirs falls from 300 living (Grazer scarce population) to a ninth of the cost at 10 or fewer (Grazer rare breeding cost). Animals
only breed, or court, when both partners can afford it (the Breeding reserve rule can make
them keep hunger to spare as well; none by default). After breeding, grazers and omnivores rest 5 to
20 turns, by fertility (Breeding cooldown), and predators half that, 3 to 10 turns (Breeding cooldown,
predators). An animal is only fertile in heat: a stretch of the turns before its next winter, the winter
of the row it stands in. The least patient are in heat from 0.6 of a year before that winter until 0.1
before it; the most patient, with more foresight, from three quarters of a year before it until a quarter
before, so their young are grown before the cold comes. No animal is in heat through its own winter, and at
the polar caps and the equator's heart, where no winter comes, an animal always is. Each species has its own
heat (Heat ends, Heat, by species, and Heat foresight). Its sense of being ready to breed means rested and in
heat. A hunter that kills a mother carrying young feels remorse (Pregnant remorse), the more so the more
empathetic it is. The litter
shares exactly the energy its parents gave, so a large parent gives more food
than a small one, and the same food fills a small newborn more. Breeding starts a
gestation: the parent that chose to breed conceives the litter at once, and
carries it for 10 turns if it is the smallest body, up to 50 for the largest,
evenly between; it can't breed again until it has given birth, and if it dies
first, its young die with it. The young's genes, minds and brains are fixed at
mating. At birth, each newborn needs room (one space plus its size) in its mother's square or one within
two cells of it, nearest first; those with no room are lost, and if a population cap is set, the world holds at most that
many living animals (plants don't count toward it). Crowding also
limits breeding: for each other animal within sight of a would-be parent,
besides its partner, of its own kind, its litter shrinks by a tenth, and with ten or more around,
it does not breed; only animals of its own kind count, for the litter and the limit alike, so other species and
other kinds of predator never touch breeding (Crowding limit; 0 switches it off). Food can also limit breeding, though by
default it doesn't (Food needed: 0). When it is set, before breeding, an animal
weighs the food it lives on within six cells against the mouths that share it,
itself included. Grazers count grass and plants (two meals each), as well as
they digest them; omnivores count grass and plants as they digest them, and
grazers as prey (ten meals each, as meat fills about ten times what grass does);
predators count grazers and omnivores as prey (ten meals each), though predators need no particular
supply to breed (Food needed, predators: none by default). A hunter's pack-mates count as
half a mouth each, as a pack shares its kills (Pack mouths). Grass is judged
by what will grow back nearby over the next 150 turns, plus a quarter of what is
standing now, so a crowd doesn't breed on a stock it is about to eat. Every
animal of its own species is a mouth, every predator to a predator whatever its
kind, and another species eating some of the same food is half a mouth (an
omnivore to a grazer, or a predator to an omnivore). Below the food needed a mouth it
doesn't breed; between that and twice it, its litter shrinks in proportion. The same weighing
decides when a starving grazer may uproot a plant: only with fewer than three meals a mouth about
it (Uproot food). Grazers' breeding also follows the plants in their home band: the climate band of
their home temperature nearest them (the nearest arctic band for a very cold home). Its plants are
shared among the grazers at home there, and with fewer than a tenth of a plant each, a grazer doesn't
breed (Plants per grazer; 0 switches it off). Predators' breeding follows the prey in their home band:
the grazers and omnivores there, shared among the predators at home there; with fewer than 2 each, a
predator doesn't breed (Prey per predator). Omnivores' follows both foods in their home band, half
each: half the share of a full litter its plants would give a grazer, and half the share its prey
(grazers and predators) would give a predator, each counted per omnivore at home there; with less than
its need, the two together, it doesn't breed. Above its need, an animal's litters grow with its home
food, to a full litter at four times the need (Full litter food), so at just the need its litter is a
quarter of what its traits would give (at least one young); a predator's only at eight times the prey it
needs (Predator full litter), so at its need an eighth. An animal that can't breed can't court
either, so a hunter short of prey hunts rather than courting a partner it couldn't breed with. And a
hunter spares the last of its prey: every animal knows how many of its own species live, and while
its prey species, all together, number under 1500% of that, fifteen prey for each of its kind (Rare
prey), a predator or omnivore doesn't breed at all, so the line rises and falls with the hunters'
own numbers. At 50% predators ate every grazer and starved; at 1000% they settled at a few hundred
beside thousands of grazers, but once predators learnt to hunt well, long-running worlds still lost their grazers to them, so it was raised to 1500%. Sparse food was made this telling after
species hunted one another to extinction, breeding on as their prey ran out.
Over 80 seeds, predators lasted in 26 worlds rather than 15, 117 of them rather than 47, with no
fewer worlds keeping their animals. Grazers and predators also sense how well fed their home band
is, 1 at twice what breeding needs (home food); an omnivore senses 0. Other
kinds of predator may still be hunted, but they are rivals for prey, not a food
supply to breed on. Omnivores also switch prey: an omnivore only goes after grazers while they
make up at least half of the food it weighs within six cells (grazers
against plants), or while none of that food is near, so grazers that have become scarce
are left alone by omnivores, and don't flee one that isn't interested in them, so they can recover.
Predators don't switch: grazers are always prey to them, and omnivores too, but only one a predator
would beat, blow for blow, as each stands now; it leaves a stronger omnivore alone. Omnivores size up
predators the same way. A predator never hunts its own kind, and other kinds only when hungry past its
reluctance, which an overcrowded territory wears down by half (Rivalry wear), as it does an omnivore's
reluctance to hunt its own species; killing a rival is a smaller relief than it was (Rival kill joy, 0.5).
A predator's pack-mates don't count as rivals crowding its territory (Pack rivalry: none), so packs
gather and herd together without unease.
Territorial fights, between omnivores or between predators, go on regardless. Zero health is fatal, and so, in
time, is age. Hunger stops at 100%: an animal (or plant) that full hunger reaches is starving, and loses 10% of its
full health each turn until it eats or dies of it (Starving damage, in the Bodies group).

Parents are attached to their young. Every animal knows its mother and father. A parent is glad of its own infants within its sight, up to 0.3 with three or more (Parent joy), and grieves 2 when one of them dies, wherever it is (Parent grief), as much as its own death costs it; both run from half to one and a half times as much with its empathy, and both parents grieve. Neither tells a parent what to do: keeping close, defending its young, or breeding where and when they will live, is for its brain to learn.

Infants are cared for. Each turn, every grown animal that isn't starving (below 80% hunger: Nursing limit) feeds the hungriest infant of its own kind within two cells (Nursing reach) a turn's worth of its food, from its own stomach, so the infant grows no hungrier while it is cared for. Infants keep close to their kind: they value herding more (Infant herding) and herd once they are more than a cell from their herd's middle, and even a solitary omnivore's infant herds with other omnivores until it grows up. An animal is feeble in its infancy, the first 10% of its lifespan (a predator only the first 5%: Predator infancy), and in its old age, from 70% of it,
when it may begin to die of age (Feeble youth and Ageing, in the Bodies group). While feeble, every body
trait (size, strength, armour, weapons, speed, sight, fertility, litter and carrion) and its full health
are at half its own (Feeble share), rounding down, with all that follows from them: a feeble animal is
smaller, so it takes less space, eats less and burns faster, and it is weaker, slower and shorter-sighted.
Its lifespan stays set by its genes. Its health's share of its full health jumps as it grows out of
infancy or into old age, but that is no lesson: the brain learns only from the turn's real gains and losses. Each newborn starts with an infant's health; growing old, an animal's
health shrinks to its feeble full health. Plants are never feeble. The inspector marks a feeble Bitzkey
FEEBLE beside its age, and the designer shows a grown one.

Meat lasts twice as long as plant food. Each Bitzkey keeps track of how much of its fullness is meat, from
hunting, attacking, feasting and scavenging, and that share is digested at half the speed of the rest
(Meat digestion), so a predator fed on meat grows hungry half as fast as one fed on plants.

A map square holds 10 spaces (Square space, in the World group). Every Bitzkey, plant or animal, takes
one space plus its size (0 to 7), so up to ten of the smallest share a square, and one of the largest
leaves room for a single smallest one. Bitzkeys sharing a square are as close as can be, and meet, fight,
eat and breed there as they would alongside. A square with no room left stops births, moves and spawning:
an animal stepping toward a full square stops short of it, a mother bears her young in her own square while
it has room and then in the squares around, and groups spawned or deployed take the nearest squares with room.
Seeds still need an empty square. Plants and animals alike live about a year at the smallest body
and fastest metabolism, five years at a middling one and ten at the largest and
slowest, their lifespan rising steadily between, so it scales with the year
length. From 70% of its lifespan, a Bitzkey's chance of
dying of old age grows each turn: about half reach their lifespan, and almost
none live past 1.3 times it. Dying of old age is not held against the brain's
last choice. A blow or a long go costs an animal food, and one that leaves it
starved kills it there ("exhausted"); so does breeding ("bred to death"), and a
plant that spends the last of its food on a seed dies of it ("seeded to death").
Each is a death like any other: counted, learnt from by its brain and seen by
its herd. Should any other cost ever starve a body without its death being
recorded, it is recorded as "starved" at the start of the next turn, so no
body lingers uncounted. A Bitzkey that moved this turn leaves a trail: its
picture at half strength on each cell it passed through on the way, back to
where it set off (the shorter way round east or west). The trail lasts only
until the next turn. A dead Bitzkey is dimmed and crossed through in red, and fades
away as its corpse decays, vanishing when the corpse is gone. A dot big enough
to hold an icon (3px or more) is filled with its social background's colour, carrying its species' icon in the dot's
colour with a thin light grey outline: a tree for plants, a leaf for grazers, an apple for omnivores and a
drumstick for predators. A smaller dot is simply its colour. The inspector, the designer and the colour graph show them too. A corpse holds the energy its body had left: up to 30 meat, plus
up to 50 more for the largest. A starved body leaves nothing, and a killed one
half, as the killer has already eaten. The meat rots away over the 30 turns
before the corpse fades. An animal's body that rots away entirely, rather than being eaten to its last
portion, has a 10% chance (Corpse plant) of leaving the seed of a new plant where it lay: a random plant
genome, at home in the temperature there, that sprouts as any seed does, once it has lain 10 turns and
the cell is free and not crowded with plants. Brains, temperament, learning and statistics are all
part of turn history, so rewind and replay are exact.

### Herds and Packs

Grazers herd and predators run in packs; omnivores live alone, in neither herd nor pack (Solitary
omnivores, in the Herding group; 0 lets them herd with their own kind again). A solitary omnivore has
no herd-mates: it never herds, hunts as a pack, alerts others or heeds their alerts, learns from what
it sees others do, or defends another, and it misses no herd, feeling no loneliness without one.
An animal's herd instinct is the
lower of its current empathy and patience, so only animals high in both herd
strongly, and every herd effect scales smoothly with it. Its herd-mates are
animals of its own species that it would accept as mates, so tolerant animals
herd with other kinds of their species, and clannish ones only with those of a
like social background. Predators only form packs with their
own kind, those within 25% in ring colour. Herd-mates are sought within its sight.

- Herding: the herd action moves an animal toward the middle of its herd-mates
  in range. The brain learns when to use it, and loyalty draws it there. Fear sets how close it keeps: a
  fearless animal is content up to three cells from the middle, and a fully
  fearful one only at the middle itself, so frightened herds bunch tightly.
- Comfort: a herd animal enjoys having herd-mates close by, and is uneasy with
  none in range; the more fearful it is, the more it feels both.
- Safety in numbers: herd-mates close by calm its fear of hunters and steady its
  morale. With three close by, a full-instinct animal is 60% less likely to break.
- Alarm: when a herd-mate within two cells is struck, the animal senses the alarm,
  and may flee whatever struck it even if it can't see a hunter of its own.
- Mutual defence: a herd-mate beside the attacker may defend the struck animal,
  striking back as if it had been struck itself, and enjoys defending its herd.
- Packs: a hunter strikes prey half as hard again, at full instinct, when a
  pack-mate is also beside the prey.

Herding never overrides territory. Pack-mates still count as rivals, so a pack
bigger than a predator will tolerate still crowds it, and still wears down its
reluctance to turn on its own kind. Herding never spares prey either: a predator
still hunts omnivores, and other predators when hungry, as before.

## Colour Graph

Press `G` for the colour graph. Time runs across the bottom. Each bar is a moment
in the run, and every Bitzkey alive then (plants included) is a slice of it in its
own exact colour. Slices are stacked by species, plants at the bottom, then
grazers, omnivores and predators, and each species' band is labelled, with its
icon, at the right end of the chart. Within a species, slices are sorted by hue,
so similar colours flow together from bar to bar. The SORT menu at the top sorts
each species by a trait instead: any of the ten mind traits, the seven social traits, the body traits, or
the climate genes (home temperature, heat tolerance, cold tolerance), taken from
the genome, lowest at the bottom (for temperature, hottest at the bottom). Slices with the same value are then
sorted by colour. A trait sorts only the Bitzkeys living now; the dead are
remembered by their colour alone, and sit beneath the living in each species,
sorted by colour. Colour value is the default. A bar's height is the
population. `Tab` steps between ring
(body), dot (mind) and social background colours. The timeline shades each winter quarter and marks
where each year begins, with a line up through the chart. The graph covers the last 250,000 turns (the Graph turns rule, under World); older turns, and the
genomes and deaths of the Bitzkeys only they held, are forgotten, so a long run's saves and death reports stay
light. It samples who was alive at evenly spaced turns, spacing them further apart as the run grows.

The graph is also a time scrubber. A playhead marks the present. Drag along the
timeline beneath the bars, or right-click (without dragging) anywhere on the graph, to move the
world to that turn; it pauses there, or at the earliest turn the history
keeps (the latest 100), if that is later. Space plays on through the recorded future, which stays
visible but faded. The mouse wheel zooms the time axis down to 10 turns,
keeping the latest turn at the right while the view reaches it, so the living
stay in view as the run goes on; once panned back into the past, it zooms around
the cursor instead. Over the bars, the wheel also zooms the height, as the map
zooms, up to 32 times, the population under the pointer staying put, so single
Bitzkeys in a crowded bar grow tall enough to pick. Drag with the right or
middle button to pan, in time and in height, as on the map; `Shift`+wheel, or a
sideways wheel, also pans in time. Zooming all the way out, or `Z`, shows the
whole run at its whole height again.

Hovering over a dead Bitzkey's slice shows how it died: the cause, and, when something killed it
(a hunter, a defender, a victim's spines, or a grazer eating or uprooting a plant), that killer's
number and species and what it was doing: hunting, attacking, defending, browsing, uprooting, or spines
(struck, its spines killed its attacker), as in "died: killed by #012 predator, hunting". A popup
beside the pointer shows what is known of it, as for the living: its genome (its colours, the traits it
was born with, its body as grown and its home temperature) and its lifespan, and, in place of how it
fares, how it died and on what turn. What it became over its life (its temperament, hunger and health)
went with it.

Pick Bitzkeys straight from the graph. Only the living can be picked, and one
that dies drops out of the selection:

- click a slice to select that Bitzkey; click empty space to clear
- drag a box to select every Bitzkey it touches, in every bar it spans
- hold `Shift` to add to the selection (`Shift`-click toggles one), or `Ctrl` to
  remove from it

A selected Bitzkey stays lit in every bar it appears in while the others fade.
Hovering over a living Bitzkey's slice shows a tooltip with its details, as the
sidebar shows them for a selected one: genome codes, temperament, body, hunger,
health, age and latest chances. The line under the chart also names it. A dead
Bitzkey shows only its ring, dot and social background colours there.

The graph and the map share one selection. Close the graph and the Bitzkeys picked
on it stay ringed on the map, while the sidebar shows no single Bitzkey and counts
how many are picked. Clicking a dot on the map selects just that one, on the
graph as well; a box dragged on the map selects every living Bitzkey in it, and
`Ctrl` adds to the selection or takes from it.

START A NEW WORLD WITH THE SELECTED, beneath the graph, starts again with the
living Bitzkeys picked: it copies them to the clipboard, leaves this world, clears
the import box on the configuration panel, pastes them in, and starts a new world
at once with the panel's settings, them alone imported.

Then press `C` or COPY to put the selection on the clipboard, or `S` or SAVE to
write a `.json` file into `exports/`. CLEAR empties the selection. Living
Bitzkeys export with their temperament and brain. Dead ones have no brain, and
those no longer on the board export as their genome only. With nothing
selected, COPY and SAVE take everyone alive.

## The Catalog

The catalog keeps your favourite Bitzkeys between runs, in `catalog.json` beside
`exports/`. Add one with ADD TO CATALOG (or `K`) under the selected Bitzkey, the
living ones picked on the colour graph, or the whole export list from its dialog.
Each is kept as it was when added: its genome, its current temperament and its
trained brain, along with what the inspector showed of it then (its hunger,
health, age, generation, children, last action and the chances it last gave its
options). The dead can't be added from the graph, their brains being gone. The
same Bitzkey added twice is kept once.

FROM CATALOG on the configuration panel opens it: every Bitzkey catalogued, listed
with when it was added, and the one last picked shown large beside its inspector
panel. Choose several as in Windows: click one to choose it alone, `Ctrl`-click
to add one or take it out, `Shift`-click to choose every one from the last one
clicked (`Ctrl`+`Shift` to add them), `Shift`+`Up`/`Down` to extend the choice,
and `Ctrl`+`A` for all. ADD TO IMPORT (`Enter`) adds every one chosen to the import
list, brains and all, REMOVE (`Delete`) takes them out of the catalog for good
after asking, DESIGNER opens the designer on the entry picked, a design applied joining the catalog
as a new entry (its genome alone, listed as "designed", with no brain, ready to import or deploy like
any other), and CLOSE (`Esc`) returns. Each entry is written as an export record, so `catalog.json` can also be
loaded like any export file.

OPEN CATALOG, in the footer beside EXPORT ALL SELECTED and at the bottom right of
the colour graph, opens the catalog from a running world: the same screen, with
DEPLOY in place of ADD TO IMPORT (REMOVE still works). The world waits while it is
open; CLOSE (`Esc`) returns to the map or the graph.

DEPLOY (`Enter`) closes the catalog and turns the cursor into a target over the map.
Each click there puts 5 of each Bitzkey chosen, each with its own copy of its
catalogued brain and temperament, within 2 cells of the cell clicked; hold the button
and drag to paint them, a group every 5 cells. A right-click or `Esc` stops. They join
the world between its turns, paused or playing; a world rewound lets its abandoned
future go, as a turn played afresh would.

## Export Format

A Bitzkey is written as its genome version and its three colour words in hex,
ring, dot, then social background. Animals also carry their current temperament
(mind and social traits) and their brain, so they pick up in the next simulation
where they left off:

```json
{"version": 3, "ring": "7421FF41", "dot": "637E4B01", "social": "0A1B2C3D",
 "mind": {"aggression": 9.412, "libido": 3.0, ...},
 "brain": {"senses": ["hunger", ...], "actions": ["eat", ...],
           "layers": [{"weights": [[...], ...], "biases": [...]}, ...],
           "output_weights": [[...], ...], "output_biases": [...]}}
```

Plants have no brain or changing temperament, so they export only their colours. Several Bitzkeys form a
JSON array, one per line; an animal with its brain takes about 13 to 31 KB, by how deep its brain is.

Import accepts single objects, arrays, several blocks pasted one after another,
and `#`-prefixed or lowercase hex. A record without a `"version"` is read as
genome version 1 and converted, so hand-written records for this version need
`"version": 3` and a `"social"` word; version 2 records, without one, are converted
(see Genome). A record without a brain gets a fresh one.
Brain weights are matched to senses and actions by name, so exports from an
earlier version still load: senses or actions they lack start neutral, and ones
this version no longer has are dropped. A brain saved with one hidden layer, from
before brains grew deeper, loads too. A saved brain is fitted to the depth and
width its Bitzkey's intellect calls for: deepened with layers that pass their input
on, or cut short, and widened with idle neurons, or narrowed. A saved world from
before brains grew wider loads with its brains fitted the same way. Every member of an imported group starts
with its own copy of the saved brain and the saved temperament, and their
children inherit both as usual. Temperament is read for the known traits only,
kept within 0 to 7; a trait it lacks starts at the gene's value. As in life, a
restored temperament slowly drifts back toward the genes unless experience
keeps it shifted.

## What the Bitzkeys Have Learned

Press `I` to swap the inspector for the insights panel; selecting a Bitzkey
brings the inspector back. The panel scrolls: use the mouse wheel over the
sidebar, or drag the scroll bar on its right edge. It shows, by species:

- Behaviour: how often each action was taken when it was possible, over recent
  turns, and the trait groups least and most keen to breed and least keen to
  fight, e.g. "high-metabolism grazers 4%".
- Life: how many are alive, and their average size, age, generation, hunger,
  health, children, herd instinct and territorial rivalry.
- Temperament: each mind trait's current average, green where experience has
  raised it above the inborn average and amber where it has lowered it.
- Keenest by action: for every action, the trait group most likely to take it.

DEATH REPORT, at the top of the insights panel, opens a report of every death over the last 250,000 turns,
gathered from the record of each death the graph keeps, back to the graph's start:
- the causes: a row for each cause, the most common first, a column for each species of the dead, and the totals;
- who killed whom: a row for each species of the dead, a column for each species of their killers;
- what the killers were doing: a row for each activity (hunting, attacking, defending, browsing, uprooting,
  spines), the most common first, a column for each species of the killers;
- the deadliest killers: the ten Bitzkeys with the most kills, each with its kills by what it was doing.

It keeps to the turn shown, so rewinding takes back the deaths after it, and a rewound world changed (by
a deploy) lets the deaths of the future it had go.

INSPECT BRAINS, beside the selected Bitzkey's heading, pauses the world and opens a matrix of the
standard situations its species can meet (the same as the brain report's): in each, its chance of
choosing every action open to it, with its temperament now. The likeliest in each row is marked with *,
and an action not open there shows as a dot. Scroll either table with the wheel or the arrow keys.

Press `B` to save a brain report to `reports/`. The report puts every living
brain in the same standard situations, such as a fed animal with a willing mate
beside it or a wounded animal beside a hunter, each with its own current
temperament. It averages the chance of each action by species and trait group
and ranks who is least and most keen. It also shows how far experience has
moved each species' temperament from its genes.

Eight of the situations probe the climate senses: too cold with warmer to the
south, and to the north; too hot with cooler to the north, and to the south; far
too cold; at home with cold coming from the north, and from the south; and at home
with the same all around. Each offers the moves north, south, east and west and
rest. Mirrored pairs tell a brain that follows the warmth senses from one that
has only a habit of heading one way: a brain that reads the weather picks
differently in each, while one that doesn't gives them all the same odds.

## Licence

Copyright (C) 2026 Phil Snelgrove. Bitzkey is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License, version 3 or (at your option) any later version, as published by the Free
Software Foundation. It is distributed in the hope that it will be useful, but without any warranty; see `LICENSE`.
