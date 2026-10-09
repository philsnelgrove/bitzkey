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
  sharing one genome, each group within two cells of a random spot. Every
  climate zone gets at least one group of plants: the plant groups take a zone
  each, from the top down, and if too few groups come up plants, the last animal
  groups become plants instead, though at least one animal group is kept.
- the grass density: the percent of cells with grass at the start (default 50).
  Each turn, 0.4% of that starting amount grows back, so density means the
  same at any grid size. Grass grows only where it is medium (see Seasons and
  Climate).
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
120 genomes are used, keeping their groups within the 600 population cap. Set
"Bitzkeys" to 0 to start with only your imports.

ADVANCED opens the advanced rules: 139 numbers that shape how the world plays
out, in groups (World, Bodies, Breeding, Feelings, Mind, Fighting, Territory,
Herding, Food, Plants, Climate, Society and Experiences). Among them are lifespans, metabolism, litter
sizes, food-aware breeding, the rewards and pains a brain learns from, blows and
thorns, territory and herd instincts, meals, fresh kills, plant seeding, the climate
zones, and how each experience reshapes temperament. Each rule has a short note, limits and a step. Choose and adjust
them as in the configuration panel; PgUp/PgDn jump ten rules, and the wheel scrolls.
Fractional rules take a typed decimal point, and the experience shifts take a minus
sign. Changed values show in amber. RESET puts the focused rule back to its default
and RESET ALL every rule. APPLY (`Enter`) keeps the changes, unless two rules
conflict (the smallest litter above the largest, equal metabolisms, or arctic
zones and an equator too wide to fit the map together); CANCEL
(`Esc`) drops them. The ADVANCED button turns amber and counts the changed rules.
They hold for every world started from the panel, `R` restarts included, until
changed again. A seed replays the same world only under the same rules.

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
that most stand out. Each card's title is a menu: click it to choose, from 43
statistics, what that card looks for, and the card shows the animal that is
highest by it (or lowest, for those that say so). By default the left column shows
the smartest, the largest, the oldest, the one with the most children and the
latest generation, and the right the strongest, the fastest, the best armoured,
the most fertile and the best fed. The choices last for the rest of the session,
through new worlds. They are:

- temperament, as born: the smartest, most aggressive, most amorous, most
  sociable, most curious, most fearful, greatest appetite, most patient, most
  empathetic and most tolerant; and the simplest, gentlest, bravest and most solitary
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

Each Bitzkey is drawn as an inner dot inside a 2px ring (a dot big enough for
its species' icon is black, with the icon in the dot's colour). Its 64-bit genome is
the two colour words together. The low 32 bits colour the dot and shape the
mind; the high 32 bits colour the ring and shape the body. In each word, the
four bytes from most to least significant are cyan, magenta, yellow and black.
Each word also holds ten three-bit traits, each 0 to 7, read from its
least-significant bits. The top two bits of each word hold the climate genes,
and like the rest they shade the colour: in the ring, the home temperature (0
hot, 1 medium, 2 cold, 3 very cold); in the dot, the heat tolerance flag (bit 30)
and the cold tolerance flag (bit 31). See Seasons and Climate.

**Inner dot: mind, emotion and social life.** These traits start at their
gene values and are then reshaped by experience:

| Trait | Effect | Experience that shifts it |
|---|---|---|
| aggression | pleasure of a fight, more for a kill | kills raise it; wounds and stings lower it |
| libido | pleasure of breeding | mating raises it; breeding into starvation sharply lowers it |
| sociability | pleasure of company | mating raises it; being wounded alone raises it |
| curiosity | pleasure of setting off in a direction | finding grass where none was seen raises it; walking into a hunter lowers it |
| fear | dread when hunters are adjacent | wounds, stings and ambushes raise it; kills lower it |
| appetite | how much hunger hurts | near-starvation raises it |
| patience | how much it values the future (its learning discount) | meals earned by foraging raise it; starving lowers it |
| intellect | how deep its brain is (twice its intellect in hidden layers, at least two), how fast it learns (its learning rate, boosted for the bright, how many recent experiences it replays each turn, and how far back it traces an outcome), and how much experience can reshape the mind | fixed |
| empathy | pain at seeing an animal wounded or killed nearby; more for its own kind, double for a death. An empathetic hunter feels some of its victim's pain and enjoys fighting less. | fixed |
| tolerance | ease among other species: they count as company, hunters are less frightening, and attacking gives less pleasure. The intolerant are uneasy whenever other species are near. The tolerant also accept mates that look less like themselves. | fixed |

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
| speed | cells moved per step of foraging, fleeing or going off in a direction (1 to 4; grazers move half as far again, rounded up: 2, 3, 5 or 6) |
| sight | how far it sees (2 to 9 cells) |
| fertility | high: cheap, frequent breeding and weak, hungry young; low: costly, rare breeding and well-fed young |
| litter | young per birth: 1 to 6, before the small-body boost. A high litter trait and a small body both make litters bigger, half each, averaged over both parents, so small animals tend to have large litters and large animals small ones. On top of that, the smallest bodies have twice as many young, falling evenly to the usual number for the largest (up to 12 in all). Predators have half as many young, rounded up. Each young is born on a free cell up to two cells from its mother (the sides first, then the corners, then the ring beyond), and the litter shares the energy its parents give. |
| carrion | how much of carrion's meat nourishes it: 20% at 0, all of it at 7 |

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
its own by more than 5%, or whose dot colour differs by more than 20% (the mean
difference of the four CMYK bytes). Tolerance widens both: at the greatest
tolerance it accepts any ring and any dot of its species. A child's ring is a
single-point crossover of its parents' rings. A child's dot comes from its
starting mind (see above); its two tolerance flags come together from either
parent. The home temperature travels with the ring. With the world's body
mutation chance (50% by default), one bit of the ring flips, which may change
the home temperature; the tolerance flags never mutate.
Minds blend, but experience and mind mutations keep them varied.

This is genome version 2. Version 1 held eight four-bit traits (0 to 15) per
word, before empathy, tolerance, litter and carrion existed. Older exports,
which carry no version, are converted when imported: each trait is scaled to 0
to 7, and empathy starts at 0, tolerance at 3, litter at 0 and carrion at 0,
with a medium home temperature and neither tolerance. Their colours change,
because the bits move. Version 2 genomes made before the climate genes existed
are read as they are: whatever their top bits were becomes their climate.

The inspector shows both words in hex and every trait. Mind traits appear
inborn and as they are now, coloured green where experience has raised them
and amber where it has lowered them. It also shows the body's energy burn,
its climate (home temperature, the temperature where it stands, coloured amber
one step from home and red two, and what it tolerates), hunger, health, age
against expected lifespan, generation, children, last action, the odds its brain
gave each option, and for a corpse, the meat left on it.

## Seasons and Climate

The map is banded into climate zones across its height. The top and bottom 20%
are arctic, cold; a strip of 20% across the middle is the equator, hot; the two
bands between are temperate, medium. The top and bottom edges of the map are
barriers: nothing walks, flees or is born across them, and seeds that would fall
beyond them are lost. East and west still wrap round. The world is light grey, each row tinted
by its temperature now: reddish brown where hot, green where medium, white where
cold and light blue where very cold, and part way between two while the weather
turns.

A year is 360 turns by default (the year length setting), in four quarters,
starting mild. Through the second quarter (the northern winter) everything from
the top of the map down to and through the equator is a step colder: the arctic
very cold, the temperate zone cold, the equator medium. The third quarter is
mild again. Through the fourth quarter (the southern winter) the same happens
from the bottom up to and through the equator. The sidebar shows the year and season.

The weather doesn't change all at once: it comes in waves, one in each climate
zone, all setting off together as the season turns, so no zone waits on another
and an animal can always see where the weather is headed. In each zone a winter
washes in from the side toward its hemisphere's edge (the top of the map for the
north, the bottom for the south) across to the side toward the equator, and as it
ends ebbs back the way it came. Each wave crosses its zone in bands of two rows
over the first half of each season (45 turns of a 360-turn year). At its front four bands stand part way
between the old weather and the new, each a quarter step further on than the
band ahead of it, so the change comes as a staircase an animal can see within
its sight: winter behind the front, the old season ahead of it. Each band turns
by a quarter step as the front moves on by one. The zones' colours turn with the
wave, and so do its effects: an animal half a step from home needs half the
extra food and loses half a step's health, and grass grows in a row, and dies
back in it, by how medium the row is. The world's first season starts settled.

Every Bitzkey has a home temperature. An animal a step or more from home needs
twice the food, and loses 10% of its full health a turn for each step it is from home:
10% one step away, 20% two and 30% three. It may freeze or overheat to death. An animal that starves while out of its
temperature has died of exposure too: it froze or overheated. Heat tolerance
halves the penalties of heat, and cold tolerance those of cold: a cold-tolerant
animal two steps too cold needs half again as much food rather than double, and
loses 10% of its health a turn rather than 20%. Tolerance
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
and an animal can feel a winter setting in before it bites. Being out of their temperature makes them uneasy, the more so the further from
home (half as much for the hardy), so a step back toward home is felt at once.
They can head north or south, but there is no action that seeks home out for
them; any migration emerges from what their brains learn to do with those senses
and the time of year. Foraging never leads further from home: an animal ignores
grass in rows whose temperature is further from home than where it stands, and
doesn't herd toward a herd whose middle is.

The zones' sizes, how much of each season a wave takes to pass and how wide its bands and front are, the food and health penalties, how much food a plant gives out
of its temperature, how fast grass dies back, how long seeds wait, how common
tolerance is at the start and how much it spares are
advanced rules (the Climate group).

## Behaviour

Grass is sown at random, and regrows each turn (see grass density above). On and around a corpse, the
cell it lies on and the eight cells next to it, grass grows twice as fast
while the body remains.
Grazers and omnivores eat grass, and browse plants; predators can do neither.
The two have separate niches: grazers are the specialists, getting a quarter
more than the food in grass and plants (1.25 times), while omnivores get half the
food in grass and three quarters of the food in plants, and live on meat.
Omnivores and predators may eat carrion; grazers never do.

A dead animal isn't carrion straight away. Through the turn it died and the
next it is a fresh kill: it doesn't rot yet, and an omnivore or predator may eat
it whole if it is something it lives on (grazers, for an omnivore; grazers and
omnivores, for a predator). A fresh kill gives three times the meal of making
the kill, doesn't count as scavenging, so the carrion trait plays no part, and
leaves nothing: the body is gone at the end of the turn. After that it rots as
carrion until it fades.

Plants stay in place and photosynthesize. Every animal carries its own small
neural network, and that brain chooses everything it does. Each turn it senses:

- its hunger and health
- grass underfoot or in sight, and plants in sight (grazers and omnivores only)
- how close the nearest carrion in sight is (omnivores and predators), and the
  nearest fresh kill it would eat (likewise)
- a willing mate beside it, and how close the nearest one in sight is
- prey in sight, how close the nearest is, and how hard it could hit prey beside it
- hunters in sight, how close the nearest is, and how dangerous one beside it is
- whether the animal that just struck it is still beside it, and how hard it could strike back
- its herd: how many herd-mates are close by, how near it is to the herd's middle,
  and whether a herd-mate close by has just been struck
- whether it has lately seen a herd-mate succeed or fail (see below), fading
  turn by turn
- its own kind and other species close by, and its own age
- the time of year, as two hands of a clock a quarter of a year apart, so it can
  learn the seasons' rhythm and what each brings
- how far the temperature is from home, and how close home weather is (see
  Seasons and Climate)
- how far north or south it is, and how much warmer or colder than home it is
  here and at the edge of its sight to the north and to the south
- its current temperament
- its own genome: each of its dot's traits as born, each of its ring's traits
  (species among them) and its climate genes

That is 69 senses in all. Its brain has hidden layers of 34 neurons, twice as
many layers as its
inborn intellect, and never fewer than two: from two layers for the dullest to
fourteen for the brightest. Plants have no brain.

The cleverer an animal, the faster it learns, three ways, while the least
intelligent learn just as they always have. Its learning rate rises with its
intellect, and then by up to as much again for the brightest (the Learning boost
rule). Each turn it re-learns from its most recent experiences, two for each point
of inborn intellect, so the brightest go over their last fourteen while the
dullest replay none (Replays per intellect). And it traces an outcome further back
through the choices that led to it: its trace decay runs from 0.9 for the dullest
to 0.97 for the brightest (Trace decay, smartest). Replaying costs time: in a busy
world it can double the time each turn spends learning.

It then picks one of the actions open to it: eat, forage, browse, uproot,
scavenge, feast, rest, court, breed, hunt, attack, defend, flee, herd, go north,
go south, go east, go west, walk north, walk south, walk east or walk west. It
isn't told which actions are open; it picks among them by the values its brain
gives every action. Each go heads straight that way as far as its speed allows,
and each walk a single cell: north and south stop at the edge of the map, while
east and west wrap round it, so they are always open. Moving costs food: a walk
saves half the turn's burn, as resting does, while a go costs a quarter of a
turn's burn more for each cell beyond the first, so a fast grazer going six cells
pays two and a quarter turns' burn. Setting off in a direction, going or
walking, is how an animal explores: it pleases the curious, and finding grass
where none was seen, or walking into a hunter, shapes its curiosity. A new brain
starts out valuing the eight headings a little below its other actions, so an
untaught animal doesn't march away from its range, its herd and its food before
it has learned where to go. What a walk saves and what going far costs are
advanced rules (the Bodies group). Foraging walks toward the
nearest grass in sight; a tuft sates 35 hunger. Grazers get 1.25 times the food in grass and plants; omnivores half of grass and three quarters of plants. Browsing walks to the nearest
plant in sight and, once alongside, takes a bite: 45 hunger, up to 70 from the
largest plants, far more than grass. Each bite costs the plant 15 health, and
plants regrow 2% of their health each turn. A bitten plant can't be bitten again
for 8 turns, so browsers move on, but enough bites in time kill one, and a dead
plant leaves no meat. A desperate grazer may uproot a plant beside it instead:
only once it is starving (80 hunger or more) and food is scarce about it (less
than the food it would need to breed), but whether the plant has recovered from
its last bite or not. Uprooting feeds it twice a bite's meal and costs the plant
45 health, three bites at once, which often kills it; an armed plant still
strikes back. The hunger, the meal and the damage are advanced rules (the Food
group). Plants spread by seed: each turn a well-fed plant has a 3%
to 10% chance, by its fertility, of sowing a seed of its own kind, with
the usual chance of mutation, on a free cell up to five cells away. Seeding
costs it 20 hunger. A seed doesn't take where more than three plants already
grow within two cells. There is no limit on the number of plants, but they
compete for light: each plant growing within three cells cuts a plant's chance
of seeding by a third, and with three or more around it, it does not seed. Scavenging walks to the
nearest carrion in sight and, once alongside, eats up to 25 meat; the carrion
trait decides how much of that feeds it. Feasting walks to the nearest fresh
kill in sight and, once alongside, eats it whole. Courting walks toward the nearest
willing mate in sight, but breeding is still a separate choice once they are
side by side. Hunting chases the nearest prey in sight. Omnivores prey on grazers; predators prey on grazers, omnivores and other predators. Predators whose ring colours differ by more than 5% (the mean difference of the ring's four CMYK bytes) are different kinds, and hunt one another freely. Hunting their own kind is cannibalism: a predator only sees its own kind as prey once its hunger, as a share, exceeds its reluctance, the average of its current empathy, tolerance and patience. The ruthless will eat kin at any hunger; the gentlest never will. Killing its own kind pains an empathetic predator as much as its own species' pain, rather than the reduced pain it feels for others. Predators and omnivores are territorial. An animal's territory is the cells within six steps of it. A predator counts other predators and omnivores in its territory as rivals for the feeding ground, and how many it will share it with depends on its current empathy: one for the least empathetic, rising to five for the most. Each predator beyond that raises its rivalry by a third, up to full rivalry three past its limit. A predator senses its rivalry, finds an overcrowded territory uncomfortable (less so if tolerant), and feels relief, as well as any joy in fighting, at killing a rival there. Crowding by other predators, though not by omnivores, also wears down its reluctance to turn on its own kind: in a territory fully overcrowded with predators, even the gentlest will fight kin for the feeding ground. Omnivores are territorial at half the strength. They count other omnivores as rivals, and predators too, each counting less the more empathetic the omnivore is, and not at all at the greatest empathy. They tolerate two rivals for the least empathetic, rising to ten for the most, and their rivalry tops out at a half. An omnivore only turns on another omnivore when its territory is overcrowded with omnivores, and then only once its hunger exceeds its reluctance as rivalry wears it down, never by more than half. A hunter that reaches
its prey pounces, striking in the same turn, and attacking strikes prey
alongside. A kill feeds the hunter 25, plus up to 45 more for the largest prey; a blow
that doesn't kill feeds it 5. Striking is hard work: each blow burns 0.2 food per point of damage it deals, so a heavy hitter pays more for each blow. Hunger figures for meals are for a middle-sized
stomach: a small animal gains twice as much, a large one half. Defending is open only to an animal struck since its last turn whose assailant is still beside it: it strikes back with the usual force of its size, strength and weapons against the assailant's size and armour. It costs effort like any blow and feeds the defender nothing, but it can kill. Fight or flight: whenever an animal is struck, by an attack or a counterblow, it checks its morale against its current fear. A fully fearful animal breaks 30% of the time when barely hurt, rising to certainty near death; the fearless never break. One that breaks loses its chance to defend: on its next turn it flees its assailant, whatever its brain would have chosen, and grows a little more fearful. Plants can't flee, so their only answer is to fight: a plant with any weapons strikes back at each animal that browses it. Its thorns come mostly from its weapons (80%) and a little from its size (20%), never its strength: at most 15 damage, less against a large or armoured browser. Fleeing runs from the nearest hunter in sight, and speed decides who wins a
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
turn by its discount (more for the patient) times nine tenths, so a death or a
meal is credited to the run of choices before it, not just the last one. How far
back it reaches is an advanced rule (the Mind group), as is how reluctant a new
brain is to set off in a direction. A child's brain is as deep as its
own inborn intellect makes it. Where both parents have a layer, the child takes
each of its neurons whole from one parent or the other; where only one parent has
it, from that one; deeper still, its layers start out passing their input on.
Each action's output comes from one parent or the other, all with small
mutations. A child also inherits its parents' learned shifts in temperament, so
learned habits and moods pass down the generations.

Animals also learn from their herd, or pack. An animal sees what happens to its
herd-mates within its sight: a herd-mate's kill is a success, and its death,
from exposure, hunger, a blow or anything but old age, a failure. The witness
learns from it as if it had made that herd-mate's last choice itself, in the
same situation, toward the death penalty for a death or the fullness the killer
gained for a kill. It feels it too: a kill lifts its mood and a death lowers it,
by how much the herd-mate gained or lost. How much it takes in is its empathy
times its intellect, so the dull or callous barely notice, while the bright and
caring learn from every loss. It also senses what it saw, fading by a fifth each
turn. How strongly this teaches, how much it moves the mood and how long it is
remembered are advanced rules (the Society group).

Each parent pays 10 to 25% of its stomach to breed, less the more fertile it is. Animals
only breed, or court, when both partners can afford it and still have at least
10 hunger to spare afterwards, so nothing breeds itself to death. The litter
shares exactly the energy its parents gave, so a large parent gives more food
than a small one, and the same food fills a small newborn more. Breeding starts a
gestation: the parent that chose to breed conceives the litter at once, and
carries it for 10 turns if it is the smallest body, up to 50 for the largest,
evenly between; it can't breed again until it has given birth, and if it dies
first, its young die with it. The young's genes, minds and brains are fixed at
mating. At birth, each newborn needs a free cell within two cells of its mother,
nearest first; those with no room are lost, and if a population cap is set, the world holds at most that
many living animals (plants don't count toward it). Crowding also
limits breeding: for each other animal within four cells of a would-be parent,
besides its partner, its litter shrinks by a tenth, and with ten or more around,
it does not breed. Animals also breed responsibly: before breeding, an animal
weighs the food it lives on within six cells against the mouths that share it,
itself included. Grazers count grass and plants (two meals each), as well as
they digest them; omnivores count grass and plants as they digest them, and
grazers as prey (two meals each);
predators count grazers and omnivores as prey. Grass is judged
by what will grow back nearby over the next 150 turns, plus a quarter of what is
standing now, so a crowd doesn't breed on a stock it is about to eat. Every
animal of its own species is a mouth, every predator to a predator whatever its
kind, and another species eating some of the same food is half a mouth (an
omnivore to a grazer, or a predator to an omnivore). Below three meals a mouth it
doesn't breed; between three and six, its litter shrinks in proportion. Other
kinds of predator may still be hunted, but they are rivals for prey, not a food
supply to breed on. Hunters also switch prey: a hunter only goes after a kind of prey while it
makes up at least a quarter of the food it weighs within six cells (grazers
against plants, for an omnivore; grazers against omnivores, for a predator), or
while none of that food is near. A prey that has become scarce is left alone,
and doesn't flee a hunter that isn't interested in it, so it can recover.
Territorial fights, between omnivores or between predators, go on regardless. Reaching 100% hunger or zero health is fatal, and so, in
time, is age. Plants and animals alike live about a year at the smallest body
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
to hold an icon (3px or more) is black, carrying its species' icon in the dot's
colour with a thin light grey outline: a tree for plants, a leaf for grazers, an apple for omnivores and a
drumstick for predators. A smaller dot is simply its colour. The inspector, the designer and the colour graph show them too. A corpse holds the energy its body had left: up to 30 meat, plus
up to 50 more for the largest. A starved body leaves nothing, and a killed one
half, as the killer has already eaten. The meat rots away over the 30 turns
before the corpse fades. Brains, temperament, learning and statistics are all
part of turn history, so rewind and replay are exact.

### Herds and Packs

Grazers, omnivores and predators can all herd. An animal's herd instinct is the
lower of its current empathy and patience, so only animals high in both herd
strongly, and every herd effect scales smoothly with it. Its herd-mates are
animals of its own species that it would accept as mates, so tolerant animals
herd with other kinds of their species. Predators only form packs with their
own kind. Herd-mates are sought within four cells.

- Herding: the herd action moves an animal toward the middle of its herd-mates
  in range. The brain learns when to use it. Fear sets how close it keeps: a
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
still hunts other predator kinds and omnivores as before.

## Colour Graph

Press `G` for the colour graph. Time runs across the bottom. Each bar is a moment
in the run, and every Bitzkey alive then (plants included) is a slice of it in its
own exact colour. Slices are stacked by species, plants at the bottom, then
grazers, omnivores and predators, and each species' band is labelled, with its
icon, at the right end of the chart. Within a species, slices are sorted by hue,
so similar colours flow together from bar to bar. The SORT menu at the top sorts
each species by a trait instead: any of the ten mind traits, the body traits, or
the climate genes (home temperature, heat tolerance, cold tolerance), taken from
the genome, lowest at the bottom (for temperature, hottest at the bottom). Slices with the same value are then
sorted by colour. A trait sorts only the Bitzkeys living now; the dead are
remembered by their colour alone, and sit beneath the living in each species,
sorted by colour. Colour value is the default. A bar's height is the
population. `Tab` switches between ring
(body) and dot (mind) colours. The timeline shades each winter quarter and marks
where each year begins, with a line up through the chart. The graph covers the whole run. It samples who was
alive at evenly spaced turns, spacing them further apart as the run grows.

The graph is also a time scrubber. A playhead marks the present. Drag along the
timeline beneath the bars, or right-click anywhere on the graph, to move the
world to that turn; it pauses there, or at the earliest turn the history
keeps (the latest 100), if that is later. Space plays on through the recorded future, which stays
visible but faded. The mouse wheel zooms the time axis around
the cursor, down to 10 turns. `Shift`+wheel, or a sideways wheel, pans the
zoomed view, and zooming all the way out shows the whole run again.

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
Bitzkey shows only its ring and dot colours there.

The graph and the map share one selection. Close the graph and the Bitzkeys picked
on it stay ringed on the map, while the sidebar shows no single Bitzkey and counts
how many are picked. Clicking a dot on the map selects just that one, on the
graph as well.

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
with when it was added, and the one picked shown large beside its inspector
panel. ADD TO IMPORT (`Enter`) adds it to the import list, brain and all, REMOVE
(`Delete`) takes it out of the catalog for good after asking, and CLOSE (`Esc`)
returns. Each entry is written as an export record, so `catalog.json` can also be
loaded like any export file.

## Export Format

A Bitzkey is written as its genome version and its two colour words in hex,
ring then dot. Animals also carry their current temperament and their brain, so
they pick up in the next simulation where they left off:

```json
{"version": 2, "ring": "7421FF41", "dot": "637E4B01",
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
`"version": 2`. A record without a brain gets a fresh one.
Brain weights are matched to senses and actions by name, so exports from an
earlier version still load: senses or actions they lack start neutral, and ones
this version no longer has are dropped. A brain saved with one hidden layer, from
before brains grew deeper, loads too. A saved brain is fitted to the depth its
Bitzkey's intellect calls for: deepened with layers that pass their input on, or
cut short. Every member of an imported group starts
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

Press `B` to save a brain report to `reports/`. The report puts every living
brain in the same standard situations, such as a fed animal with a willing mate
beside it or a wounded animal beside a hunter, each with its own current
temperament. It averages the chance of each action by species and trait group
and ranks who is least and most keen. It also shows how far experience has
moved each species' temperament from its genes.
