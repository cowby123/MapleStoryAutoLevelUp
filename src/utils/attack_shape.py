'''
Attack reach of directional (ranged) attacks: rectangle (upstream) or wedge.

MapleStory Classic targeting (meowdb.com/msclassic/formulas/targeting): a
ranged basic attack / spell hits the nearest monster in a set of bands in
front of the character that widen with distance -- the bands start `start` px
ahead (65 for bows/claws, 50 for spells), the half-height at distance d is
floor(d / 4), they are centred 28 px above the feet, and they end at the
attack's range (bow 300 + Eye of Amazon, claw 200 + Keen Eyes, spell 300,
Energy Bolt 350). Here that is a continuous wedge: half-height = d / slope.
Walls / platforms blocking the shot are not modelled.

All lengths are game pixels; the bot multiplies them by the frame's UI/world
scale (src/utils/resolution.py; pig sprites measured at 1.054 on maple
1366x768, so game px -> screen px is assumed to be that same scale).
'''


def eye_of_amazon(level):
    '''bow / crossbow range bonus: 45 + 5 x Lv (Lv 1-15), 0 without the skill'''
    level = int(level or 0)
    return 0 if level <= 0 else 45 + 5 * min(level, 15)


def keen_eyes(level):
    '''claw range bonus: 40 + 10 x Lv (Lv 1-14), 200 at Lv 15'''
    level = int(level or 0)
    if level <= 0:
        return 0
    return 200 if level >= 15 else 40 + 10 * level


PRESETS = {
    # name: (start, base range, bonus function or None)
    "archer": (65, 300, eye_of_amazon),        # bow / crossbow basic attack
    "arrow_blow": (65, 350, eye_of_amazon),
    "mage": (50, 300, None),                   # spells without their own area
    "energy_bolt": (50, 350, None),
    "thief_claw": (65, 200, keen_eyes),        # throwing stars
}


def preset_values(name, skill_level=0):
    '''(start, range) of a preset'''
    start, base, bonus = PRESETS[name]
    return start, base + (bonus(skill_level) if bonus else 0)


def wedge_test(px, py, box, facing_left, start, rng, slope, center_dy):
    '''
    Where a monster box (x, y, w, h) is relative to the wedge of a character
    at (px, py) (the bot's player position; the wedge's centre line is at
    py + center_dy). Like the game, the monster's position (here the box's
    horizontal centre) must be in a band; vertically any part of the box counts.
    Returns (state, distance ahead of the monster centre):
      "in"     inside the wedge
      "close"  in front but nearer than start (the dead zone)
      "far"    in front, beyond the range
      "out"    in front, within [start, range] but above / below the reach
      "behind" behind the character
    '''
    x, y, w, h = box
    cx = x + w / 2.0
    d = (px - cx) if facing_left else (cx - px)
    if d < 0:
        return "behind", d
    if d < start:
        return "close", d
    if d > rng:
        return "far", d
    reach = d / float(slope)
    cy = py + center_dy
    if y <= cy + reach and y + h >= cy - reach:
        return "in", d
    return "out", d


def wedge_polygon(px, py, facing_left, start, rng, slope, center_dy):
    '''corner points of the wedge (for drawing)'''
    s = -1 if facing_left else 1
    cy = py + center_dy
    return [(int(px + s * start), int(cy - start / slope)), (int(px + s * rng), int(cy - rng / slope)),
            (int(px + s * rng), int(cy + rng / slope)), (int(px + s * start), int(cy + start / slope))]
