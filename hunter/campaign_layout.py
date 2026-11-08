"""The Campaign Hunter's senses layouts, by name only: nothing here loads pygame, so the trainer server and the policy
loader can know them. hunter.campaign_senses fills them in."""
from hunter.senses import FEATURE_NAMES

# The pickup weapons the ship can hold, in the catalog's order (every drop but the shield, which is not held).
HELD_WEAPON_IDS = ('twin_blaster', 'scattergun', 'railgun', 'seeker', 'inflator', 'grenade_launcher')

# The first trained layout: the hunter's senses, the exit compass and exit_open, and the beacon compass.
CAMPAIGN_FEATURE_NAMES_V1 = FEATURE_NAMES + (
    'exit_compass_present', 'exit_compass_sin', 'exit_compass_cos', 'exit_compass_dist', 'exit_open',
    'beacon_compass_present', 'beacon_compass_sin', 'beacon_compass_cos', 'beacon_compass_dist')

# Then which pickup weapon is held (all 0 for the plain gun) and how much of a pickup shield is left.
CAMPAIGN_FEATURE_NAMES = CAMPAIGN_FEATURE_NAMES_V1 + tuple(f'weapon_{w}' for w in HELD_WEAPON_IDS) + (
    'shield_remaining',)

# Older layouts a policy file may still be in, carried over to the current one with the new inputs at zero weight.
OLDER_CAMPAIGN_LAYOUTS = (CAMPAIGN_FEATURE_NAMES_V1,)
