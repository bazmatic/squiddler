"""Carry a trained policy over to a longer senses layout: python -m tools.migrate_policy OLD.npz NEW.npz

The old layout must be a prefix of the target one (--layout hunter, the default, or campaign) (new senses are only ever added at the end). The new inputs
get zero weights, so the migrated pilot flies exactly as the old one did until training teaches it to use them.
"""
import sys
import numpy as np
from hunter.senses import FEATURE_NAMES
from hunter.neural.policy import NeuralPolicy


def migrate(old_path, new_path, feature_names=FEATURE_NAMES):
    data = dict(np.load(old_path, allow_pickle=False))
    old_names = tuple(str(n) for n in data['feature_names'])
    if old_names != feature_names[:len(old_names)]:
        raise ValueError('the old senses are not a prefix of the current layout; it cannot be migrated')
    added = len(feature_names) - len(old_names)
    if added:
        embed = data['embed_w']
        data['embed_w'] = np.concatenate([embed, np.zeros((embed.shape[0], added), embed.dtype)], axis=1)
        data['mean'] = np.concatenate([data['mean'], np.zeros(added, data['mean'].dtype)])
        data['scale'] = np.concatenate([data['scale'], np.ones(added, data['scale'].dtype)])
    data['feature_names'] = np.array(feature_names)
    np.savez(new_path, **data)
    NeuralPolicy.load(new_path, feature_names)              # raises InvalidPolicy if the result does not fit
    return added


def _layout(name):
    if name == 'campaign':
        from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES     # imports pygame-side modules: only on demand
        return CAMPAIGN_FEATURE_NAMES
    return FEATURE_NAMES


if __name__ == '__main__':
    args = sys.argv[1:]
    layout = 'hunter'
    if len(args) == 4 and args[2] == '--layout' and args[3] in ('hunter', 'campaign'):
        layout, args = args[3], args[:2]
    if len(args) != 2:
        sys.exit(__doc__)
    print(f'migrated: {migrate(args[0], args[1], _layout(layout))} new inputs, zero-weighted')
