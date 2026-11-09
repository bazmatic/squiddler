"""The reward rules panel: what a run trained with, from its own config, as the page shows it.

Each test serves runs from a temporary folder through the real trainer server and, where the page matters, renders the
endpoint's answer with the page's own module (tools/trainer/static/rewards.js) under Node, then reads the visible text."""
import json
import re
import shutil
import subprocess
import threading
import urllib.request
from argparse import Namespace
from pathlib import Path

import pytest

from tests.test_trainer_control import make_run
from tools.trainer import server as server_module

NODE = shutil.which('node')
REWARDS_JS = (Path(__file__).resolve().parents[1] / 'tools' / 'trainer' / 'static' / 'rewards.js').as_uri()


@pytest.fixture
def server(tmp_path):
    httpd = server_module.make_server(0, root=tmp_path, processes=lambda: [], launch=None)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Namespace(root=tmp_path, url=f'http://127.0.0.1:{httpd.server_address[1]}')
    httpd.shutdown()
    httpd.server_close()


def add_run(server, name, config, log_rows=()):
    run = server.root / name
    run.mkdir()
    (run / 'config.json').write_text(json.dumps(config))
    lines = [' '.join(f'{k}={v}' for k, v in dict(iteration=i, update_s=1.0, **row).items())
             for i, row in enumerate(log_rows)]
    (run / 'train.log').write_text(''.join(line + '\n' for line in lines))
    return run


def rewards(server, name):
    with urllib.request.urlopen(f'{server.url}/api/runs/{name}/rewards') as response:
        return json.loads(response.read())


def rule(payload, key):
    return next(r for r in payload['rules'] if r['key'] == key)


def shown(payload):
    """The panel and cards as the page renders them: (whole visible text, {rule or card row title: its visible text})."""
    if NODE is None:
        pytest.skip('Node.js is needed to render the panel')
    script = (f"import {{renderRewards,renderCards}} from '{REWARDS_JS}';import fs from 'node:fs';"
              "const p=JSON.parse(fs.readFileSync(0,'utf8'));console.log(renderRewards(p)+renderCards(p.cards))")
    out = subprocess.run([NODE, '--input-type=module', '-e', script], input=json.dumps(payload),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    text = lambda html: ' '.join(re.sub(r'<[^>]*>', ' ', html).replace('&amp;', '&').split())
    items = [text(li) for li in re.findall(r'<li\b.*?</li>', out.stdout, re.S)]
    return text(out.stdout), items


def item(items, title):
    return next(i for i in items if title in i)


JELLY = dict(task='jelly_team', args=dict(episodes=96))


def test_the_panel_shows_each_runs_own_values_and_not_recorded_with_the_default_for_an_old_run(server):
    add_run(server, 'run1', dict(JELLY, args=dict(episodes=96, jelly_catch_reward=12.0, jelly_death_penalty=7.0)))
    add_run(server, 'run2', dict(JELLY, args=dict(episodes=96, jelly_catch_reward=3.0, jelly_death_penalty=7.0)))
    add_run(server, 'run3', JELLY)             # from before the Jelly's rewards were settings
    first, second, old = rewards(server, 'run1'), rewards(server, 'run2'), rewards(server, 'run3')
    assert (rule(first, 'catch')['value'], rule(second, 'catch')['value']) == (12.0, 3.0)
    assert rule(first, 'catch')['recorded'] and rule(first, 'death')['value'] == 7.0
    assert (rule(old, 'catch')['value'], rule(old, 'catch')['recorded']) == (10.0, False)
    text, items = shown(old)
    assert 'not recorded' in item(items, 'Catch') and '10' in item(items, 'Catch')


def test_active_rules_follow_the_runs_current_stage(server):
    add_run(server, 'run1', JELLY, [dict(stage1_episodes=96)])
    add_run(server, 'run2', JELLY, [dict(stage1_episodes=96), dict(stage1_episodes=24, stage2_episodes=72)])
    add_run(server, 'run3', JELLY, [dict(stage1_episodes=12, stage2_episodes=12, stage3_episodes=72)])
    add_run(server, 'run4', JELLY)            # nothing logged yet: no stage, every rule shown active
    active = {name: {r['key'] for r in rewards(server, name)['rules'] if r['active']}
              for name in ('run1', 'run2', 'run3', 'run4')}
    assert [rewards(server, name)['stage'] for name in ('run1', 'run2', 'run3', 'run4')] == [1, 2, 3, None]
    assert {'catch', 'progress', 'stuck'} <= active['run1'] and not active['run1'] & {'flank', 'bunch', 'ship_hit'}
    assert {'flank', 'bunch', 'catch'} <= active['run2'] and not active['run2'] & {'ship_hit', 'ship_kill', 'death'}
    assert {'flank', 'bunch', 'ship_hit', 'ship_kill', 'death'} <= active['run3'] and 'catch' in active['run3']  # retention cases still pay catches
    assert active['run4'] == {r['key'] for r in rewards(server, 'run4')['rules'] if r['value']}   # boredom defaults to 0


# The old page's Wallcrawler cards (loadRules) for a run that recorded these rewards: every reward and penalty it could
# list, each with its recorded value, and its episode description.
CRAWLER = dict(task='wallcrawler_team', args=dict(episodes=8), manifest=dict(horizon=60, rewards=dict(team_hit=2, shot=-.02)))
FULL = dict(task='wallcrawler_team', args=dict(episodes=12), manifest=dict(
    horizon=60, reward_scope='individual', ships_fire=False,
    rewards=dict(team_hit=1, shot=-.05, successful_landing=1, trial_completion=10, jump_proximity=1, contact_hit=-5,
                 bullet_hit=-1, walk_reversal=-.03, voluntary_jump=-.1, predicted_shot_accuracy=.02),
    battle_rewards=dict(alive_per_second=.1, survival_bonus=20, death_cost=40, jelly_kill=5),
    jump_proximity=dict(range_pixels=150), curriculum=dict(shooting_share=.75, dodge_jump_cost=.02),
    pursuers=dict(count=4, health=2, contact_damage=4, speed=2.5), jump_teacher=dict(rehearsal_weight=.1)))


@pytest.mark.parametrize('bonus', [0, .01])
def test_wallcrawler_rules_show_the_rewards_and_penalties_the_old_cards_showed(server, bonus):
    manifest = dict(CRAWLER['manifest'], rewards=dict(team_hit=2, shot=-.02, **({'predicted_shot_accuracy': bonus} if bonus else {})))
    add_run(server, 'run92', dict(CRAWLER, manifest=manifest))
    text, items = shown(rewards(server, 'run92'))
    assert 'Rewards' in text and 'Penalties' in text
    assert '+2' in item(items, 'Accepted projectile hit') and 'inactive' not in item(items, 'Accepted projectile hit')
    assert '−0.02' in item(items, 'Actual shot')
    assert '8 episodes per iteration' in text
    assert ('inactive' not in item(items, 'Predicted shot accuracy')) == (bonus > 0)
    if bonus:
        assert '+0.01' in item(items, 'Predicted shot accuracy')


def test_every_wallcrawler_reward_and_penalty_row_is_still_shown(server):
    add_run(server, 'run145', FULL, [dict(battle_stage=2)])
    text, items = shown(rewards(server, 'run145'))
    for title, value in [('Accepted projectile hit', '+1'), ('Actual shot', '−0.05'), ('Battle time alive', '+0.1'),
                         ('Battle survived', '+20'), ('Crawler killed', '−40'), ('Jelly destroyed', '+5'),
                         ('Aiming trial cleared', '+10'), ('Jump near enemy', '+1'), ('Struck by Jellyfish', '−5'),
                         ('Struck by bullet', '−1'), ('Walking reversal', '−0.03'), ('Voluntary jump', '−0.1'),
                         ('Predicted shot accuracy', '+0.02')]:
        assert value in item(items, title) and 'inactive' not in item(items, title), title
    assert 'inactive' in item(items, 'Successful landing')   # battle runs pay no landing bonus
    for fact in ['150px body clearance', 'Shooting trials: 0.1; dodge trials: 0.02; pursuit trials: 0.02',
                 'Reward only for the crawler whose bullet hit', '12 episodes per iteration', 'Bootstrap teacher',
                 'rehearsal (weight 0.1)', '25% pursuit trials (4 allied Jellyfish, 2 HP each',
                 'Opponent ships do not fire.', 'Individual rewards: each crawler receives only its own hits and costs.']:
        assert fact in text, fact
    landing = dict(FULL, manifest={k: v for k, v in FULL['manifest'].items() if k != 'battle_rewards'})
    add_run(server, 'run130', landing)
    assert '+1' in item(shown(rewards(server, 'run130'))[1], 'Successful landing')


def test_a_hunter_run_shows_the_rewards_its_settings_train_with_and_its_episode_card(server):
    run = make_run(server.root, 'run83', 2)              # an untagged (Hunter) run as the trainer records it
    config = json.loads((run / 'config.json').read_text())
    config['args'].update(powerup_reward=-3.0, death_penalty=-5.0, horizon=60.0, after_clear=30.0, episodes=96,
                          health=10, levels=[3, 5, 6, 7, 25])
    (run / 'config.json').write_text(json.dumps(config))
    payload = rewards(server, 'run83')
    text, items = shown(payload)
    assert '+100' in item(items, 'Following the enemy compass') and '−3' in item(items, 'Flying below top speed')
    assert '+100' in item(items, 'Kills')
    assert 'inactive' in item(items, 'Boredom')         # not set in this run
    assert ('Up to 60 s of fighting, then up to 30 s more to collect powerups once the last enemy is dead. '
            '96 episodes per iteration.') in text
    assert 'The hunter survives 10 energy' in text and 'Levels: 3,5-7,25.' in text
    assert '−3' in item(items, 'Powerup collected') and '+5' in item(items, 'Death')
    assert payload['episodes'] == 96
    # A run that records no settings shows none: rewards rows and episode blocks it kept are not read.
    config.pop('args')
    config['rewards'] = [{'name': 'Powerup collected', 'sign': '+', 'value': 15.0}]
    (run / 'config.json').write_text(json.dumps(config))
    _, legacy_items = shown(rewards(server, 'run83'))
    assert 'not recorded' in item(legacy_items, 'Powerup collected')


def test_new_jelly_runs_panel_displays_the_reward_values_the_environment_uses(server):
    from tools import ppo_neural
    from learning.adapters import jelly_scenario
    args = ppo_neural.build_parser().parse_args(['--task', 'jelly_team', '--jelly-catch-reward', '12',
                                               '--jelly-flank-range', '250', '--jelly-death-penalty', '7'])
    manifest = jelly_scenario.metadata(args)
    add_run(server, 'run150', dict(task=args.task, args=vars(args), manifest=manifest))
    payload = rewards(server, 'run150')
    text, items = shown(payload)
    assert '+12' in item(items, 'Catch') and 'not recorded' not in item(items, 'Catch')
    assert '250' in item(items, 'Flank range') and '−7' in item(items, 'Jelly killed')
    assert manifest['rewards']['catch'] == 12 and manifest['flank']['range'] == 250
    assert manifest['rewards']['stage3_death'] == -7
