"""The Wallcrawler team's dashboard charts, in the order and words the trainer page has always used. Keys come from
crawler_scenario.metrics(), learning.run (battle_stage) and learning.ppo.update_team (imitation_loss)."""
import math

from learning.dashboard import Layout, Metric, RewardRule, number

METRICS = (
    Metric('hits_per_minute', 'Team projectile hits per minute',
           'Wallcrawler team: accepted bullet impacts on the frozen opponents.', higher_is_better=True,
           short='Team hits / min', digits=1),
    Metric('mean_aim_quality', 'Predicted shot quality',
           'Mean launch-time interception quality, from 0 to 1. A shaping signal, not measured hits.'),
    Metric('target_clear_rate', 'Targets cleared',
           'Share of practice trials in which every target was hit.', higher_is_better=True),
    Metric('accuracy', 'Team accuracy',
           'Accepted hits divided by actual crawler shots.', higher_is_better=True, short='Accuracy'),
    # The per-drill keys appear only in iterations that flew that kind of trial.
    Metric('precision_accuracy', 'Stationary-target accuracy',
           'Accepted hits / shots in harmless stationary-target drills.', higher_is_better=True, optional=True),
    Metric('tracking_accuracy', 'Moving-target accuracy',
           'Accepted hits / shots in harmless moving-target drills.', higher_is_better=True, optional=True),
    # Logged only by runs with the battle curriculum.
    Metric('battle_stage', 'Battle curriculum stage',
           '0: one Jelly; 1: two; 2: four; 3: four plus the armed hunter.', optional=True),
    Metric('pursuit_survival', 'Battle survival',
           'Fraction of battle crawlers alive at trial end.', higher_is_better=True, optional=True),
    Metric('pursuit_accuracy', 'Combat accuracy',
           'Accepted hits / shots in retained Jellyfish and neural-hunter encounters.', higher_is_better=True,
           optional=True),
    Metric('reversals_per_episode', 'Walking reversals per trial',
           'Actual direction changes while walking; stops and airborne motion excluded.'),
    Metric('jumps_per_episode', 'Voluntary jumps per trial',
           'Actual voluntary launches, excluding damage-induced detachment.'),
    Metric('bullet_hits_taken_per_episode', 'Bullet hits received per trial',
           'Average damaging bullet impacts received by the whole crawler team per trial. Lower is better.',
           higher_is_better=False),
    Metric('contact_hits_taken_per_episode', 'Jellyfish contact hits per trial',
           'Average accepted damaging contacts received by crawlers; repeated overlap is limited by a shared cooldown.',
           higher_is_better=False),
    Metric('imitation_loss', 'Demonstration rehearsal loss',
           'Weighted auxiliary loss that helps retain demonstrated jump and landing choices. This is not an '
           'escape-success score.'),
    Metric('airborne_fraction', 'Time airborne',
           'Share of crawler simulation time spent unattached, including stun and voluntary flight.'),
    Metric('survival', 'Crawler survival',
           'Share of the team still alive at trial end.', higher_is_better=True, short='Survival'),
)

LAYOUT = Layout(strip=('hits_per_minute', 'accuracy', 'survival'))

# Raw counts and per-drill copies crawler_scenario.metrics() logs that the page has never charted.
HIDDEN = ('hits', 'shots', 'episodes', 'hit_probability', 'hits_per_crawler', 'aim_drill_episodes',
          'battle_trials', 'battle_episodes', 'battle_crawlers', 'battle_survivors', 'battle_successes',
          'jump_reward_per_minute', 'landings_per_minute', 'contact_hits_received_per_minute',
          'bullet_hits_received_per_minute', 'reposition_targets_per_episode',
          'precision_survival', 'precision_mean_return', 'precision_hits_per_minute',
          'tracking_survival', 'tracking_mean_return', 'tracking_hits_per_minute',
          'pursuit_mean_return', 'pursuit_hits_per_minute')

# Every reward and penalty the trainer page's old Wallcrawler cards could list, in their order and words. Values come from
# the run's manifest (rewards, else the older shaping record, and battle_rewards), shown as magnitudes; a term the run
# does not record is not part of it, so there is no default.
_RULES = (   # key, title, kind, where the manifest records it, description
    ('team_hit', 'Accepted projectile hit', 'reward', 'rewards.team_hit',
     'For a genuine accepted bullet hit: shared by the team, or only for the crawler whose bullet hit when rewards are '
     'individual (see Reward details).'),
    ('shot', 'Actual shot', 'penalty', 'rewards.shot', 'Cost for each shot actually fired'),
    ('successful_landing', 'Successful landing', 'reward', 'rewards.successful_landing',
     'Combat trials only in current runs; aiming drills earn no landing bonus. Once per completed voluntary jump to the '
     'landing crawler. Battle-curriculum runs pay no landing bonus.'),
    ('alive_per_second', 'Battle time alive', 'reward', 'battle_rewards.alive_per_second',
     'Per living crawler per second'),
    ('survival_bonus', 'Battle survived', 'reward', 'battle_rewards.survival_bonus',
     'Per crawler alive at terminal battle success: time limit or hunter-free Jelly clear'),
    ('death_cost', 'Crawler killed', 'penalty', 'battle_rewards.death_cost', 'Once per death in battle'),
    ('jelly_kill', 'Jelly destroyed', 'reward', 'battle_rewards.jelly_kill', 'To the shooter, in addition to hit rewards'),
    ('trial_completion', 'Aiming trial cleared', 'reward', 'rewards.trial_completion',
     'Once per successful clear, shared equally by living crawlers. No bonus for timeout or team death.'),
    ('jump_proximity', 'Jump near enemy', 'reward', 'rewards.jump_proximity',
     'Pursuit trials: per actual launch, scales from zero at the recorded body clearance (see Reward details) to this '
     'maximum at contact; nearest visible enemy only; reward goes to jumper only.'),
    ('contact_hit', 'Struck by Jellyfish', 'penalty', 'rewards.contact_hit',
     'Cost to the struck crawler only; shared contact cooldown prevents repeated overlap charges'),
    ('bullet_hit', 'Struck by bullet', 'penalty', 'rewards.bullet_hit',
     'Cost to the struck crawler only, once per damaging bullet impact'),
    ('walk_reversal', 'Walking reversal', 'penalty', 'rewards.walk_reversal',
     'Cost per actual attached walking reversal; stopping does not reset direction history'),
    ('voluntary_jump', 'Voluntary jump', 'penalty', 'rewards.voluntary_jump',
     'Cost per actual voluntary launch; damage-induced detachment is excluded. Curriculum runs charge dodge and pursuit '
     'trials their own cost (see Reward details).'),
    ('predicted_shot_accuracy', 'Predicted shot accuracy', 'reward', 'rewards.predicted_shot_accuracy',
     'Up to this amount per shot for predicted interception quality; blocked paths excluded. This is not a measured hit.'),
)
REWARD_RULES = tuple(RewardRule(key, title, description, kind, (), f'manifest.{where}')
                     for key, title, kind, where, description in _RULES)


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def recorded(config):
    manifest = config.get('manifest') or {}
    rewards = manifest.get('rewards') or {}
    battle = manifest.get('battle_rewards') or {}
    found = {}
    for key, _, _, where, _ in _RULES:
        group, name = where.split('.')
        value = (battle if group == 'battle_rewards' else rewards).get(name)
        if _numeric(value):
            found[key] = abs(value)
    if battle and 'successful_landing' in found:
        found['successful_landing'] = 0.   # battle and aiming-drill trials pay no landing bonus (manifest landing_reward)
    return found


def stage(row):
    return int(row['battle_stage']) if _numeric(row.get('battle_stage')) else None


def _percent(share):
    return math.floor(share*100+.5)


def cards(config):
    """The old page's Wallcrawler Episodes card, and the reward facts its rows spelled out from the manifest."""
    m, args = config.get('manifest') or {}, config.get('args') or {}
    rewards = m.get('rewards') or {}
    individual = m.get('reward_scope') == 'individual'
    targets, teacher = m.get('practice_targets') or {}, m.get('jump_teacher')
    curriculum, pursuers, proximity = m.get('curriculum'), m.get('pursuers'), m.get('jump_proximity')
    horizon, episodes = number(m.get('horizon', args.get('horizon'))), number(args.get('episodes'))
    ending = ' or every target is hit' if targets.get('dies_on_hit') else ''
    note = [f'Up to {horizon} seconds or until all crawlers die{ending}. {episodes} episodes per iteration.']
    if teacher:
        weight = teacher.get('rehearsal_weight')
        rehearsal = f' with jump/landing rehearsal (weight {number(weight)})' if weight else ''
        note.append('Bootstrap teacher: pre-contact escape demonstrations train aiming, firing, detachment and legal '
                    f'landing selection; PPO episodes use the learned policy{rehearsal}.')
    if curriculum:
        share = curriculum.get('shooting_share', 0)
        trials, shooting = _percent(1 - share), _percent(share)
    if pursuers and curriculum:
        health = f", {number(pursuers['health'])} HP each" if pursuers.get('health') else ''
        if 'speed' in pursuers:
            speed = f"; speed {number(pursuers['speed'])} px/frame"
        elif 'speed_range' in pursuers:
            low, high = pursuers['speed_range']
            speed = f'; speed {number(low)}–{number(high)} px/frame'
        else:
            speed = ''
        note.append(f"Jellyfish pursuit curriculum: {trials}% pursuit trials ({number(pursuers.get('count'))} allied "
                    f"Jellyfish{health}; {number(pursuers.get('contact_damage'))} HP contact damage{speed}), {shooting}% "
                    'shooting trials. Two crawlers until paired evaluation supports progression.')
    elif curriculum:
        note.append(f"Dodge and reposition curriculum: {trials}% dodge trials ({number(curriculum.get('dodge_targets'))} "
                    f"targets, {number(curriculum.get('burst_size'))}-shot aimed bursts), {shooting}% shooting trials. "
                    'Two crawlers until paired evaluation supports progression.')
    if m.get('ships_fire') is False:
        note.append('Opponent ships do not fire.')
    if m.get('aiming_practice'):
        walking = 'walking enabled' if m.get('practice_walking') else 'walking disabled'
        jumping = 'enabled' if m.get('practice_jumping') else 'disabled'
        frames = m.get('practice_fire_interval_frames')
        reload = f' Reload time: {number(frames / 60)} seconds.' if frames else ''
        note.append(f'Aiming practice: {walking}; voluntary jumping {jumping}.{reload}')
    if targets:
        fire = (targets.get('return_fire') or {}).get('interval_frames')
        slow = f' Slow return fire every {number(fire / 60)} seconds.' if fire else ''
        note.append(f"Up to {number(targets.get('count'))} practice targets.{slow}")
    else:
        note.append('Frozen neural opponents;')
    note.append('No pickups or passive power drain.')
    note.append('Individual rewards: each crawler receives only its own hits and costs.' if individual
                else 'Rewards are shared by the crawler team.')
    details = [dict(label='Accepted projectile hit', value='Individual' if individual else 'Shared',
                    description='Reward only for the crawler whose bullet hit' if individual
                    else 'Shared team reward for a genuine accepted bullet hit')]
    if rewards.get('jump_proximity') and proximity:
        reach = number(proximity.get('range_pixels'))
        details.append(dict(label='Jump near enemy', value=f'{reach}px', description=(
            f'Pursuit trials: per actual launch, scales from zero at {reach}px body clearance to the full reward at '
            'contact; nearest visible enemy only; reward goes to jumper only.')))
    if rewards.get('voluntary_jump') and curriculum:
        dodge = curriculum.get('dodge_jump_cost')
        pursuit = (proximity or {}).get('pursuit_jump_cost', dodge)
        shooting_cost = number(abs(rewards['voluntary_jump']))
        details.append(dict(label='Voluntary jump', value='by trial', description=(
            f'Shooting trials: {shooting_cost}; dodge trials: {number(dodge)}; pursuit trials: {number(pursuit)}. '
            'Damage-induced detachment excluded.')))
    return [dict(title='Episodes', rows=[], note=' '.join(note)), dict(title='Reward details', rows=details)]
