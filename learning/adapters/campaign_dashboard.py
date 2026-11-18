"""The Campaign Hunter's dashboard: the keys learning.adapters.campaign_hunter.summary() adds to each train.log row
(tools/ppo_neural.py writes the PPO stats and mean_return beside them), and its reward points from
the reward table in learning.campaign_rewards."""
from learning.campaign_rewards import TABLE
from learning.reward_table import RewardKind
from learning.dashboard import Layout, Metric, number

METRICS = (
    Metric('exited', 'Episodes exited',
           'Higher is better. Share of episodes in which the ship leaves the level through the exit.',
           section='Outcome', higher_is_better=True, short='Exited'),
    Metric('cleared_exit', 'Episodes exited with every enemy dead',
           'Higher is better. Share of episodes in which the ship leaves through the exit with no enemy left.',
           section='Outcome', higher_is_better=True, short='Cleared exit'),
    Metric('scored_out', 'Episodes scored out',
           'Lower is better. Share of episodes that end because the score runs out.',
           section='Outcome', higher_is_better=False, short='Scored out'),
    Metric('own_kills', 'Own kills per episode',
           'Higher is better. Enemies the ship destroys itself, on average per episode.',
           section='Outcome', higher_is_better=True, short='Own kills', digits=1),
    Metric('wall_penalty', 'Wall collision penalty per episode',
           'Lower is better. Total training points deducted for wall collisions, averaged across episodes. '
           'For impact-scaled runs, each contact contributes its collision-induced velocity change divided '
           'by ship top speed, multiplied by the penalty coefficient used at that iteration. Slow and glancing '
           'contacts cost less than hard impacts; repeated impacts add up and shielded contacts are free. '
           'Older runs that charged a flat cost per strike show that original penalty. Values are points lost, '
           'so 0 is best and 80 means an average deduction of 80 points per episode.',
           section='Flying', higher_is_better=False, short='Wall penalty', digits=1, optional=True),
    Metric('shots', 'Shots fired per episode',
           'Triggered volleys the ship fires, on average per episode; a spread volley counts once.',
           section='Flying', short='Shots', digits=1, optional=True),
    Metric('damaging_hits', 'Damaging projectile hits per episode',
           'Higher is better alongside kills. Average player projectile hits that reduce enemy health. '
           'Absorbed shots and allied hunter hits do not count. One hit is one damage-dealing contact, '
           'so piercing weapons can register more than one hit per bullet.',
           section='Shooting', higher_is_better=True, short='Damaging hits', optional=True),
    Metric('unaimed_shot_share', 'Shots outside the aim window',
           'Lower is better alongside damaging hits and kills. Share of fired volleys outside the aim window. '
           'Graded-shooting runs use a 30-degree rough window around a seen enemy or its intercept at firing '
           'time. Historical runs use the 20-degree decision-snapshot check. Zero can also mean no shots '
           'were fired, so read it with shots and hits.',
           unit='%', section='Shooting', higher_is_better=False, short='Unaimed', optional=True),
    Metric('missed_opportunities', 'Firing opportunities declined per episode',
           'Lower is better alongside damaging hits and kills. Time spent declining a ready opportunity toward '
           'a seen enemy, measured in weapon firing intervals and averaged across episodes. New graded-shooting '
           'runs use a 30-degree rough window with a clear reachable intercept; older runs require a direct hit. '
           'Cooldown, empty ammunition and explosive or bouncing weapon predictions are excluded. '
           'Measured only when the missed-opportunity penalty is enabled.',
           section='Shooting', higher_is_better=False, short='Declined shots', optional=True),
    Metric('missed_opportunity_reward', 'Declined firing opportunity penalty per episode',
           'Actual signed training points for declining firing opportunities, averaged across episodes. '
           'Closer to zero is better. A coefficient of −0.1 costs 0.1 points per ready weapon firing interval '
           'spent declining a valid predicted shot; holding the trigger incurs no such penalty.',
           section='Shooting', higher_is_better=True, short='Declined-shot penalty', optional=True),
    Metric('shot_accuracy_reward', 'Shot accuracy reward per episode',
           'Signed accuracy points earned by emitted volleys, averaged across episodes. In graded-shooting runs '
           'the best lane grades from zero at 30 degrees intercept error to full credit at zero error, with a '
           'clear reachable path. New runs square this score, making near-miss credit tiny; historical runs '
           'used a linear curve. Actual damaging hits earn their separate, much larger reward. Rough shots '
           'pay no shooting costs. Historical runs without graded shooting earn none of this reward.',
           section='Shooting', higher_is_better=True, short='Aim reward', optional=True),
    Metric('thrust', 'Thruster use',
           'Share of the pilot\'s decisions with the thruster on. 1 means it never coasts.',
           section='Flying', short='Thrust', optional=True),
    Metric('spinning', 'Time spent spinning',
           'Lower is better. Share of the pilot\'s decisions spent in a spin: turning the same way for about two '
           'seconds or more without a break. Steering toward a target is far shorter than this, so a high value '
           'means the ship is circling in place. Measured from runs that record it.',
           unit='%', section='Flying', higher_is_better=False, short='Spinning', optional=True),
    Metric('search_found', 'Hidden-start episodes finding an enemy',
           'Higher is better. Of the episodes that began with no enemy in sight, the share in which the hunter '
           'eventually saw at least one enemy. A value of 1 means 100% found an enemy; 0.5 means half did. '
           'Seeing an enemy counts even if it was not killed. Episodes that began with an enemy already visible '
           'are excluded. Read this alongside search time below.',
           section='Searching', higher_is_better=True, short='Contact', optional=True),
    Metric('search_seconds', 'Time to first enemy sighting',
           'Lower is better when the contact rate above stays high. Average seconds to the first enemy sighting '
           'in episodes that began with no enemy in sight. An episode that never finds an enemy counts as the '
           'full configured time limit (for example, 180 seconds), even if it scores out earlier. This prevents '
           'an early failure from looking like a quick search.',
           section='Searching', higher_is_better=False, short='Search time', optional=True),
    Metric('unseen_seconds', 'Time searching without enemy sight',
           'Average total seconds per episode with enemies still alive but none directly visible. Includes '
           'the initial search and later periods after losing sight of enemies; counting stops once all enemies '
           'are dead. Lower usually means less time searching, but a short failed episode can also give a low '
           'value. Check cleared exits and score-outs alongside it.',
           section='Searching', higher_is_better=False, short='Unseen time', optional=True),
    Metric('hunt_progress_reward', 'Hunt progress reward per episode',
           'Actual signed training points for ship movement toward its preceding enemy goal by maze route, '
           'averaged across episodes. Retreat subtracts points. Enemy movement and wall destruction alone '
           'earn nothing. Circling can cancel out, and longer episodes offer more chances to earn; read with '
           'contact time and cleared exits. This is the base reward, before the unseen-pursuit bonus. '
           'Available from runs that record reward totals.',
           section='Searching', higher_is_better=True, short='Hunt reward', optional=True),
    Metric('unseen_progress_reward', 'Unseen pursuit reward per episode',
           'Actual signed extra training points for route progress when no enemy was visible at the preceding '
           'decision, averaged across episodes. Adds to Hunt progress; retreat subtracts points. Zero may mean '
           'no progress or no unseen phase, so read with contact rate and search time. Available from runs that '
           'record reward totals.',
           section='Searching', higher_is_better=True, short='Unseen reward', optional=True),
    Metric('exit_progress_reward', 'Exit progress reward per episode',
           'Actual signed training points for moving toward the open exit by route, averaged across episodes. '
           'Moving away subtracts points. Pursuit drills finish at contact, so this is normally zero there. '
           'Available from runs that record reward totals.',
           section='Outcome', higher_is_better=True, short='Exit reward', optional=True),
    Metric('lesson_contact', 'Pursuit lessons reaching enemy sight',
           'Share of pursuit drills that began with no enemy visible and reached first enemy sight before '
           'the time limit or score-out. A visible start earns no completion bonus. Read with first-contact '
           'time; a missed search counts the full drill time limit.',
           unit='%', section='Searching', higher_is_better=True, short='Lesson contact', optional=True),
    Metric('breach_blocks', 'Necessary wall blocks destroyed',
           'Average blocks the player completely destroys on a planned breach route while no living enemy has '
           'an open route. Partial damage, unrelated blocks and ally or enemy destruction earn nothing. '
           'Several blocks can count when a route needs a multi-block breach. Read with contacts and cleared '
           'exits: a low value is normal when enemies already have open routes.',
           section='Searching', short='Breach blocks', optional=True),
)

LAYOUT = Layout(sections=('Training', 'Outcome', 'Searching', 'Shooting', 'Flying', 'Policy'),
                strip=('exited', 'cleared_exit', 'own_kills'))

REWARD_RULES = TABLE.reward_rules()
HIDDEN = ('wall_crashes', 'wall_impulse')


def chart_rows(config, rows):
    """Derive actual penalty points from recorded impacts and each iteration's reward coefficient.

    Keep the raw logs intact. The presence of wall_impulse distinguishes severity-scaled rollouts from older
    flat-per-contact rollouts; replaying old rows with today's collision rule would misrepresent their training.
    """
    import math
    from tools import run_record
    rows = [dict(row) for row in rows]
    if not run_record.rules(config).get('graded_shooting', False):
        for row in rows:
            row.pop('shot_accuracy_reward', None)
    weight = ((config.get('args') or {}).get('wall_crash'))
    if weight is None:
        weight = (config.get('rewards') or config.get('shaping') or {}).get('wall_crash')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
        return rows
    result = []
    for row in rows:
        coefficient = weight
        for edit in reversed(config.get('resume_settings_changes', [])):
            after = edit.get('after_iteration')
            if after is not None and row.get('iteration', math.inf) <= after:
                for change in edit.get('changes', []):
                    if change['setting'] == 'wall_crash':
                        coefficient = change['previous']
        measure = row.get('wall_impulse', row.get('wall_crashes'))
        derived = dict(row)
        if measure is not None:
            derived.setdefault('wall_penalty', -coefficient * measure)
        result.append(derived)
    return result


def recorded(config):
    """The points config.json's args record, with a penalty's sign flipped so the page shows it as a penalty."""
    args = (config or {}).get('args') or {}
    values = {t.key: args[t.key] for t in TABLE.terms
              if isinstance(args.get(t.key), (int, float)) and not isinstance(args.get(t.key), bool)}
    from tools import run_record
    if not run_record.rules(config).get('graded_shooting', False):
        values['shot_accuracy'] = 0.0
    return {key: -value if TABLE.by_key[key].kind is RewardKind.PENALTY else value for key, value in values.items()}


def cards(config):
    from tools import run_record
    if not run_record.has_settings(config):
        return []
    e = run_record.episode(config)
    text = (f"Whole Demo levels of up to {number(e.get('horizon_seconds'))} s, ending at the exit or when the score runs "
            f"out. {number(e.get('episodes_per_iteration'))} episodes per iteration.")
    if (config.get('args') or {}).get('pursuit_lesson'):
        text = (f"Pursuit lessons: one stationary hidden target on a verified open route in a real Demo maze, "
                f"ending at first enemy sight or after {number(e.get('horizon_seconds'))} s. "
                f"Waiting cannot bring the target closer. Visible starts earn no contact bonus. "
                f"{number(e.get('episodes_per_iteration'))} episodes per iteration. Exit charts do not apply.")
        combat = config['args'].get('combat_practice_levels')
        if combat:
            text = (f"Mixed practice: full combat episodes on levels {','.join(map(str, combat))}; "
                    f"stationary hidden-target pursuit drills on the other selected levels. "
                    f"Limit {number(e.get('horizon_seconds'))} s; "
                    f"{number(e.get('episodes_per_iteration'))} episodes per iteration. "
                    f"Contact bonuses and lesson contact rate apply only to pursuit drills. "
                    f"Exit, score-out and kill metrics cover combat episodes; other totals average all episodes.")
    if e.get('levels'):
        text += f" Levels: {','.join(str(level) for level in e['levels'])}."
    return [dict(title='Episodes', rows=[], note=text)]
