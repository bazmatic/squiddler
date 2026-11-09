"""Task-registered training lifecycle, bounded runs, evaluation and reproducible replays."""
import json
import itertools
import math
import os
from pathlib import Path
import time
import numpy as np


def _worker_init():
    # SDL must not turn worker SIGTERM into a queued window event: Pool.terminate()
    # would otherwise wait forever for workers that never poll that event queue.
    os.environ['SDL_NO_SIGNAL_HANDLERS'] = '1'
    os.environ['SDL_VIDEODRIVER'] = os.environ['SDL_AUDIODRIVER'] = 'dummy'
    os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'


def rollout_job(job):
    from learning.tasks import task
    from learning.policy import RecurrentPolicy
    from learning.training import collect
    adapter = task(job['task']).adapter()
    config_path = Path(job['weights']).parent/'config.json'
    recorded = json.loads(config_path.read_text()) if config_path.exists() else {}
    options = getattr(adapter, 'environment_options', lambda manifest: {})(recorded.get('manifest', {}))
    env = adapter.environment(opponent=job['opponent'], horizon=job['horizon'], muted=job['muted'],
                              arenas=recorded.get('manifest', {}).get('arenas'), scripted=job['mode']=='scripted', **options)
    policy = RecurrentPolicy.load(job['weights'], adapter.SPEC)
    if job['mode'] == 'sample' and hasattr(env, 'enable_training_limits'):
        env.enable_training_limits()
    teacher = env.adapter.teacher_action if job['mode'] == 'teacher' else None
    case = dict(job['case']); seed = case.pop('seed')
    labeller = (lambda env, i: getattr(env.adapter, 'label_action', env.adapter.teacher_action)(env, i, None)) if (
        job['mode'] == 'sample' and getattr(env, 'teacher_weight', 0)) else None
    try:
        return collect(env, policy, seed, case, greedy=job['mode'] != 'sample', teacher=teacher, labeller=labeller)
    finally:
        env.close()


class Rollouts:
    def __init__(self, workers):
        _worker_init()  # Also applies to in-process scripted validation and baselines.
        import multiprocessing
        self.deadline = float('inf')
        self.pool = multiprocessing.get_context('spawn').Pool(workers, initializer=_worker_init) if workers > 1 else None
    def __call__(self, jobs):
        if self.pool is None:
            _worker_init()
            results = []
            for job in jobs:
                if time.monotonic() >= self.deadline:
                    raise TimeoutError('training wall-clock budget exhausted')
                results.append(rollout_job(job))
            return results
        timeout = None if math.isinf(self.deadline) else max(.001,self.deadline-time.monotonic())
        return self.pool.map_async(rollout_job, jobs).get(timeout=timeout)
    def close(self):
        if self.pool:
            self.pool.terminate(); self.pool.join()


def make_jobs(args, weights, cases, mode='sample', muted=False):
    return [dict(task=args.task,weights=str(weights),opponent=args.opponent,horizon=args.horizon,
                 case=case,mode=mode,muted=muted) for case in cases]


def evaluate(pool, args, weights, cases, mode=None, muted=False):
    from learning.tasks import task
    adapter = task(args.task).adapter()
    if mode is None:
        mode = getattr(adapter, 'EVALUATION_ACTION_MODE', 'greedy')
    flown = pool(make_jobs(args,weights,cases,mode,muted))
    rows = [dict(result[1],**case) for result,case in zip(flown,cases)]
    return dict(summary=adapter.metrics(rows),episodes=rows)


def expected_manifest(args):
    """The manifest a run with these arguments records: its task's rules (the adapter's metadata) and its training
    settings. A run resumes only while its recorded manifest is still this; tools.trainer recomputes it when a paused
    run's settings are edited."""
    from learning.tasks import task
    adapter = task(args.task).adapter()
    expected = json.loads(json.dumps(adapter.metadata(args)))
    expected['training'] = {k:getattr(args,k) for k in ('seed','lr','gamma','gae_lambda','episodes','bootstrap_episodes')}
    if getattr(args, 'bootstrap_epochs', 0):
        expected['training']['bootstrap_epochs'] = args.bootstrap_epochs
    return expected


def train(args):
    entered = time.monotonic()
    import torch
    import shutil
    from learning.tasks import task
    from learning.policy import random_policy, RecurrentPolicy
    from learning.models import export
    from learning.training import build_network, clone_behaviour
    from learning.ppo import update_team
    from tools.ppo_neural import write_state, PAUSE_FILE
    adapter = task(args.task).adapter()
    torch.set_num_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    uses_opponent = getattr(adapter, 'USES_OPPONENT', True)
    if args.resume and uses_opponent:
        args.opponent = str((out/'opponent.npz').resolve())
    bootstrap_epochs = getattr(args, 'bootstrap_epochs', 0)
    if bootstrap_epochs < 0:
        raise ValueError('bootstrap epochs must be nonnegative (0 uses the task default)')
    expected = expected_manifest(args)
    config_path = out/'config.json'
    resume_path = out/'resume.pt'
    if args.resume:
        recorded = json.loads(config_path.read_text())
        if recorded['manifest'] != expected:
            raise ValueError('resume task, rules, schemas, opponent or arenas changed')
        saved = torch.load(resume_path, map_location='cpu', weights_only=False)
        policy = RecurrentPolicy.load(out/saved['policy'], adapter.SPEC)
    else:
        saved = None
        if resume_path.exists() or (out/'train.log').exists():
            raise ValueError('run already exists; use resume')
        if args.start and Path(args.start).resolve() != Path(args.opponent).resolve():
            policy = (adapter.load_transfer_policy(args.start) if hasattr(adapter,'load_transfer_policy')
                      else RecurrentPolicy.load(args.start, adapter.SPEC))
        else:
            policy = random_policy(adapter.SPEC, args.seed)
        actor_scale=getattr(args,'actor_scale',1.)
        if actor_scale!=1.:
            from learning.initialization import dampen_actor
            policy=dampen_actor(policy,actor_scale)
        if uses_opponent:
            shutil.copyfile(args.opponent, out/'opponent.npz')
        (out/'initial.npz').parent.mkdir(parents=True, exist_ok=True)
        policy.save(out/'initial.npz')
        # Its settings once (args), and its rules snapshot; tools.run_record derives the rest from them.
        record=dict(task=args.task, args=vars(args), manifest=expected)
        if actor_scale!=1.:
            record['actor_initialization']={'action_head_scale':actor_scale,'aim_std':.35,
                'preserved':'recurrent_backbone_and_normalization','applied':'once_before_first_iteration'}
        config_path.write_text(json.dumps(record,indent=2))
    # Use the immutable copy in every worker/replay, never whichever model is later installed in assets.
    args.opponent = str((out/'opponent.npz').resolve()) if uses_opponent else ''
    torch.manual_seed(args.seed)
    net = build_network(policy, adapter.CRITIC_SIZE)
    optimiser = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    first, elapsed, team_size, best_score, current = 0, 0., adapter.INITIAL_STAGE, -float('inf'), 'initial.npz'
    if saved:
        net.load_state_dict(saved['net']); optimiser.load_state_dict(saved['optimiser'])
        rng.bit_generator.state = saved['rng']; torch.set_rng_state(saved['torch_rng'])
        first, elapsed, team_size = saved['iteration'], saved['elapsed'], saved['team_size']
        best_score, current = saved['best_score'], saved['policy']
    else:
        torch.manual_seed(args.seed)
    curriculum = dict(saved['curriculum']) if saved and saved.get('curriculum') is not None else (
        adapter.initial_curriculum(args) if hasattr(adapter, 'initial_curriculum') else
        dict(adapter.INITIAL_CURRICULUM) if hasattr(adapter,'INITIAL_CURRICULUM') else None)
    def case_options():
        return {'curriculum':curriculum} if curriculum is not None else {}
    def validation_cases():
        return (adapter.curriculum_validation_cases(curriculum)
                if curriculum is not None and hasattr(adapter, 'curriculum_validation_cases')
                else adapter.validation_cases())
    def selection_score(summary):
        return (adapter.curriculum_selection_score(summary, curriculum)
                if curriculum is not None and hasattr(adapter, 'curriculum_selection_score')
                else adapter.selection_score(summary))
    def selection_context():
        return adapter.curriculum_key(curriculum) if curriculum is not None and hasattr(adapter, 'curriculum_key') else None
    pool = Rollouts(args.workers)
    started = entered
    unlimited = args.budget_seconds == 0
    pool.deadline = float('inf') if unlimited else started+max(0,args.budget_seconds-elapsed)
    reserve = 0. if unlimited else min(600., args.budget_seconds/6)
    training_limit = float('inf') if unlimited else max(0., args.budget_seconds-reserve)
    (out/'episodes').mkdir(exist_ok=True)
    validation = validation_cases()
    best = out/'best.npz'
    history = []
    demonstrations=[]
    demonstration_path=out/'demonstrations.pt'
    if saved and getattr(adapter,'IMITATION_REHEARSAL_WEIGHT',0):
        if args.bootstrap_episodes and not demonstration_path.exists():
            raise ValueError('resume requires the saved demonstration trajectories')
        if demonstration_path.exists():demonstrations=torch.load(demonstration_path,weights_only=False,map_location='cpu')
    def build_rehearsal(episodes):
        if not episodes or not getattr(adapter,'IMITATION_REHEARSAL_WEIGHT',0):return None
        from learning.imitation import Demonstrations
        focus=getattr(adapter,'IMITATION_FOCUS_HEAD',None)
        data=Demonstrations(episodes,dict(adapter.SPEC.heads),adapter.IMITATION_HEADS,focus)
        if focus and not data.focus:raise ValueError('demonstrations contain no examples for the configured focus head')
        return data
    rehearsal=build_rehearsal(demonstrations)
    def rehearsal_loss(model,random):
        return rehearsal.rehearse(model,random,getattr(adapter,'IMITATION_REHEARSAL_HEADS',adapter.IMITATION_HEADS))*adapter.IMITATION_REHEARSAL_WEIGHT
    log = (out/'train.log').open('a', buffering=1)
    def used(): return elapsed+time.monotonic()-started
    def note_selection():
        """Whether best.npz was chosen by selection_score (a finite score beat the start); if not, say so once."""
        if best_score > -math.inf:
            return True
        text = ('selection_gate_passed=False: no checkpoint passed selection_score; '
                'best.npz is only the fallback policy, not a selected one')
        print(text, flush=True); log.write(text+'\n')
        return False
    def save_resume(next_iteration):
        temp = out/'resume.tmp'
        torch.save(dict(net=net.state_dict(), optimiser=optimiser.state_dict(), rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), iteration=next_iteration, elapsed=used(),
                        team_size=team_size, best_score=best_score, policy=current, curriculum=curriculum), temp)
        temp.replace(resume_path)
    try:
        write_state(out, 'training', first-1 if first else None)
        if not saved:
            teacher_cases=adapter.training_cases(0,args.bootstrap_episodes,team_size,bootstrap=True,**case_options())
            teacher_jobs = make_jobs(args,out/current,teacher_cases,'teacher')
            demonstrated = pool(teacher_jobs)
            (out/'episodes'/'bootstrap.json').write_text(json.dumps(dict(task=args.task,jobs=teacher_cases,
                episodes=[dict(result[1],**case) for result,case in zip(demonstrated,teacher_cases)],
                actions=[result[2] for result in demonstrated])))
            demonstrations=[e for result in demonstrated for e in result[0]]
            torch.save(demonstrations,demonstration_path)
            # Cloning scripted play over a trained start policy destroys it; such a run only rehearses the demonstrations.
            if not (args.start and not getattr(adapter,'IMITATION_CLONE_OVER_START',True)):
                clone_behaviour(net,demonstrations,dict(adapter.SPEC.heads),selected_heads=adapter.IMITATION_HEADS,
                                epochs=bootstrap_epochs or getattr(adapter,'IMITATION_EPOCHS',4),balanced=getattr(adapter,'IMITATION_BALANCED',False))
            rehearsal=build_rehearsal(demonstrations)
            policy = export(net, policy); current = 'bootstrap.npz'; policy.save(out/current)
            policy.save(best)
            bootstrap_evaluation = evaluate(pool, args, best, validation)
            (out/'bootstrap-evaluation.json').write_text(json.dumps(bootstrap_evaluation, indent=2))
            best_score = selection_score(bootstrap_evaluation['summary'])
            save_resume(0)
        iterations = itertools.count(first) if args.iterations == 0 else range(first, args.iterations)
        for iteration in iterations:
            if used() >= training_limit:
                break
            weights = out/current
            cases = adapter.training_cases(iteration,args.episodes,team_size,**case_options())
            jobs = make_jobs(args,weights,cases)
            rollout_at = time.monotonic()
            flown = pool(jobs)
            rollout_seconds = time.monotonic()-rollout_at
            update_at = time.monotonic()
            stats = update_team(net, optimiser, [e for result in flown for e in result[0]], rng,
                                gamma=args.gamma, lam=args.gae_lambda,imitation_loss=rehearsal_loss if rehearsal is not None else None,
                                teacher_weight=adapter.teacher_weight(args) if hasattr(adapter,'teacher_weight') else 0., teacher_heads=tuple(adapter.IMITATION_HEADS))
            policy = export(net, policy)
            previous, current = current, f'iter{iteration:04d}.npz'
            policy.save(out/current)
            rows = [dict(result[1], **case) for result,case in zip(flown,cases)]
            summary = adapter.metrics(rows)
            record = dict(iteration=iteration, policy=previous, task=args.task,
                          jobs=cases,
                          episodes=rows, actions=[result[2] for result in flown])
            (out/'episodes'/f'iter{iteration:04d}.json').write_text(json.dumps(record))
            if iteration % 5 == 0:
                validation_summary = evaluate(pool, args, out/current, validation)['summary']
                if hasattr(adapter, 'validation_evidence'):
                    summary.update(adapter.validation_evidence(validation_summary, curriculum, args, deadline=pool.deadline))
                score = selection_score(validation_summary)
                if score > best_score:
                    best_score = score; policy.save(best)
            if curriculum is not None:
                summary[getattr(adapter, 'CURRICULUM_METRIC', 'battle_stage')] = curriculum['stage']
                previous_context = selection_context()
                curriculum = adapter.advance_curriculum(curriculum,summary)
                if previous_context != selection_context():
                    # Scores on different lessons are incomparable. Keep the previous winner, then select anew.
                    shutil.copyfile(best, out/f'best-{previous_context}.npz')
                    policy.save(best)
                    best_score = -math.inf
                    validation = validation_cases()
                    save_resume(iteration+1)
                    lesson_evaluation = evaluate(pool, args, best, validation)
                    (out/f'{selection_context()}-evaluation.json').write_text(json.dumps(lesson_evaluation, indent=2))
                    best_score = selection_score(lesson_evaluation['summary'])
            team_size = (adapter.curriculum_promote(team_size, summary, curriculum)
                         if curriculum is not None and hasattr(adapter, 'curriculum_promote')
                         else adapter.promote(team_size,summary))
            line = dict(iteration=iteration, update_s=round(time.monotonic()-update_at,3),
                        mean_return=sum(sum(e['rewards']) for r in flown for e in r[0])/sum(len(r[0]) for r in flown),
                        rollout_s=round(rollout_seconds,3), team_size=team_size, elapsed_s=round(used(),1), **summary, **stats)
            text = ' '.join(f'{k}={v}' for k,v in line.items())
            print(text, flush=True); log.write(text+'\n'); history.append(line)
            save_resume(iteration+1)
            if (out/PAUSE_FILE).exists():
                (out/PAUSE_FILE).unlink(); write_state(out, 'paused', iteration)
                return dict(paused=True, history=history, last=str(out/current))
        write_state(out, 'evaluating', history[-1]['iteration'] if history else first)
        # Fixed held-out arenas and seeds: no best-checkpoint selection on these results.
        cases = (adapter.curriculum_evaluation_cases(curriculum)
                 if curriculum is not None and hasattr(adapter, 'curriculum_evaluation_cases')
                 else adapter.evaluation_cases())
        policy.save(out/current)
        if not best.exists(): policy.save(best)
        gate_passed = note_selection()
        reports = {mode: evaluate(pool, args, best, cases, mode='scripted' if mode=='scripted' else None,
                                  muted=mode=='muted') for mode in getattr(adapter, 'EVALUATION_MODES', ('trained','scripted','muted'))}
        if 'muted' in reports:
            reports['communication_effect'] = adapter.compare(reports['trained'],reports['muted'])
        reports['versus_scripted'] = adapter.compare(reports['trained'],reports['scripted'])
        if hasattr(adapter, 'deployment_report'):
            reports['deployment'] = adapter.deployment_report(reports, args, deadline=pool.deadline)
        import hashlib
        reports['checkpoint_sha256'] = hashlib.sha256(best.read_bytes()).hexdigest()
        reports['wall_seconds'] = used()
        reports['selection_gate_passed'] = gate_passed
        if selection_context() is not None:
            reports['selection_context'] = selection_context()
        if curriculum is not None and hasattr(adapter, 'curriculum_benchmark_cases'):
            benchmark_cases = adapter.curriculum_benchmark_cases(curriculum)
            if benchmark_cases:
                reports['cave_benchmark'] = evaluate(pool, args, best, benchmark_cases)
        reports['opponent_sha256'] = expected['opponent_sha256']
        (out/'evaluation.json').write_text(json.dumps(reports, indent=2))
        write_state(out,'finished',history[-1]['iteration'] if history else first)
        return dict(paused=False,history=history,last=str(best))
    except (TimeoutError, __import__('multiprocessing').TimeoutError):
        (out/'evaluation.json').write_text(json.dumps(dict(budget_exhausted=True, wall_seconds=used(),
            message='Stopped at the wall-clock limit; completed checkpoints remain available.',
            selection_gate_passed=note_selection(), selection_context=selection_context())))
        write_state(out,'finished',history[-1]['iteration'] if history else first)
        return dict(paused=False,history=history,last=str(best if best.exists() else out/current))
    finally:
        log.close(); pool.close()
