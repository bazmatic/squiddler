"""Small, isolated DAgger experiment for the Campaign Hunter's open-exit lesson.

The teacher labels the student's visited states on practice level 1032. A small
fraction of teacher actions changes the trajectory to provide recovery states.
Normal campaign observations anchor the policy via KL rehearsal. Nothing here
installs a policy or changes campaign gameplay.
"""
import argparse
import dataclasses
import json
import random
from pathlib import Path

import numpy as np

from experiments.campaign_exit_teacher import ExitTeacher
from experiments.campaign_rollout import fly_campaign
from experiments.student import StudentSpec
from hunter.campaign_pilot import CampaignPilot
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
from hunter.neural.controls import to_action
from hunter.neural.policy import NeuralPolicy


class LabeledPilot:
    """Let the student fly, recording an exit-teacher action at each decision."""
    def __init__(self, policy, seed, beta=0.0, greedy=False, vary=True):
        self.student = CampaignPilot(policy, 6, greedy=greedy, rng=np.random.default_rng(seed * 101 + 1032),
                                     sticky_targets=True, target_recovery=True)
        self.teacher = ExitTeacher()
        self.seed, self.beta, self.vary = seed, beta, vary
        self.rng = np.random.default_rng(seed + 90000)
        self.labels, self.overrides = [], 0
        self.pending = None
        self.initialized = False
        self.student.on_observed(self._label)

    def __getattr__(self, name):
        return getattr(self.student, name)

    def _label(self, senses, last_controls):
        self.labels.append(self.pending)

    def reset(self):
        self.student.reset()
        self.teacher.reset()
        self.labels.clear()
        self.overrides = 0
        self.initialized = False

    def decide(self, game):
        if not self.initialized:
            if self.vary:
                # A seeded sample of the final passage, at different headings
                # and momenta. The fixed original scenario is held out too.
                rng = np.random.default_rng(self.seed)
                ship = game.ship
                col = int(rng.choice([8, 10, 12, 14, 16, 18, 20, 22, 24]))
                ship.x, ship.y = game.maze.position_calculator.grid_center_to_screen(col, 28)
                ship.prev_x, ship.prev_y = ship.x, ship.y
                ship.angle = float(rng.uniform(0, 360))
                ship.vx, ship.vy = map(float, rng.uniform(-1.5, 1.5, 2))
            self.initialized = True
        ctrl = self.student.controller
        if ctrl.frames % ctrl.decision_frames == 0:
            teacher = self.teacher.decide(game)
            self.pending = None if teacher is None else (teacher.turn + 1, int(teacher.thrust), int(teacher.fire))
            if self.pending is not None and self.rng.random() < self.beta:
                ctrl.override = to_action(self.pending)
                self.overrides += 1
        return self.student.decide(game)


def collect(policy_path, seed, beta=0.0, greedy=False, vary=True):
    """Return the real campaign trajectory and labels, including optional teacher interventions."""
    random.seed(seed)
    np.random.seed(seed)
    policy = NeuralPolicy.load(policy_path, CAMPAIGN_FEATURE_NAMES)
    pilot = LabeledPilot(policy, seed, beta=beta, greedy=greedy, vary=vary)
    episode = fly_campaign(1032, seed, StudentSpec(weights=str(policy_path), greedy=greedy), horizon=60,
                           pilot=pilot, sticky_targets=True, target_recovery=True, graded_shooting=True)
    n = len(episode['senses'])
    labels = pilot.labels[:n]
    assert len(labels) == n
    return episode, labels, pilot.overrides


def normal_episode(policy_path, level, seed):
    random.seed(seed)
    np.random.seed(seed)
    return fly_campaign(level, seed, StudentSpec(weights=str(policy_path), greedy=False, sample_seed=seed),
                        horizon=60, sticky_targets=True, target_recovery=True, graded_shooting=True)


def fit(source, data, rehearsal, destination, epochs=3, lr=0.00003, kl_weight=3.0):
    """Fit turn/thrust labels with KL rehearsal against the source on ordinary flights."""
    import torch
    import torch.nn.functional as F
    from learning.models import network_class, load_into
    from hunter.neural.policy import HEAD_SIZES
    from tools.train_neural import to_numpy_policy
    torch.set_num_threads(1)
    torch.manual_seed(0)
    policy = NeuralPolicy.load(source, CAMPAIGN_FEATURE_NAMES)
    cls = network_class(len(CAMPAIGN_FEATURE_NAMES), HEAD_SIZES)
    net = cls(policy.hidden, policy.params['embed_w'].shape[0])
    reference = cls(policy.hidden, policy.params['embed_w'].shape[0])
    load_into(net, policy, HEAD_SIZES)
    load_into(reference, policy, HEAD_SIZES)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    mean, scale = policy.mean, policy.scale
    def tensor(senses):
        return torch.from_numpy((np.asarray(senses, np.float32) - mean) / scale).unsqueeze(0)
    practice = [(tensor(e['senses']), labels) for e, labels in data if len(e['senses'])]
    normal = [tensor(e['senses']) for e in rehearsal if len(e['senses'])]
    if not practice or not normal:
        raise ValueError('Need both labelled practice and ordinary campaign rehearsal')
    for epoch in range(epochs):
        imitation_sum = rehearsal_sum = 0.0
        for i, (x, labels) in enumerate(practice):
            valid = torch.tensor([label is not None for label in labels], dtype=torch.bool)
            if not valid.any():
                continue
            targets = torch.tensor([label if label is not None else (0, 0, 0) for label in labels], dtype=torch.long)
            logits, _ = net(x)
            imitation = sum(F.cross_entropy(logits[k][0, valid], targets[valid, j])
                            for j, k in enumerate(('turn', 'thrust'))) / 2
            rx = normal[i % len(normal)]
            actual, _ = net(rx)
            with torch.no_grad():
                expected, _ = reference(rx)
            kl = sum(F.kl_div(F.log_softmax(actual[k], dim=-1), F.softmax(expected[k], dim=-1),
                              reduction='batchmean') / rx.shape[1] for k in ('turn', 'thrust', 'fire'))
            loss = imitation + kl_weight * kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            imitation_sum += float(imitation.detach())
            rehearsal_sum += float(kl.detach())
        print(f'epoch {epoch+1}: imitation={imitation_sum/len(practice):.4f} rehearsal_kl={rehearsal_sum/len(practice):.5f}', flush=True)
    exported = to_numpy_policy(net, mean, scale)
    exported.feature_names = CAMPAIGN_FEATURE_NAMES
    exported.save(destination)
    NeuralPolicy.load(destination, CAMPAIGN_FEATURE_NAMES)


def main():
    from experiments.run import _use_dummy_drivers
    _use_dummy_drivers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--beta', type=float, default=.2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=.00003)
    parser.add_argument('--kl-weight', type=float, default=3.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(source=args.source, rounds=args.rounds, episodes=args.episodes, beta=args.beta,
                    teacher='experiments.campaign_exit_teacher.ExitTeacher', lr=args.lr, epochs=args.epochs,
                    kl_weight=args.kl_weight, selected_heads=['turn','thrust'], seed_sets=[])
    (out / 'experiment.json').write_text(json.dumps(metadata, indent=2))
    print('Collecting ordinary campaign rehearsal', flush=True)
    rehearsal = [normal_episode(args.source, level, 300 + level + i)
                 for level in (2, 11, 32) for i in range(2)]
    for index, episode in enumerate(rehearsal):
        np.savez_compressed(out / f'rehearsal{index}.npz', senses=episode['senses'], controls=episode['controls'])
    print('rehearsal decisions', sum(len(e['senses']) for e in rehearsal), flush=True)
    policy = args.source
    dataset = []
    for round_index in range(args.rounds):
        seeds = list(range(300 + round_index * args.episodes, 300 + (round_index + 1) * args.episodes))
        metadata['seed_sets'].append(seeds)
        results = []
        for seed in seeds:
            episode, labels, overrides = collect(policy, seed, beta=args.beta)
            dataset.append((episode, labels))
            saved_labels = np.asarray([label if label is not None else (-1,-1,-1) for label in labels], dtype=np.int8)
            np.savez_compressed(out / f'round{round_index+1}-seed{seed}.npz',
                                senses=episode['senses'], labels=saved_labels, controls=episode['controls'])
            results.append(dict(seed=seed, result=dataclasses.asdict(episode['result']), overrides=overrides))
            print(f"round {round_index+1} seed {seed}: {episode['result'].end}, overrides={overrides}", flush=True)
        (out / f'round{round_index+1}-episodes.json').write_text(json.dumps(results, indent=1))
        destination = out / f'candidate{round_index+1}.npz'
        fit(args.source, dataset, rehearsal, destination, epochs=args.epochs, lr=args.lr, kl_weight=args.kl_weight)
        policy = str(destination)
        (out / 'experiment.json').write_text(json.dumps(metadata, indent=2))
        print('Saved', destination, flush=True)


if __name__ == '__main__':
    main()
