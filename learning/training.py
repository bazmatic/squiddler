"""Shared training model and trajectory collection; numerical interfaces only."""
import numpy as np
from learning.models import network_class, load_into


def build_network(policy, critic_size):
    import torch
    base = network_class(len(policy.spec.features), dict(policy.spec.heads))
    class TeamNetwork(base):
        def __init__(self):
            super().__init__(policy.hidden, policy.params['embed_w'].shape[0])
            self.continuous = policy.spec.continuous
            self.critic = torch.nn.Sequential(torch.nn.Linear(critic_size, 128), torch.nn.Tanh(),
                                              torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))
        def forward(self, x, context):
            logits, _ = super().forward(x)
            return logits, self.critic(context).squeeze(-1)
    net = TeamNetwork()
    load_into(net, policy, dict(policy.spec.heads))
    return net


def collect(env, policy, seed, scenario, greedy=False, teacher=None, labeller=None, expert=None, handover=None, mix=0., leads=None, assist=None):
    """One episode. `labeller(env, agent)`, if given, is asked what the teacher would do in every state the policy
    actually visits; the answers are recorded as e['teacher'] (DAgger-style supervision for the update).

    Scheduled sampling: with probability `mix`, drawn once per episode from the seed, the episode is expert-led. The
    `expert(env, agent, action)` chooses the actions instead of the policy until `handover(env, agent)` first returns
    true (never, if there is no handover: the expert flies the whole episode). Episodes `leads(env)` refuses stay the policy's. Each step records whether the expert
    acted, as e['driven'], so the update can learn from those steps without treating them as the policy's own."""
    coin = mix > 0 and expert is not None and np.random.default_rng(np.random.SeedSequence([seed, 7919])).random() < mix
    snap = env.reset(seed, **scenario)
    led = coin and (leads is None or leads(env))      # `leads(env)`: whether the task has an expert for this episode
    gave = set()

    def handed(env, agent):
        if agent not in gave and handover(env, agent):
            gave.add(agent)
        return agent in gave
    ids = tuple(snap.observations)
    states = {i: policy.initial_state() for i in ids}
    rngs = {i: np.random.default_rng(np.random.SeedSequence([seed, i])) for i in ids}
    episodes = {i: dict(x=[], critic=[], actions=[], alive=[], rewards=[], enabled=[],
                        legal={k: [] for k,_ in policy.spec.heads}, teacher=[], driven=[]) for i in ids}
    while True:
        # New agents start independent trajectories at birth, with zero recurrent state.
        # Prefix padding would incorrectly advance their GRU before their first action.
        for i in snap.observations:
            if i in episodes:
                continue
            states[i] = policy.initial_state()
            rngs[i] = np.random.default_rng(np.random.SeedSequence([seed, i]))
            episodes[i] = dict(x=[], critic=[], actions=[], alive=[], rewards=[], enabled=[],
                               legal={k:[] for k,_ in policy.spec.heads}, teacher=[], driven=[])
        ids = tuple(episodes)
        actions = {}
        for i in ids:
            e = episodes[i]
            alive = i in snap.observations
            x = snap.observations.get(i, np.zeros(len(policy.spec.features), np.float32))
            masks = snap.masks.get(i, {k: np.ones(n, bool) for k,n in policy.spec.heads})
            if alive:
                action, states[i] = policy.choose(x, states[i], masks, greedy, rngs[i])
                if teacher is not None:
                    action = teacher(env, i, action)
                driving = led and not (handover is not None and handed(env, i))
                keeps = None
                if driving:
                    action = expert(env, i, action)
                elif led and assist is not None:
                    # After the handover the expert may still drive every head but the ones the policy keeps; PPO then scores only those.
                    helped = assist(env, i, action)
                    if helped is not None:
                        action, keeps = helped
                actions[i] = action
                e['teacher'].append(labeller(env, i) if labeller else None)
                e['driven'].append(bool(driving))
            else:
                action = tuple(0 for _ in policy.spec.heads)
                e['teacher'].append(None)
                e['driven'].append(False)
            active = env.adapter.enabled(x, action, masks) if alive else [False]*len(policy.spec.heads)
            if alive and keeps is not None:
                active = [a and j in keeps for j,a in enumerate(active)]
            e['x'].append((x-policy.mean)/policy.scale); e['critic'].append((snap.agent_critics or {}).get(i,snap.critic)); e['actions'].append(action)
            e['alive'].append(alive); e['enabled'].append(active)
            for k in e['legal']:
                e['legal'][k].append(masks[k])
        transition = env.step(actions)
        for i,e in episodes.items():
            e['rewards'].append(transition.reward if transition.agent_rewards is None else transition.agent_rewards.get(i,0.))
        snap = transition.snapshot
        if transition.terminated or transition.truncated:
            break
    for i,e in episodes.items():
        e.update(terminated=transition.terminated, final_critic=(snap.agent_critics or {}).get(i,snap.critic))
    return list(episodes.values()), transition.events, env.action_trace


def clone_behaviour(net, episodes, head_names, selected_heads=None, epochs=4, lr=1e-3, balanced=False):
    """Warm-start only the action heads explicitly selected by the task adapter."""
    import torch
    if balanced:
        from learning.imitation import Demonstrations
        Demonstrations(episodes,head_names,selected_heads or head_names).fit(net,epochs=epochs,lr=lr)
        return
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    names = tuple(selected_heads or head_names)
    indices = {k:i for i,k in enumerate(head_names)}
    for _ in range(epochs):
        for e in episodes:
            x = torch.as_tensor(np.asarray(e['x'])).unsqueeze(0)
            y = torch.as_tensor(np.asarray(e['actions']))
            mask = torch.as_tensor(e['alive'])
            if not mask.any():
                continue
            logits, _ = net(x, torch.as_tensor(np.asarray(e['critic'])).unsqueeze(0))
            losses = []
            for k in names:
                active = mask & torch.as_tensor(np.asarray(e['enabled'])[:,indices[k]], dtype=torch.bool)
                if active.any():
                    labels = y[active,indices[k]]
                    if k in getattr(net,'continuous',()):
                        losses.append(torch.nn.functional.mse_loss(logits[k][0,active,0].tanh(), labels.tanh()))
                        continue
                    labels = labels.long()
                    counts = torch.bincount(labels,minlength=logits[k].shape[-1]).float()
                    weights = (len(labels)/(len(counts)*counts.clamp_min(1))).clamp(max=8)
                    losses.append(torch.nn.functional.cross_entropy(logits[k][0][active],labels,weight=weights))
            if not losses:
                continue
            loss = sum(losses)
            optimiser.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1); optimiser.step()
