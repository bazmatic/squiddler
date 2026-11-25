"""Creature-independent recurrent PPO losses and sequence batching."""
import numpy as np

REWARD_SCALE = 0.1

def pack(episodes, mean, scale):
    import torch
    batch, length = len(episodes), max(len(e['senses']) for e in episodes)
    x = torch.zeros(batch, length, len(mean))
    controls = torch.zeros(batch, length, len(episodes[0]['controls'][0]), dtype=torch.long)
    rewards = torch.zeros(batch, length)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    temperatures = torch.ones(batch, length)       # what each choice was sampled at (training's panic raises it)
    for i, e in enumerate(episodes):
        n = len(e['senses'])
        x[i, :n] = torch.from_numpy((np.asarray(e['senses'], dtype=np.float32) - mean) / scale)
        controls[i, :n] = torch.from_numpy(np.asarray(e['controls'], dtype=np.int64))
        rewards[i, :n] = torch.from_numpy(np.asarray(e['rewards'], dtype=np.float32)) * REWARD_SCALE
        mask[i, :n] = True
        if 'temperatures' in e:
            temperatures[i, :n] = torch.from_numpy(np.asarray(e['temperatures'], dtype=np.float32)[:n])
    return {'x': x, 'controls': controls, 'rewards': rewards, 'mask': mask, 'temperatures': temperatures}


def joint_logp(logits, controls, temperatures=None):
    """The log-probability of the chosen controls, each scored at the temperature it was sampled at."""
    import torch
    total = 0.0
    for k, head in enumerate(logits):
        z = logits[head] if temperatures is None else logits[head] / temperatures[..., None]
        logp = torch.log_softmax(z, dim=-1)
        total = total + logp.gather(-1, controls[..., k:k + 1]).squeeze(-1)
    return total


def gae(rewards, values, mask, gamma=0.99, lam=0.95):
    """Advantages and returns; each episode ends at its last unmasked step (terminal, value 0 after)."""
    import torch
    batch, length = rewards.shape
    zero = torch.zeros(batch)
    advantages = torch.zeros_like(rewards)
    running = zero
    for t in reversed(range(length)):
        has_next = mask[:, t + 1] if t + 1 < length else torch.zeros(batch, dtype=torch.bool)
        next_value = torch.where(has_next, values[:, t + 1], zero) if t + 1 < length else zero
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        running = delta + gamma * lam * torch.where(has_next, running, zero)
        advantages[:, t] = torch.where(mask[:, t], running, zero)
    return advantages, advantages + torch.where(mask, values, torch.zeros_like(values))


def _kl_and_entropy(logits, ref_logits):
    import torch
    kl, entropy = 0.0, 0.0
    for head in logits:
        logp = torch.log_softmax(logits[head], dim=-1)
        p = logp.exp()
        kl = kl + (p * (logp - torch.log_softmax(ref_logits[head], dim=-1))).sum(-1)
        entropy = entropy - (p * logp).sum(-1)
    return kl, entropy


def improve(net, ref, optimiser, episodes, mean, scale, kl_coef, rng, clip=0.2, epochs=4, minibatches=4,
            gamma=0.99, lam=0.95, value_coef=0.5, entropy_coef=0.003):
    """One PPO update on a batch of whole episodes. Returns the mean loss terms of the last epoch."""
    import torch
    batch = pack(episodes, mean, scale)
    x, controls, mask = batch['x'], batch['controls'], batch['mask']
    with torch.no_grad():
        logits, values = net(x)
        old_logp = joint_logp(logits, controls, batch['temperatures'])
        advantages, returns = gae(batch['rewards'], values, mask, gamma, lam)
        ref_logits, _ = ref(x)
        chosen = advantages[mask]
        advantages = torch.where(mask, (advantages - chosen.mean()) / (chosen.std() + 1e-8),
                                 torch.zeros_like(advantages))
    stats = {}
    for _ in range(epochs):
        order = rng.permutation(len(episodes))
        sums, count = {'policy_loss': 0.0, 'value_loss': 0.0, 'kl': 0.0, 'entropy': 0.0,
                       'clip_fraction': 0.0}, 0
        for chunk in np.array_split(order, min(minibatches, len(episodes))):
            index = torch.as_tensor(chunk, dtype=torch.long)
            m = mask[index]
            logits, values = net(x[index])
            # Clamped so a policy that has moved far within one update cannot overflow the ratio's gradient.
            logp = joint_logp(logits, controls[index], batch['temperatures'][index])
            ratio = torch.exp(torch.clamp(logp - old_logp[index], -20.0, 20.0))
            a = advantages[index]
            surrogate = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
            policy_loss = -surrogate[m].mean()
            value_loss = ((values - returns[index]) ** 2)[m].mean()
            kl, entropy = _kl_and_entropy(logits, {k: v[index] for k, v in ref_logits.items()})
            loss = (policy_loss + value_coef * value_loss + kl_coef * kl[m].mean()
                    - entropy_coef * entropy[m].mean())
            if not torch.isfinite(loss):
                continue
            optimiser.zero_grad()
            loss.backward()
            # A finite loss can still have an infinite gradient, which clipping turns into NaN weights: skip it.
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)):
                optimiser.zero_grad()
                continue
            optimiser.step()
            for key, value in (('policy_loss', policy_loss), ('value_loss', value_loss), ('kl', kl[m].mean()),
                               ('entropy', entropy[m].mean()),
                               ('clip_fraction', ((ratio - 1).abs() > clip)[m].float().mean())):
                sums[key] += float(value.detach())
            count += 1
        stats = {key: value / max(count, 1) for key, value in sums.items()}
    return stats



def bootstrapped_gae(rewards, values, valid, terminated, bootstrap, gamma=0.99, lam=0.95):
    """A horizon bootstraps its final state; only true termination cuts future value."""
    import torch
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], dtype=rewards.dtype)
    following = bootstrap
    for t in reversed(range(rewards.shape[1])):
        continuation = (~terminated[:, t]).to(rewards.dtype)
        delta = rewards[:, t]+gamma*following*continuation-values[:, t]
        running = torch.where(valid[:, t], delta+gamma*lam*continuation*running, running)
        advantages[:, t] = torch.where(valid[:, t], running, torch.zeros_like(running))
        following = torch.where(valid[:, t], values[:, t], following)
    return advantages, advantages+values


def masked_statistics(logits, actions, legal, enabled, continuous=()):
    import torch
    logp = torch.zeros(actions.shape[:2], device=actions.device)
    entropy = torch.zeros_like(logp)
    for i, name in enumerate(logits):
        if name in continuous:
            from learning.continuous import statistics
            lp, ent = statistics(logits[name], actions[...,i])
            logp += lp*enabled[...,i]
            entropy += ent*enabled[...,i]
            continue
        z = logits[name].masked_fill(~legal[name], -1e9)
        lp = torch.log_softmax(z, -1)
        mask = enabled[..., i]
        logp += lp.gather(-1, actions[..., i:i+1].long()).squeeze(-1)*mask
        entropy -= (lp.exp()*lp).sum(-1)*mask
    return logp, entropy


def update_team(net, optimiser, episodes, rng, epochs=3, gamma=0.99, lam=0.95, clip=0.2, imitation_loss=None,
                teacher_weight=0., teacher_heads=()):
    """Shared recurrent PPO with optional privileged value inputs and per-agent action masks.

    Agent trajectories share team rewards. Dead agents have no actor loss; the team critic
    continues until the encounter ends, retaining credit for delayed projectile hits.
    """
    import torch
    b, length = len(episodes), max(len(e['x']) for e in episodes)
    d, context = len(episodes[0]['x'][0]), len(episodes[0]['critic'][0])
    h = len(net.heads)
    x, critic = torch.zeros(b, length, d), torch.zeros(b, length, context)
    actions = torch.zeros(b, length, h, dtype=torch.float32)
    valid = torch.zeros(b, length, dtype=torch.bool)
    alive, done = valid.clone(), valid.clone()
    driven = valid.clone()       # scheduled sampling: steps the expert took; imitated and valued, not policy-gradient
    rewards = torch.zeros(b, length)
    enabled = torch.zeros(b, length, h)
    legal = {k: torch.ones(b, length, head.out_features, dtype=torch.bool) for k, head in net.heads.items()}
    bootstrap = torch.zeros(b)
    # DAgger: what the teacher would have done in each state the policy visited (episodes recorded with a labeller).
    labelled = torch.zeros(b, length, h, dtype=torch.bool)
    labels = torch.zeros(b, length, h)
    phases = torch.zeros(b, length, dtype=torch.long)    # optional 9th label value: which phase of the task the label teaches
    for i, e in enumerate(episodes):
        for t, label in enumerate(e.get('teacher', ())):
            if label is not None and teacher_weight:
                label = np.asarray(label, np.float32)
                given = torch.as_tensor(label[:h])
                labelled[i, t] = ~torch.isnan(given)      # NaN: this head is not taught in this state
                labels[i, t] = torch.nan_to_num(given)
                if len(label) > h:
                    phases[i, t] = int(label[h])
        n = len(e['x']); valid[i, :n] = True
        x[i, :n] = torch.as_tensor(np.asarray(e['x']))
        critic[i, :n] = torch.as_tensor(np.asarray(e['critic']))
        actions[i, :n] = torch.as_tensor(np.asarray(e['actions']))
        alive[i, :n] = torch.as_tensor(e['alive'])
        if e.get('driven'):
            driven[i, :n] = torch.as_tensor(e['driven'])
        rewards[i, :n] = torch.as_tensor(e['rewards'])
        enabled[i, :n] = torch.as_tensor(np.asarray(e['enabled']))
        done[i, n-1] = e['terminated']
        for k in legal:
            legal[k][i, :n] = torch.as_tensor(np.asarray(e['legal'][k]))
        if not e['terminated']:
            with torch.no_grad():
                bootstrap[i] = net.critic(torch.as_tensor(e['final_critic']).reshape(1, -1)).squeeze()
    with torch.no_grad():
        logits, values = net(x, critic)
        old_logp, _ = masked_statistics(logits, actions, legal, enabled, getattr(net,'continuous',()))
        advantages, returns = bootstrapped_gae(rewards, values, valid, done, bootstrap, gamma, lam)
        acting = alive & ~driven
        chosen = advantages[acting]
        if not alive.any():
            return {'policy_loss': 0., 'value_loss': 0., 'entropy': 0.}
        if chosen.numel():
            advantages = (advantages-chosen.mean())/chosen.std(unbiased=False).clamp_min(1e-6)
        else:                                  # an all-expert batch: nothing for the policy gradient, still imitation and value
            advantages = torch.zeros_like(advantages)
    stats = {}
    for _ in range(epochs):
        for chunk in np.array_split(rng.permutation(b), min(4, b)):
            i = torch.as_tensor(chunk, dtype=torch.long)
            logits, values = net(x[i], critic[i])
            logp, entropy = masked_statistics(logits, actions[i], {k:v[i] for k,v in legal.items()}, enabled[i], getattr(net,'continuous',()))
            ratio = (logp-old_logp[i]).clamp(-20, 20).exp()
            surrogate = torch.minimum(ratio*advantages[i], ratio.clamp(1-clip, 1+clip)*advantages[i])
            mask = alive[i]                     # every step the creature lived: imitation applies to all of them
            if not mask.any():
                continue
            acting = mask & ~driven[i]         # the steps the policy itself chose: the only ones PPO may score
            if acting.any():
                policy_loss = -surrogate[acting].mean()
                ent = entropy[acting].mean()
            else:
                policy_loss, ent = surrogate.sum()*0, entropy.sum()*0
            value_loss = (values-returns[i]).square()[valid[i]].mean()
            imitation = imitation_loss(net,rng) if imitation_loss is not None else policy_loss*0
            teacher_loss = policy_loss*0
            if teacher_weight and (labelled[i] & mask[..., None]).any():
                for name in teacher_heads:
                    k = list(net.heads).index(name)
                    taught = labelled[i][..., k] & mask
                    if taught.any():
                        # Each phase present counts equally, however many decisions it spans (one phase: a plain mean).
                        phase = phases[i][taught]
                        counts = torch.bincount(phase)
                        weight = 1/(counts[phase]*(counts > 0).sum())
                        if name in getattr(net, 'continuous', ()):
                            error = (logits[name][..., 0].tanh()-labels[i][..., k].tanh()).square()[taught]
                        else:
                            z = logits[name].masked_fill(~legal[name][i], -1e9)[taught]
                            error = torch.nn.functional.cross_entropy(z, labels[i][..., k][taught].long(), reduction='none')
                        teacher_loss = teacher_loss+(error*weight).sum()/len(teacher_heads)
            imitation = imitation+teacher_weight*teacher_loss
            loss = policy_loss+0.5*value_loss-0.01*ent+imitation
            if not torch.isfinite(loss):
                raise ValueError('non-finite PPO loss')
            optimiser.zero_grad(); loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(net.parameters(), 1)):
                raise ValueError('non-finite PPO gradient')
            optimiser.step()
            stats = dict(policy_loss=float(policy_loss.detach()), value_loss=float(value_loss.detach()),
                         entropy=float(ent.detach()), imitation_loss=float(imitation.detach()),
                         teacher_loss=float(teacher_loss.detach()))
    return stats
