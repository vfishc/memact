"""Private evidence scoring for reward ablations; never exposed in policy inputs."""
import math
MODES=('pixel','framing','joint','historical','legacy')

def validate_config(mode, pixel_reference, effective_gain_threshold):
    if mode not in MODES:raise ValueError('unknown reward mode')
    if not math.isfinite(pixel_reference) or not 0<pixel_reference<=1:raise ValueError('pixel_reference must be in (0,1]')
    if not math.isfinite(effective_gain_threshold) or not 0<=effective_gain_threshold<=1:raise ValueError('effective_gain_threshold must be in [0,1]')

def quality(node, required, mode, pixel_reference):
    if mode=='legacy':
        value=float(node['evidence'])
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('invalid legacy evidence')
        return value
    if not required or len(required)>2 or len(set(required))!=len(required):raise ValueError('bind one or two unique required_objects')
    scores=[]
    for obj in required:
        try:
            metrics=node['object_evidence'][obj];p=float(metrics['pixel_fraction']);f=float(metrics['framing'])
        except (KeyError,TypeError,ValueError) as e:raise ValueError('missing/unknown object evidence: '+str(obj)) from e
        if not all(math.isfinite(v) and 0<=v<=1 for v in (p,f)):raise ValueError('object evidence outside [0,1]')
        v=min(p/pixel_reference,1.)
        scores.append(v if mode=='pixel' else v*f)
    return sum(scores)/len(scores) if mode in ('pixel','framing') else min(scores)

def transition_reward(current, successor, best, cfg, valid=True, cost=1.):
    if not math.isfinite(cost) or cost<0:raise ValueError('invalid action cost')
    new_best=max(best,successor)
    gain=max(0.,successor-(best if cfg.reward_mode in ('historical','legacy') else current)) if valid else 0.
    return new_best,cfg.evidence_weight*gain-cfg.action_cost*cost-cfg.invalid_penalty*(not valid)

def metrics(initial_best, final_best, cfg):
    gain=max(0.,final_best-initial_best)
    return {'evidence_gain_points':100*gain,'effective_gain':gain>cfg.effective_gain_threshold}
