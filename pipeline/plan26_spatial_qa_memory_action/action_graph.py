"""Stateful, question-local execution of private cached camera graphs."""
import copy

COMMANDS = [dict(type='action', name=n, arguments={'angle_deg':45} if n.startswith('rotate_') else {'distance_m':0.5})
            for n in ('rotate_left_45','rotate_right_45','move_forward_d','move_left_d','move_right_d')]

class ActionCoverageError(RuntimeError):
    """Missing counterfactual data is not a physically invalid action."""

class GraphActionExecutor:
    def __init__(self, graph, max_attempts=3):
        self.graph = copy.deepcopy(graph)
        self.max_attempts = max_attempts
        self.nodes = self.graph['nodes']
        self.edges = {}
        for e in self.graph['edges']:
            key = (e['source'], e['action'])
            if key in self.edges or e['action'] not in {c['name'] for c in COMMANDS}:
                raise ValueError('duplicate or unknown graph edge')
            if e['source'] not in self.nodes or e['target'] not in self.nodes or type(e['valid']) is not bool:
                raise ValueError('invalid graph edge')
            if not e['valid'] and e['target'] != e['source']:
                raise ValueError('invalid physical action must remain at its source')
            self.edges[key] = e
        self.begin_question(0, {})

    def begin_question(self, index, payload):
        roots = self.graph.get('question_roots', {})
        self.node = roots.get(str(index), self.graph['root'])
        if self.node not in self.nodes:
            raise ValueError('unknown question root')
        self.attempts = 0

    def preflight(self):
        frontier = set(self.graph.get('question_roots', {}).values()) or {self.graph['root']}
        for _ in range(self.max_attempts):
            nxt = set()
            for node in frontier:
                for c in COMMANDS:
                    e = self.edges.get((node,c['name']))
                    if e is None:
                        raise ActionCoverageError(f'missing branch {node}/{c["name"]}')
                    target = self.nodes[e['target']]
                    if not isinstance(target.get('frame'),str):
                        raise ValueError('graph node requires a public frame handle')
                    nxt.add(e['target'])
            frontier = nxt
        return {'status':'pass','nodes':len(self.nodes),'edges':len(self.edges),'max_attempts':self.max_attempts}

    def __call__(self, command):
        if self.attempts >= self.max_attempts:
            raise ValueError('action attempt budget exhausted')
        if command not in COMMANDS:
            self.attempts += 1
            return {'status':'invalid_action','reason':'unsupported_command','remaining_actions':self.max_attempts-self.attempts}
        edge = self.edges.get((self.node,command['name']))
        if edge is None:
            raise ActionCoverageError(f'missing branch {self.node}/{command["name"]}')
        self.attempts += 1
        self.node = edge['target']
        result = {'status':'ok' if edge['valid'] else 'invalid_action',
                  'action':copy.deepcopy(command),'remaining_actions':self.max_attempts-self.attempts}
        if edge['valid']:
            result['frame'] = self.nodes[self.node]['frame']
        else:
            result['reason'] = 'physical_constraint'
        return result


def make_executor(data, max_attempts=3):
    if data.get('action_graph'):
        executor = GraphActionExecutor(data['action_graph'],max_attempts)
        executor.preflight()
        missing = {n['frame'] for n in executor.nodes.values()} - data['media'].keys()
        if missing:
            raise ActionCoverageError('graph frames missing from media mapping')
        return executor
    if data.get('actions'):
        from action_stream_adapter import CachedActionExecutor
        if max_attempts > 1:
            raise ActionCoverageError('legacy one-step cache cannot evaluate a multistep budget; supply action_graph')
        missing = [c['name'] for c in COMMANDS
                   if not any(e['action'] == c for e in data['actions'])]
        if missing:
            raise ActionCoverageError('legacy one-step cache has unknown outcomes: ' + ','.join(missing))
        return CachedActionExecutor(data['actions'])
    return None
