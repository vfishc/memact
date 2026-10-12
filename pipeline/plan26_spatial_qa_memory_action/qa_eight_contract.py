"""Eight slot-preserving question forms; exact letter/numeric answer contract."""
import hashlib
import json
import math
import random
import re

from qa_difficulty_contract import (CAMERA_DISTANCE_BINS, OBJECT_DISTANCE_BINS,
    CAMERA_MOTION_BINS, DISTANCE_CHANGE_BINS, DIRECTION_OPTIONS)

VERSION = 'p26_qa_choices_eight_v1'
ANSWER_INSTRUCTION = ('For every QA, output exactly {"type":"answer","answer":value}. '
    'When labeled options are provided, value must be the single uppercase option letter, not its text. '
    'Otherwise value must be a JSON number in the unit stated in the question; counts must be integers.')
PURE_INSTRUCTION = ('Consume the chronological observation stream and retain the shared causal history. '
    'Answer each [QA] using observations available so far and earlier answers in this episode. ' + ANSWER_INSTRUCTION)
CHOICE_FORMS = (
    'Which option describes {quantity}?',
    'Select the option that identifies {quantity}.',
    'What is {quantity}? Choose from the listed options.',
    'Among the choices, which one gives {quantity}?',
    'Identify {quantity} by selecting an option.',
    'Choose the answer corresponding to {quantity}.',
    'Determine {quantity} and select the matching choice.',
    'From the options below, indicate {quantity}.',
)
NUMERIC_FORMS = (
    'What is {quantity}? {unit}',
    'Estimate {quantity}. {unit}',
    'Give a numerical value for {quantity}. {unit}',
    'How much is {quantity}? {unit}',
    'Report {quantity} as a number. {unit}',
    'Determine {quantity}. {unit}',
    'Provide your numerical estimate of {quantity}. {unit}',
    'Calculate {quantity} from the observations. {unit}',
)
UNITS = {name:'meters' for name in ('camera_to_object_distance_m','camera_path_length_m',
    'single_seen_object_then_closest_distance','object_to_object_distance_m','object_size_m')}
UNITS.update(camera_rotation_deg='degrees',first_seen_timestamp_s='seconds',seen_count_by_category='count')


def extract(pattern, question):
    match = re.fullmatch(pattern, question)
    if not match: raise ValueError(f'Unsupported source wording: {question!r}; expected {pattern!r}')
    return match.groups()


def quantity(template, turn, q):
    """Parse public wording only. Never use answer/IDs to describe the referent."""
    if template in ('closest_category_then_distance','closest_category_then_relation','object_category_then_later_visibility') and turn == 0:
        template='closest_object_category'
    if template=='first_seen_category_then_timestamp':
        if turn==0: template='first_seen_object_category'
        else: return 'the first-seen timestamp of the category selected in the previous question, measured from the start of this episode', 'seconds'
    if template in ('first_seen_object_category','seen_category_order'):
        return 'the category seen earliest among the listed choices so far in this episode', None
    if template=='closest_object_category':
        return 'the category of the currently visible object closest to the camera by 3D center distance',None
    if template=='closest_category_then_distance':
        return "the current camera-to-3D-center distance range of the object selected in the previous question",None
    if template=='closest_category_then_relation':
        return "the category of the other currently visible object nearest to the previously selected object by 3D center distance",None
    if template=='object_category_then_later_visibility':
        return 'whether the object selected in the previous question is visible now',None
    if template=='camera_to_object_distance_bucket':
        ref,=extract(r'What is the camera-to-center distance range for (.+)\?',q)
        return f'the camera-to-3D-center distance range for {ref}',None
    if template=='camera_to_object_distance_m':
        ref,=extract(r'Estimate the camera-to-center distance to (.+), in meters\.',q)
        return f'the camera-to-3D-center distance to {ref}', 'meters'
    if template=='object_relative_to_camera':
        ref,=extract(r'Where is (.+) relative to the camera now\?',q)
        return f'the direction of {ref} relative to the camera now',None
    if template in ('camera_motion_bucket','camera_path_length_m','camera_rotation_deg'):
        match=re.search(r'over the last ([0-9.]+) seconds',q)
        if not match:raise ValueError(q)
        elapsed=match[1]
        measure={'camera_motion_bucket':"the range of the camera's straight-line endpoint displacement",
                 'camera_path_length_m':"the total length of the camera's trajectory",
                 'camera_rotation_deg':"the net difference between the camera's starting and ending orientations"}[template]
        return f'{measure} over the last {elapsed} seconds',UNITS.get(template)
    if template in ('became_visible_category','disappeared_category'):
        elapsed,verb=extract(r'Comparing now with ([0-9.]+) seconds ago, which category (became visible|disappeared from view)\?',q)
        return f'the category that {verb} between {elapsed} seconds ago and now',None
    if template=='approached_object_category':
        elapsed,=extract(r'Among objects visible both now and ([0-9.]+) seconds ago, which category\x27s object had the greatest camera-to-center distance decrease\?',q)
        return f"the category whose object had the largest decrease in camera-to-3D-center distance among objects visible both now and {elapsed} seconds ago",None
    if template=='closer_than_before':
        ref,elapsed=extract(r'How did the camera-to-center distance to the (.+) visible at both times change over the last ([0-9.]+) seconds\?',q)
        return f'the change in camera-to-3D-center distance to the {ref} over the last {elapsed} seconds, considering the object visible at both times',None
    if template=='first_seen_timestamp_s':
        ref,=extract(r'At what timestamp did the (.+) category first appear, in seconds\?',q)
        return f'the first-seen timestamp of the {ref} category, measured from the start of this episode','seconds'
    if template=='seen_count_by_category':
        ref,=extract(r'How many different (.+) objects have been seen so far\?',q)
        return f'the number of distinct {ref} objects observed so far in this episode','count'
    if template.startswith('single_seen_object_then_closest_'):
        elapsed,ref,tail=extract(r'Among the other objects visible at ([0-9.]+) seconds with the (.+) seen earlier and absent now, (.+)',q)
        scope=f'among the other objects visible at timestamp {elapsed} seconds alongside the {ref} seen earlier and absent now'
        if template.endswith('category'):
            return f'the category nearest to that {ref} by 3D center distance, {scope}',None
        return f'the 3D center distance from that {ref} to its nearest neighbor, {scope}','meters'
    if template=='object_size_m':
        ref,=extract(r'What is the longest physical side of the 3D bounding box of (.+), in meters\?',q)
        return f'the longest physical side length of the 3D bounding box of {ref}','meters'
    if template=='object_to_object_distance_m':
        refs,=extract(r'What is the 3D center-to-center distance between (.+), in meters\?',q)
        return f'the 3D center-to-center distance between {refs}','meters'
    if template in ('object_to_object_distance_bucket','object_spatial_proximity'):
        refs,=extract(r'Which 3D center-to-center distance range describes (.+)\?',q)
        return f'the 3D center-to-center distance range between {refs}',None
    raise ValueError(f'Unknown template/turn {template}/{turn}')


def option_values(template, turn, payload):
    if payload.get('options'):return list(payload['options'])
    table={'camera_to_object_distance_bucket':[r[0] for r in CAMERA_DISTANCE_BINS],
        'camera_motion_bucket':[r[0] for r in CAMERA_MOTION_BINS],
        'object_relative_to_camera':list(DIRECTION_OPTIONS),
        'closer_than_before':[r[0] for r in DISTANCE_CHANGE_BINS],
        'object_to_object_distance_bucket':[r[0] for r in OBJECT_DISTANCE_BINS],
        'object_spatial_proximity':[r[0] for r in OBJECT_DISTANCE_BINS]}
    if turn==1 and template=='closest_category_then_distance':return [r[0] for r in CAMERA_DISTANCE_BINS]
    if turn==1 and template=='object_category_then_later_visibility':return ['yes','no']
    if template not in table:raise ValueError(f'No option domain for {template}/{turn}')
    return table[template]


def render_question(template, turn, payload, answer, identity, seed=26):
    semantic,unit=quantity(template,turn,payload['question'])
    key=json.dumps([VERSION,seed,identity,template,turn],ensure_ascii=False)
    index=int(hashlib.sha256((key+'/wording').encode()).hexdigest(),16)%8
    forms=NUMERIC_FORMS if unit else CHOICE_FORMS
    unit_text='Answer with an integer count (objects).' if unit=='count' else f'Answer in {unit}.'
    questions=[p.format(quantity=semantic,unit=unit_text) for p in forms]
    assert len(set(questions))==8
    result={**payload,'question':questions[index],'options':None}
    private={'original_question':payload['question'],'original_answer':answer,'template':template,
             'turn':turn,'form':index,'quantity':semantic,'unit':unit,'forms':questions}
    if unit:
        if type(answer) not in (int,float) or not math.isfinite(answer):raise ValueError('Numeric GT required')
        if unit=='count' and type(answer) is not int:raise ValueError('Count must be integer')
        result.update(answer_type='numeric',unit=unit)
        return result,answer,private
    choices=option_values(template,turn,payload)
    if not 2<=len(choices)<=5 or len(set(choices))!=len(choices) or choices.count(answer)!=1:
        raise ValueError(f'Invalid choice domain {template}: {answer!r}, {choices!r}')
    random.Random(key+'/options').shuffle(choices)
    mapping={chr(65+i):v for i,v in enumerate(choices)}
    result.update(options=mapping,answer_type='choice')
    letter=next(k for k,v in mapping.items() if v==answer)
    private['option_mapping']=mapping
    return result,letter,private


def valid_answer(value, question):
    if question.get('answer_type')=='choice':return type(value) is str and value in question['options']
    if question.get('answer_type')=='numeric':
        return type(value) in (int,float) and math.isfinite(value) and (question.get('unit')!='count' or type(value) is int)
    raise ValueError('Question missing answer contract')


def full_instruction(original):
    start=original.index('If options exist answer with one complete category option, not a letter.')
    end=original.index('Never write memory during QA.',start)
    return original[:start]+ANSWER_INSTRUCTION+' '+original[end:]
