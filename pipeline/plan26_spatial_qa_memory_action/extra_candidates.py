"""Existing-media candidates for the final additive evaluation bank."""
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'extension_round1'), str(HERE.parent)]
import numpy as np
from extra_templates import EpisodeFacts, make, memory_candidates, validate_memory
from room_area import validate_rooms, contains
from qa_semantic_contract import categories_conflict, clean_category_options, VISUAL_CATEGORY_ALLOWLIST
from camera_geometry import euler_xyz_matrix


def options(answer, actual, scene):
    pool = list(actual) + [c for c in sorted(VISUAL_CATEGORY_ALLOWLIST) if c not in scene]
    _, choices, _ = clean_category_options(answer, pool)
    return choices[:4] if len(choices) >= 4 else None


def spatial_candidates(f):
    scene = set(f.categories.values())
    for n in range(2, len(f.frames)-1, 3):
        # Private sampling uses run interiors; model never receives future frames.
        stable = set.intersection(*(f.frames[i]['visible'] for i in (n-1,n,n+1)))
        counts = f.counts(n)
        ids = sorted(o for o in stable if counts[f.categories[o]] == 1)
        for anchor in ids:
            other = [o for o in ids if o != anchor and not categories_conflict(f.categories[o], f.categories[anchor])]
            ranked = sorted(other, key=lambda o: (math.dist(f.center(o),f.center(anchor)),o))
            distinct = []
            for o in ranked:
                if not any(categories_conflict(f.categories[o],f.categories[x]) for x in distinct):
                    distinct.append(o)
            if len(distinct) < 2: continue
            distances = {o:math.dist(f.center(o),f.center(anchor)) for o in distinct}
            if distances[distinct[1]]-distances[distinct[0]] < .4: continue
            opts = options(f.categories[distinct[0]], [f.categories[o] for o in distinct[:4]], scene)
            if not opts: continue
            s = make(f,'closest_neighbor_category',n,
                f'the listed category whose currently visible object has its 3D center closest to the {f.categories[anchor]}',
                f.categories[distinct[0]],opts,evidence=[n],categories=[f.categories[anchor]])
            s['private_geometry']={'anchor':anchor,'distances':distances}; yield s
            break
        regions = f.regions(n)
        ids2 = [o for o in ids if o in regions]
        for anchor in ids2:
            _,ax,ay = regions[anchor]
            found = False
            for side,sign in [('left',-1),('right',1)]:
                positive = [o for o in ids2 if sign*(regions[o][1]-ax)>0 and regions[o][2]==ay
                            and not categories_conflict(f.categories[o],f.categories[anchor])]
                negative = [o for o in ids2 if sign*(regions[o][1]-ax)<0
                            and not categories_conflict(f.categories[o],f.categories[anchor])]
                if not positive or not negative: continue
                target = positive[0]
                opts = options(f.categories[target],[f.categories[target]]+[f.categories[o] for o in negative],scene)
                if not opts: continue
                s=make(f,'fixed_reference_relation',n,
                    f'the listed category whose currently visible object is to the {side} of the {f.categories[anchor]} in the image at the time of this question',
                    f.categories[target],opts,evidence=[n],categories=[f.categories[anchor],f.categories[target]])
                s['private_geometry']={'anchor':anchor,'target':target,'regions':regions,'side':side};yield s
                found=True;break
            if found:break
        if f.source != 'sim_front3d': continue
        for group in itertools.combinations(ids,4):
            if any(categories_conflict(f.categories[a],f.categories[b]) for a,b in itertools.combinations(group,2)):continue
            if any('bbox_min' not in f.objects[o] or 'bbox_max' not in f.objects[o] for o in group):continue
            heights={o:f.objects[o]['bbox_max'][2]-f.objects[o]['bbox_min'][2] for o in group}
            rank=sorted(group,key=lambda o:heights[o],reverse=True)
            if heights[rank[0]]-heights[rank[1]]<.3:continue
            s=make(f,'multi_candidate_height',n,
                'the listed category whose currently visible object has the greatest vertical height (not elevation above the floor)',
                f.categories[rank[0]],[f.categories[o] for o in group],evidence=[n],categories=[f.categories[o] for o in group])
            s['private_geometry']={'height_m':heights,'axis':'Sim world Z'};yield s;break


def area_candidates(f):
    if f.source != 'sim_front3d':return
    path=Path(f.record['paths']['room_info']);rooms,_=validate_rooms(json.loads(path.read_text()))
    for name,room in rooms.items():
        points=room['polygon'];boundary=np.array([[a[0]+(b[0]-a[0])*t/8,a[1]+(b[1]-a[1])*t/8]
            for a,b in zip(points,points[1:]+points[:1]) for t in range(8)])
        run=[];covered=set();directions=[]
        for n,frame in enumerate(f.frames):
            pose=f.poses[frame['source_frame']];xy=np.array(pose['location'][:2])
            owners=[key for key,r in rooms.items() if contains(r['polygon'],xy,.2)]
            if owners!=[name]:run=[];covered=set();directions=[];continue
            R=np.array(euler_xyz_matrix(pose['rotation_euler_deg']))
            forward=(-R[:,2])[:2];length=np.linalg.norm(forward)
            if length<.2:continue
            forward/=length;directions.append(forward);run.append(n)
            K=pose.get('K',f.record.get('K'));resolution=f.record.get('resolution')
            if K is None or resolution is None:continue
            half=math.atan(resolution[0]/(2*K[0][0]))
            offsets=boundary-xy;cosines=(offsets@forward)/np.linalg.norm(offsets,axis=1)
            covered.update(np.where(cosines>=math.cos(half))[0].tolist())
            span=max(math.degrees(math.acos(float(np.clip(np.dot(a,forward),-1,1)))) for a in directions)
            if len(run)<8 or len(covered)/len(boundary)<.9 or span<150:continue
            s=make(f,'room_floor_area',n,
                'the total floor-footprint area of the room just surveyed that the camera is currently inside, including floor beneath furniture',
                round(room['area_m2'],2),unit='square meters',evidence=run)
            s['private_geometry']={'room':name,**room,'sweep_start':run[0],'sweep_end':n,
                'boundary_angular_coverage':len(covered)/len(boundary),'heading_span_deg':span,
                'evidence_limit':'angular boundary coverage; no wall-pixel occlusion labels',
                'room_file':str(path),'room_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            yield s;break


def worker(row):
    path=Path(row['input'])/'private_facts.json';raw=path.read_bytes();f=EpisodeFacts(json.loads(raw))
    media=json.loads((path.parent/'private_media.json').read_text())
    unique={};rejections=collections.Counter()
    for s in itertools.chain(memory_candidates(f),spatial_candidates(f),area_candidates(f)):
        key=(s['template'],tuple(s.get('categories',[])),tuple(s['evidence_frames']))
        if key in unique:continue
        if s['template']=='historical_relative_direction':
            # The existing router must recover one unambiguous relation.
            try:validate_memory(f,s)
            except (AssertionError,ValueError,KeyError) as e:
                rejections['historical_direction_replay']+=1;continue
        try:
            for index in set(s['evidence_frames']+[s['question_index']]):
                m=media[f.frames[index]['frame']]
                if not Path(m['path']).is_file():raise ValueError('missing_media')
            assert len(s['forms'])==8 and len(set(s['forms']))==8
        except (AssertionError,ValueError,KeyError):
            rejections['media_or_forms']+=1;continue
        s.update(facts_path=str(path),facts_sha256=hashlib.sha256(raw).hexdigest(),
            parent_destination=row['destination'],source_scene_id=f.record.get('scene_id'),
            physical_scene_id=row.get('physical_scene_id'),split=row['split'],
            timestamp=f.frames[s['question_index']]['timestamp'],
            route='memory' if s['template'] in ('last_seen_timestamp','historical_coappearance_category',
                'historical_relative_direction','first_appearance_order') else 'current',
            validation_status='candidate_geometry_and_boolean_evidence_checked')
        unique[key]=s
    return row['episode'],list(unique.values()),dict(rejections)
