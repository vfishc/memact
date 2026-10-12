"""Additive data-only proposals; no production templates or artifacts mutated."""
import collections,hashlib,itertools,json,pathlib,random,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent),str(ROOT.parent.parent)]
from routed_facts import EpisodeFacts
from qa_eight_contract import CHOICE_FORMS,NUMERIC_FORMS
from qa_semantic_contract import VISUAL_CATEGORY_ALLOWLIST,categories_conflict,clean_category_options
from memory_runtime import MemoryManager

def make(f,template,index,quantity,answer,options=None,unit=None,evidence=(),categories=()):
 seed=hashlib.sha256(f'{f.record["episode_id"]}|{template}|{index}|{quantity}'.encode()).hexdigest()
 options=list(options) if options else None
 if options:
  assert 4<=len(options)<=5 and len(set(options))==len(options) and answer in options
  random.Random(seed).shuffle(options);mapped={chr(65+i):v for i,v in enumerate(options)};answer=next(k for k,v in mapped.items() if v==answer)
 else:mapped=None
 forms=[p.format(quantity=quantity,unit=f'Answer in {unit}.') for p in (CHOICE_FORMS if options else NUMERIC_FORMS)]
 chosen=int(seed[:8],16)%8
 return {'template':template,'episode':f.record['episode_id'],'source':f.source,'question_index':index,'public':{'mode':'[QA]','question':forms[chosen],'answer_type':'choice' if options else 'numeric','options':mapped,'unit':unit},'answer':answer,'forms':forms,'form_index':chosen,'evidence_frames':list(evidence),'categories':list(categories),'validation_status':'proposal_not_production','id':seed[:20]}

def categories_options(answer,excluded):
 candidates=[answer]+[c for c in sorted(VISUAL_CATEGORY_ALLOWLIST) if c not in excluded]
 _,options,_=clean_category_options(answer,candidates)
 return options[:4] if len(options)>=4 else None

def memory_candidates(f):
 categories=[set(f.counts(i)) for i in range(len(f.frames))]
 for n in range(10,len(f.frames),5):
  # Written memory covers [0,n); require latest sighting before pending5s window.
  last={c:max(i for i in range(n) if c in categories[i]) for c in set.union(*categories[:n])}
  for cat,i in sorted(last.items()):
   if cat in categories[n] or i>n-5:continue
   yield make(f,'last_seen_timestamp',n,f'the last timestamp at which the {cat} category was observed before this question',f.frames[i]['timestamp'],unit='seconds',evidence=[i],categories=[cat])
   # State time explicitly: do not hide GT-selected query bounds from the question.
   for other in sorted(categories[i]-{cat}):
    if categories_conflict(cat,other):continue
    options=categories_options(other,categories[i]|{cat})
    if options:
     yield make(f,'historical_coappearance_category',n,f'the listed category also visible alongside the {cat} at timestamp {f.frames[i]["timestamp"]:g} seconds',other,options,evidence=[i],categories=[cat]);break
   counts=f.counts(i);regions=f.regions(i)
   targets=[o for o in regions if f.categories[o]==cat and counts[cat]==1]
   if not targets:continue
   target=targets[0];_,tx,ty=regions[target]
   for other in sorted(regions):
    name=f.categories[other];_,x,y=regions[other]
    if counts[name]!=1 or categories_conflict(cat,name):continue
    if y==ty and x!=tx:direction='left' if x<tx else 'right'
    elif x==tx and y!=ty:direction='above' if y<ty else 'below'
    else:continue
    yield make(f,'historical_relative_direction',n,f'the image-plane position of the {name} relative to the {cat} in the observation at timestamp {f.frames[i]["timestamp"]:g} seconds',direction,['left','right','above','below'],evidence=[i],categories=[cat,name]);break
  first=f.first_categories(n-1)
  for triple in itertools.combinations(first,3):
   ids=[i for i,c in triple];cats=[c for i,c in triple]
   if ids[-1]>n-5 or any(f.frames[b]['timestamp']-f.frames[a]['timestamp']<2 for a,b in zip(ids,ids[1:])):continue
   if any(categories_conflict(a,b) for a,b in itertools.combinations(cats,2)):continue
   answer=' → '.join(cats);others=[' → '.join(x) for x in itertools.permutations(cats) if list(x)!=cats]
   yield make(f,'first_appearance_order',n,'the first-observation order of the categories '+', '.join(sorted(cats))+' since this episode began',answer,[answer]+others[:3],evidence=ids,categories=cats)
   break

def stable_events(sequence,start,options,appearing=True):
 found={}
 for i in range(start+2,len(sequence)):
  for c in options:
   if c not in found and all((c in sequence[j])==appearing for j in [i-1,i]):found[c]=i
 return found

def timing_candidates(f):
 sequence=[set(f.counts(i)) for i in range(len(f.frames))]
 for start in range(2,len(sequence)-8,5):
  for appearing in [True,False]:
   if appearing:
    available=set.union(*sequence[start+1:])-sequence[start]-sequence[start-1]
   else:available=sequence[start]&sequence[start-1]
   for target in sorted(available):
    if appearing:options=categories_options(target,sequence[start]|sequence[start-1])
    else:
     _,options,_=clean_category_options(target,[target]+sorted(available-{target}));options=options[:4]
    if not options or len(options)<4:continue
    found=stable_events(sequence,start,options,appearing)
    if target not in found:continue
    t=min(found.values());winners=[c for c,v in found.items() if v==t]
    if winners!=[target] or t+5>=len(sequence):continue
    action='becomes stably visible' if appearing else 'stops being stably visible'
    sample=make(f,'timing_appearance' if appearing else 'timing_disappearance',start,'the listed category that '+action+' first after this question (confirm with two consecutive observations)',target,options,evidence=[t-1,t],categories=options)
    sample.update(first_answerable=t,deadline=t+5,checks=list(range(start+1,t+6)),teacher=[{'index':i,'type':'wait' if i<t else 'answer',**({'answer':sample['answer']} if i==t else {})} for i in range(start+1,t+1)])
    yield sample;break

def validate_memory(f,s):
 n=s['question_index'];manager=MemoryManager();writes=[]
 for i in range(n):
  manager.observe(f.frames[i]['frame'],f.frames[i]['timestamp'])
  if (i+1)%5==0:
   payload=f.teacher(range(i-4,i+1));manager.write(payload,i-4,i+1);writes.append(payload)
 before=manager.replay().summaries;manager.begin_question()
 with_frames=s['template']=='historical_relative_direction'
 request={'type':'use_memory','categories':s['categories'],'with_frames':with_frames}
 if s['template'] in ['historical_coappearance_category','historical_relative_direction']:
  t=f.frames[s['evidence_frames'][0]]['timestamp'];request['time_range']=[t,t+1]
 pages=[];query=dict(request)
 for _ in range(100):
  result=manager.query(query);pages.append(result)
  if not result['next_cursor']:break
  query={**request,'cursor':result['next_cursor']}
 else:raise ValueError('pagination_exhausted')
 assert manager.summaries==before
 frame_handles={x['frame'] for p in pages for x in p['frames']}
 if with_frames:assert {f.frames[i]['frame'] for i in s['evidence_frames']}<=frame_handles
 assert all(i<n for i in s['evidence_frames']) and len(frame_handles)<=4
 records={r['frame']:r for page in pages for r in page['records']}
 if s['template']=='last_seen_timestamp':
  derived=max(r['timestamp'] for r in records.values() if any(o['category']==s['categories'][0] for o in r['objects']))
 elif s['template']=='first_appearance_order':
  first={c:min(r['timestamp'] for r in records.values() if any(o['category']==c for o in r['objects'])) for c in s['categories']}
  derived=' → '.join(sorted(first,key=first.get))
 elif s['template']=='historical_coappearance_category':
  present={o['category'] for r in records.values() for o in r['objects']}
  matches=[v for v in s['public']['options'].values() if v in present];assert len(matches)==1;derived=matches[0]
 else:
  wanted=set(s['categories']);labels={'left_of_in_image':'left','right_of_in_image':'right','above_in_image':'above','below_in_image':'below'};inverse={'left':'right','right':'left','above':'below','below':'above'}
  relations=[r for page in pages for r in page['relations'] if {r['subject']['category'],r['object']['category']}==wanted]
  values={labels[r['predicate']] if r['subject']['category']==s['categories'][1] else inverse[labels[r['predicate']]] for r in relations};assert len(values)==1;derived=values.pop()
 expected=s['public']['options'][s['answer']] if s['public']['options'] else s['answer'];assert derived==expected,(derived,expected)
 return {'memory_writes':writes,'request':request,'pages':pages,'qa_did_not_write_memory':True}

