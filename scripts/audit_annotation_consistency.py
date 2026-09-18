"""Read-only comparison of independent grading and Vietnamese report annotations.

This is a text/label consistency screen, NOT image adjudication or a validated
clinical NLP system. Unmentioned findings are unknown, never negative.
Patient-level CSVs contain local research identifiers and belong in output/.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

LEVELS = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
FIELDS = ['disc_herniation', 'disc_bulging', 'spondylolisthesis', 'modic', 'disc_narrowing', 'endplate_any']
PATTERNS = {
    'disc_herniation': r'\bthoat vi\b(?!\s+noi\b)',
    'disc_bulging': r'\b(?:phinh|phong)\b',
    'protrusion_word': r'\bloi\b',
    'spondylolisthesis': r'\btruot\b',
    'modic': r'\bmodic\b',
    'endplate_any': r'\b(?:noi xop|noi xap|schmorl)\b',
}
LEVEL_RE = re.compile(r'\bL\s*([1-5])\s*[/\-]\s*([LS]?)\s*([1-5])\b', re.I)


def normalize(text):
    text = text.lower().replace('đ', 'd').replace('–', '-').replace('−', '-')
    return re.sub(r'\s+', ' ', ''.join(c for c in unicodedata.normalize('NFD', text)
                                      if unicodedata.category(c) != 'Mn')).strip()


def levels_in(text):
    levels = []
    for m in LEVEL_RE.finditer(text):
        a, prefix, b = m.groups()
        level = f'L{a}/{prefix.upper() or ("S" if a == "5" and b == "1" else "L")}{b}'
        if level in LEVELS and level not in levels:
            levels.append(level)
    return levels


def sentences(report):
    if isinstance(report, str):
        report = [report]
    for original in report or []:
        # Preserve original full sentence to determine scope of 'remaining discs'.
        prepared = re.sub(r'([LS]\d\s*[/\-]\s*[LS]?\d)\.\s*(?=[LS]\d\s*[/\-])', r'\1, ', original, flags=re.I)
        for part in re.split(r'[.;](?!\d)', prepared):
            if part.strip():
                yield original.strip(), part.strip(), normalize(part)


def extract(report, section):
    events = []
    for original, part, text in sentences(report):
        remaining = 'con lai' in normalize(original)
        for feature, pattern in PATTERNS.items():
            for match in re.finditer(pattern, text):
                if feature == 'protrusion_word' and 'dia dem' not in text:
                    continue  # posterior vertebral wall protrusion is not a disc bulge
                before = text[:match.start()]
                # Negation after the finding (e.g. no root compression) does not
                # negate the finding itself. Modic descriptions often say 'no
                # STIR increase'; that does not negate Modic.
                if feature == 'modic':
                    negative = bool(re.search(r'(?:khong|chua)(?: thay| co)?\s+(?:thoai hoa\s+)?$', before))
                else:
                    neg = re.search(r'\b(?:khong|chua)\s+(?:thay|co|ghi nhan)\b[^.;]{0,65}$', before)
                    if re.search(r'\b(?:khong|chua)\s*$', before):
                        neg = True
                    negative = bool(neg)
                qualified = 'dien hinh' in text
                polarity = 'qualified_negative' if negative and qualified else ('negative' if negative else 'positive')
                # Restrict morphology attribution to the descriptive head,
                # before consequences mentioning canal/root levels.
                head = re.split(r'\b(?:gay|chen ep|de day|tiep xuc|duong kinh)\b', text)[0]
                if feature in ['disc_herniation', 'disc_bulging', 'protrusion_word']:
                    head = re.sub(r'\b(?:khong|chua)\s*$', '', head).strip()
                    head = re.split(r'\b(?:kem phi dai|va day day chang|kem day day chang)\b', head)[0]
                    cut = re.search(r'\brach\b', head[match.end():]) if match.end() < len(head) else None
                    tail = head[match.end():]
                    if cut and levels_in(tail[:cut.start()]):
                        tail = tail[:cut.start()]
                    level_list = levels_in(tail) or levels_in(head[:match.start()])
                    mixed = sum(bool(re.search(PATTERNS[f], head)) for f in
                                ['disc_herniation', 'disc_bulging', 'protrusion_word']) > 1
                    if mixed:
                        level_list = []  # defer ambiguous mixed-clause attribution
                else:
                    level_list = []  # vertebra != disc: do not invent mapping
                scope = 'remaining' if remaining and negative else ('level' if levels_in(text) else 'global')
                if negative and feature in ['disc_herniation','disc_bulging','protrusion_word'] and levels_in(normalize(original)):
                    scope = 'remaining' if remaining else 'level'
                events.append(dict(feature=feature, polarity=polarity, scope=scope,
                                   levels=level_list, section=section, evidence=original,
                                   clause=part, head=head))
        # Narrowing means disc height loss, not canal/foraminal narrowing.
        if ('dia dem' in text or 'dia dem' in normalize(original)) and 'chieu cao' in text:
            if ('than dot' in text or 'tuong truoc' in text) and 'dia dem' not in text:
                continue
            height = text[text.index('chieu cao'):]
            normal = bool(re.match(r'chieu cao(?:\s+va tin hieu)?\s+(?:con\s+)?(?:trong gioi han\s+)?binh thuong', height))
            reduced = bool(re.search(r'giam(?: nhe)?\s+chieu cao', text) or
                           re.match(r'chieu cao(?:\s+va tin hieu(?: dia dem)?)?\s+giam', height))
            if normal or reduced:
                events.append(dict(feature='disc_narrowing', polarity='negative' if normal else 'positive',
                    scope='remaining' if remaining else ('level' if levels_in(text[:text.index('chieu cao')]) else 'global'),
                                   levels=[], section=section, evidence=original, clause=part, head=text))
        if re.search(r'(?:hep\s+khe\s+dia dem|khe\s+dia dem[^.;]*\bhep\b)', text):
            events.append(dict(feature='disc_narrowing', polarity='positive', scope='level', levels=[],
                               section=section, evidence=original, clause=part, head=text))
    return events


def state(events, feature):
    allowed = [feature, 'protrusion_word'] if feature == 'disc_bulging' else [feature]
    found = [e for e in events if e['feature'] in allowed]
    positive = any(e['polarity'] == 'positive' for e in found)
    negative = any(e['polarity'] == 'negative' and e['scope'] == 'global' for e in found)
    if positive and negative:
        return 'internal_conflict'
    if positive:
        return 'positive'
    if negative:
        return 'negative'
    return 'unspecified'


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(root, output):
    source = root / 'dataset/dataset_patients.jsonl'
    patients = [json.loads(line) for line in source.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    with (root/'dataset/dataset_master.csv').open(encoding='utf-8-sig',newline='') as f:
        master = list(csv.DictReader(f))
    missing = {(r['patient_id'],r['level'],feature) for r in master for feature in FIELDS if feature in r and not r[feature].strip()}
    for p in patients:
        for level in p['levels']:
            level['gradings']['endplate_any'] = int(bool(level['gradings'].get('up_endplate') or level['gradings'].get('low_endplate')))
            for feature in FIELDS:
                if (str(p['patient_id']),level['level'],feature) in missing:
                    level['gradings'][feature] = None
    output.mkdir(parents=True, exist_ok=True)
    comparisons, level_comparisons, assertions, patient_rows = [], [], [], []
    all_events = {}
    for patient in patients:
        pid = str(patient['patient_id'])
        vi = patient['reports']['vi']
        findings = vi.get('findings', vi.get('mo_ta', []))
        impression = vi.get('impression', vi.get('ket_luan', []))
        if not findings and not impression:
            continue
        events = extract(findings, 'findings') + extract(impression, 'impression')
        all_events[pid] = events
        grades = {level['level']: level['gradings'] for level in patient['levels']}
        # JSON currently imputes one missing disc_bulging to zero. Restore the
        # actual missing value from CSV before any negative/positive assertion.
        all_events[pid] = events
        for e in events:
            assertions.append({'patient_id':pid, **e, 'levels':'|'.join(e['levels'])})
        for feature in FIELDS:
            vals = [grades[level].get(feature) for level in LEVELS]
            grade_state = 'positive' if any(v is not None and v > 0 for v in vals) else ('negative' if all(v == 0 for v in vals) else 'unknown')
            report_state = state(events, feature)
            mismatch = (grade_state, report_state) in [('negative','positive'),('positive','negative')]
            comparisons.append(dict(patient_id=pid, fold1=patient['folds']['fold1'], feature=feature,
                                    grade_state=grade_state, report_state=report_state, mismatch=mismatch,
                                    grade_positive_levels='|'.join(l for l in LEVELS if grades[l].get(feature,0)),
                                    evidence=' || '.join(dict.fromkeys(e['evidence'] for e in events if e['feature']==feature or (feature=='disc_bulging' and e['feature']=='protrusion_word')))))
        # Morphology comparisons at an explicitly named disc level.
        for feature in ['disc_herniation','disc_bulging','protrusion_word','displacement_any']:
            allowed = ['disc_herniation','disc_bulging','protrusion_word'] if feature=='displacement_any' else (['disc_bulging','protrusion_word'] if feature=='disc_bulging' else [feature])
            for level in LEVELS:
                positives = [e for e in events if e['feature'] in allowed and e['polarity']=='positive' and level in e['levels']]
                if not positives:
                    continue
                val = grades[level].get('disc_bulging' if feature=='protrusion_word' else feature)
                if feature=='displacement_any':
                    pair=[grades[level].get('disc_herniation'),grades[level].get('disc_bulging')]
                    val=1 if any(v is not None and v>0 for v in pair) else (0 if all(v==0 for v in pair) else None)
                level_comparisons.append(dict(patient_id=pid, fold1=patient['folds']['fold1'], level=level,
                                              feature=feature, grade_value=val, mismatch=val==0,
                                              evidence=' || '.join(dict.fromkeys(e['evidence'] for e in positives))))
        # Broad union avoids declaring bulge/herniation terminology interchange
        # to be an absence-of-abnormality conflict.
        positive_any = [e for e in events if e['feature'] in ['disc_herniation','disc_bulging','protrusion_word'] and e['polarity']=='positive']
        grade_values=[grades[l].get(f) for l in LEVELS for f in ['disc_herniation','disc_bulging']]
        grade_any=any(v is not None and v>0 for v in grade_values)
        patient_rows.append(dict(patient_id=pid, fold1=patient['folds']['fold1'],
                                 has_findings=bool(findings),has_impression=bool(impression),
                                 report_displacement=bool(positive_any),grade_displacement=grade_any,
                                 broad_absence_conflict=bool(positive_any) and all(v==0 for v in grade_values)))
    summary={'cohort':len(patients),'with_report':len(patient_rows), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'features':{}}
    for feature in FIELDS:
        rows=[r for r in comparisons if r['feature']==feature]
        summary['features'][feature]={'matrix':dict(collections.Counter(r['grade_state']+'/'+r['report_state'] for r in rows)),
                                       'mismatch_patients':sum(r['mismatch'] for r in rows)}
    summary['patient_any_conflict']=len({r['patient_id'] for r in comparisons if r['mismatch']})
    summary['patient_conflict_without_height']=len({r['patient_id'] for r in comparisons if r['mismatch'] and r['feature']!='disc_narrowing'})
    summary['broad_absence_conflict']=sum(r['broad_absence_conflict'] for r in patient_rows)
    summary['levels']={f:{'explicit_positive_pairs':sum(r['feature']==f for r in level_comparisons),
                          'grade_zero_pairs':sum(r['feature']==f and r['mismatch'] for r in level_comparisons),
                          'patients_with_conflict':len({r['patient_id'] for r in level_comparisons if r['feature']==f and r['mismatch']})}
                       for f in ['disc_herniation','disc_bulging','protrusion_word','displacement_any']}
    summary['by_fold1']={split:{'patients':sum(r['fold1']==split for r in patient_rows),
                                      'patient_conflict':len({r['patient_id'] for r in comparisons if r['mismatch'] and r['fold1']==split})}
                          for split in ['train','val','test']}
    # Independent reading sections are counted separately in sensitivity checks.
    summary['by_section']={section:{feature:sum(
        (r['grade_state'],state([e for e in all_events[r['patient_id']] if e['section']==section],feature))
        in [('negative','positive'),('positive','negative')]
        for r in comparisons if r['feature']==feature)
        for feature in FIELDS} for section in ['findings','impression']}
    summary['pfirrmann_numeric_mentions']=sum(bool(re.search(r'pfirr',normalize(' '.join(
        p['reports']['vi'].get('findings',p['reports']['vi'].get('mo_ta',[]))+
        p['reports']['vi'].get('impression',p['reports']['vi'].get('ket_luan',[])))))) for p in patients)
    # Produce a review queue, never change either doctor's labels.
    review=[]
    for p in patient_rows:
        pid=p['patient_id']
        records=[r for r in comparisons if r['patient_id']==pid]
        level_records=[r for r in level_comparisons if r['patient_id']==pid]
        strong=[]
        if p['broad_absence_conflict']: strong.append('report_displacement_but_all_herniation_and_bulging_zero')
        strong += [r['feature'] for r in records if r['mismatch'] and r['feature'] in ['spondylolisthesis','modic']]
        strong += ['report_internal_'+r['feature'] for r in records if r['report_state']=='internal_conflict']
        needs_level=any(r['feature']=='displacement_any' and r['mismatch'] for r in level_records)
        other=[r['feature'] for r in records if r['mismatch']]
        subtype=any(r['feature'] in ['disc_bulging','disc_herniation'] and r['mismatch'] for r in level_records)
        priority='A_patient_or_internal' if strong else ('B_level_or_definition' if needs_level or other or subtype else 'C_no_flag_not_adjudicated')
        review.append(dict(patient_id=pid,fold1=p['fold1'],priority=priority,
                           reasons='|'.join(strong+other+(['level_displacement_absent'] if needs_level else [])+(['level_subtype_difference'] if subtype else [])),
                           reviewed_by='',image_review_needed='',accepted_grading='',accepted_report='',review_notes=''))
    summary['review_queue']=dict(collections.Counter(r['priority'] for r in review))
    summary['review_queue_by_fold']={s:dict(collections.Counter(r['priority'] for r in review if r['fold1']==s)) for s in ['train','val','test']}
    month={r['patient_id']:r['month'] for r in master}
    summary['by_month']={m:{'paired_patients':sum(month[p['patient_id']]==m for p in patient_rows),
                            'patient_conflicts':len({r['patient_id'] for r in comparisons if r['mismatch'] and month[r['patient_id']]==m})}
                         for m in sorted(set(month.values()))}
    write_csv(output/'review_queue.csv', review)
    write_csv(output/'patient_comparisons.csv', comparisons)
    write_csv(output/'level_comparisons.csv', level_comparisons)
    write_csv(output/'report_assertions.csv', assertions)
    write_csv(output/'patient_summary.csv', patient_rows)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('output/annotation_audit'))
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.output),ensure_ascii=False,indent=2))
