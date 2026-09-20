"""Independent source-to-export reconciliation. Stdlib only; never edits sources.

python scripts/audit_dataset_merge.py --output output/merge_audit_2026-09-19
Detailed differences can contain research IDs: keep output private.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

GRADE = dict(zip(['IVD label', 'Modic', 'UP endplate', 'LOW endplate', 'Spondylolisthesis', 'Disc herniation', 'Disc narrowing', 'Disc bulging', 'Pfirrman grade', 'month'], ['ivd_label', 'modic', 'up_endplate', 'low_endplate', 'spondylolisthesis', 'disc_herniation', 'disc_narrowing', 'disc_bulging', 'pfirrmann_grade', 'month']))
LOC = {**dict(zip(['i', 'j', 'k'], ['voxel_i', 'voxel_j', 'voxel_k'])), **{x:x for x in ['x_lps','y_lps','z_lps','volume','spacing_i','spacing_j','spacing_k']}, 'source':'loc_source','qc_status':'loc_qc_status'}


def run(root, out):
    used = set()
    def readcsv(rel):
        path=root/rel; used.add(path)
        with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
    def readjson(path):
        used.add(path); return json.loads(path.read_text(encoding='utf-8-sig'))
    def readjsonl(path):
        used.add(path); return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    def txt(x):
        if x is None: return ''
        if isinstance(x,list): return '\n'.join(txt(v) for v in x if txt(v))
        return str(x).strip()
    def same(a,b):
        if txt(a)==txt(b): return True
        if not txt(a) or not txt(b): return False
        try: return math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-9)
        except (TypeError, ValueError): return False
    differences=[]; checks=collections.Counter(); totals=collections.Counter(); duplicates={}
    def compare(kind,key,field,a,b,numeric=False):
        totals[kind]+=1
        if not (same(a,b) if numeric else txt(a)==txt(b)):
            checks[kind]+=1; differences.append(dict(check=kind,key=str(key),field=field,source=a,export=b))
    def index(rows,keys,name):
        result={}; duplicates[name]=0
        for r in rows:
            key=tuple(str(r[k]) for k in keys)
            duplicates[name]+=key in result; result[key]=r
        return result
    master=readcsv('dataset/dataset_master.csv'); m=index(master,['patient_id','level'],'master')
    cohort={k[0] for k in m}; patient={r['patient_id']:r for r in master}
    grading=readcsv('dataset_local/grading/grading_all.csv'); g=index(grading,['Patient ID','level'],'grading_all')
    coverage={'grading_only_keys':len(g.keys()-m.keys()), 'master_only_keys':len(m.keys()-g.keys())}
    for key in m.keys() & g.keys():
        for a,b in GRADE.items(): compare('grading_to_master',key,b,g[key][a],m[key][b],True)
    months=[]
    for month in [7,8,9]:
        rs=readcsv(f'dataset_local/grading/grading_month{month:02d}.csv')
        for r in rs: r['month']=str(month)
        months+=rs
    monthly=index(months,['Patient ID','level'],'monthly')
    coverage['monthly_vs_all_key_difference']=len(monthly.keys() ^ g.keys())
    for key in monthly.keys() & g.keys():
        for col in GRADE: compare('monthly_to_all',key,col,monthly[key][col],g[key][col],True)
    frozen=readcsv('dataset_local/grading/cohort_frozen.csv')
    frozen_id='patient_id' if 'patient_id' in frozen[0] else 'Patient ID'
    coverage['frozen_vs_master_patient_difference']=len({r[frozen_id] for r in frozen} ^ cohort)
    frozen_index=index(frozen,[frozen_id,'level'],'cohort_frozen')
    coverage['frozen_vs_master_key_difference']=len(frozen_index.keys()^m.keys())
    for key in frozen_index.keys() & m.keys():
        for a,b in {**{v:v for v in GRADE.values() if v not in ['month','pfirrmann_grade']},'pfirrmann':'pfirrmann_grade'}.items():
            compare('frozen_to_master',key,b,frozen_index[key][a],m[key][b],True)
    frozen_flags={c:dict(collections.Counter(r[c] for r in frozen)) for c in ['level_mapping_verified','modic_mixed_type']}
    loc=readcsv('dataset_local/localize/disc_localization.csv'); local=index([r for r in loc if r['in_cohort']=='1'],['patient_id','level'],'localization')
    coverage.update(localization_outside_cohort_rows=sum(r['in_cohort']!='1' for r in loc), localization_missing_keys=len(m.keys()-local.keys()),localization_unused_in_cohort_keys=len(local.keys()-m.keys()))
    for key in m.keys() & local.keys():
        for a,b in LOC.items(): compare('localization_to_master',key,b,local[key][a],m[key][b],a not in ['volume','source','qc_status'])
    assignment=index(readcsv('dataset_local/folds/fold_assignment.csv'),['patient_id'],'fold_assignment')
    coverage['assignment_vs_master_patient_difference']=len({k[0] for k in assignment} ^ cohort)
    coverage['assignment_extra_match_report_extras']=({k[0] for k in assignment}-cohort)
    for fold in range(1,6):
        fold_rows=[]
        for split in ['train','val','test']:
            rows=readcsv(f'dataset_local/folds/fold{fold}/{split}.csv'); fold_rows+=rows
            for r in rows:
                key=(r['Patient ID'],r['level'])
                if key not in m: coverage['fold_orphan_rows']=coverage.get('fold_orphan_rows',0)+1; continue
                compare('fold_split_to_master',key,f'fold{fold}',split,m[key][f'fold{fold}_split'])
                compare('fold_split_to_assignment',key,f'fold{fold}',split,assignment.get((key[0],),{}).get(f'fold{fold}'))
                for a,b in GRADE.items():
                    if a in r: compare('fold_grading_to_master',key,b,r[a],m[key][b],True)
        fi=index(fold_rows,['Patient ID','level'],f'fold{fold}')
        coverage[f'fold{fold}_key_difference']=len(fi.keys() ^ m.keys())
    metadata=[readjson(f) for f in sorted((root/'dataset_local/reports_json/metadata').glob('*.json'))]
    reports=[readjson(f) for f in sorted((root/'dataset_local/reports_json/report').glob('*.json'))]
    meta=index(metadata,['patient_id'],'metadata'); report=index(reports,['patient_id'],'report')
    for name,ix in [('metadata',meta),('report',report)]:
        ids={k[0] for k in ix}; coverage[name]={'source_patients':len(ids),'matched':len(ids&cohort),'outside_cohort':len(ids-cohort),'cohort_without_source':len(cohort-ids)}
    coverage['assignment_extra_match_report_extras']=coverage['assignment_extra_match_report_extras']==({k[0] for k in report}-cohort)
    reportflags=collections.Counter()
    for pid in cohort:
        me=meta.get((pid,),{}); re=report.get((pid,),{}); row=patient[pid]
        for c in ['sub_id','sex','birth_year','age_at_scan','study_date']:
            compare('metadata_to_master',pid,c,me.get(c),row[c],c in ['birth_year','age_at_scan','study_date'])
        if re:
            compare('report_metadata_subid',pid,'sub_id',re.get('sub_id'),me.get('sub_id'))
            for flag in ['has_conclusion','needs_review','source_type']:
                reportflags[f'{flag}={re.get(flag)}']+=1
        for new,old in [('technique','ky_thuat'),('findings','mo_ta'),('impression','ket_luan')]:
            compare('vi_report_to_master',pid,new,re.get(new,re.get(old)),row[f'report_vi_{new}'])
        for new,old in [('technique','kythuat'),('findings','mota'),('impression','ketluan')]:
            compare('master_alias',pid,new,row[f'report_vi_{new}'],row[f'report_vi_{old}'])
    submap={str(r['sub_id']):str(r['patient_id']) for r in metadata if r.get('sub_id')}
    duplicates['metadata_sub_id']=len([r for r in metadata if r.get('sub_id')])-len(submap)
    texts=[]
    for file in sorted((root/'dataset_local/reports_text').glob('*.csv')):
        for r in readcsv(str(file.relative_to(root))): texts.append({**r,'_file':file.name})
    ti=index(texts,['sub_id'],'english_sub_id')
    enmap={submap[sub]:r for (sub,),r in ti.items() if sub in submap}
    coverage['english']={'rows':len(texts),'unmapped_sub_id':sum(r['sub_id'] not in submap for r in texts),'matched_cohort':len(set(enmap)&cohort),'outside_cohort':len(set(enmap)-cohort)}
    for pid in cohort:
        r=enmap.get(pid,{})
        compare('english_to_master',pid,'report_en',r.get("Clinician's Notes"),patient[pid]['report_en'])
        compare('english_split_to_master',pid,'split',r.get('split'),patient[pid]['reports_text_split'])
    refs=[]
    for f in sorted((root/'dataset_local/reports_text_v1_reference').glob('*.csv')): refs+=readcsv(str(f.relative_to(root)))
    ri=index(refs,['sub_id'],'reference_sub_id')
    coverage['english_reference']={'rows':len(refs),'key_difference':len(ti.keys()^ri.keys()),'text_difference':sum(txt(ti[k]["Clinician's Notes"])!=txt(ri[k]["Clinician's Notes"]) for k in ti.keys()&ri.keys())}
    pts=readjsonl(root/'dataset/dataset_patients.jsonl'); ji=index(pts,['patient_id'],'patient_jsonl')
    coverage['json_vs_master_patient_difference']=len({k[0] for k in ji}^cohort)
    for p in pts:
        pid=str(p['patient_id']); row=patient[pid]
        compare('master_to_json',pid,'sub_id',row['sub_id'],p['sub_id'])
        for c in ['sex','birth_year','age_at_scan','study_date']: compare('master_to_json',pid,c,row[c],p['demographics'][c],c!='sex')
        for i in range(1,6): compare('master_to_json',pid,f'fold{i}',row[f'fold{i}_split'],p['folds'][f'fold{i}'])
        li=index(p['levels'],['level'],f'json_levels_{pid}')
        compare('json_structure',pid,'number_levels',len(p['levels']),5,True)
        for l in p['levels']:
            key=(pid,l['level']); rowlevel=m[key]
            compare('master_to_json',key,'ivd_label',rowlevel['ivd_label'],l['ivd_label'],True)
            for c,v in l['gradings'].items(): compare('master_to_json_grading',key,c,rowlevel[c],v,True)
            for c,v in l['coordinates'].items(): compare('master_to_json_coordinates',key,c,rowlevel[c],v,True)
        for new,old in [('technique','ky_thuat'),('findings','mo_ta'),('impression','ket_luan')]:
            compare('master_to_json_vi',pid,new,row[f'report_vi_{new}'],p['reports']['vi'].get(new))
            if old in p['reports']['vi']:
                compare('json_alias',pid,new,p['reports']['vi'].get(new),p['reports']['vi'][old])
        compare('master_to_json_en',pid,'report_en',row['report_en'],p['reports']['en']['clinicians_notes'])
        compare('master_to_json_en',pid,'split',row['reports_text_split'],p['reports']['en']['split'])
    axial=readcsv('dataset_local/localize/axial_level_assignment.csv')
    summary={'counts':{'master_rows':len(master),'patients':len(cohort),'grading_rows':len(grading),'monthly_rows':len(months),'localization_rows':len(loc),'axial_rows':len(axial),'axial_patients':len({r['patient_id'] for r in axial}),'axial_cohort_patients':len({r['patient_id'] for r in axial}&cohort)},'coverage':coverage,'checks':{k:{'compared':v,'differences':checks[k]} for k,v in totals.items()},'duplicates':{k:v for k,v in duplicates.items() if v},'report_flags_in_cohort':dict(reportflags),'frozen_flags':frozen_flags,
        'not_exported':{'frozen_to_master':['cohort','level_mapping_verified','modic_positive','modic_type_ii','modic_mixed_type','pfirrmann_merged','source_file'], 'localization_to_master':[c for c in loc[0] if c not in ['patient_id','level'] and c not in LOC], 'report_to_master':[c for c in reports[0] if c not in ['patient_id','sub_id','technique','findings','impression','ky_thuat','mo_ta','ket_luan']], 'metadata_to_master':[c for c in metadata[0] if c not in ['patient_id','sub_id','sex','age_at_scan','birth_year','study_date']], 'english_to_master':['case_id','image_path'], 'master_to_patient_json':['month','volume','spacing_i','spacing_j','spacing_k','loc_source','loc_qc_status'], 'axial_columns':list(axial[0])},
        'source_hashes':{str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(used)}}
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'differences.private.json').write_text(json.dumps(differences,ensure_ascii=False,indent=2),encoding='utf-8')
    # Minimal notebook companion embeds the exact executed checks for inspection.
    code=(root/'scripts/audit_dataset_merge.py').read_text(encoding='utf-8').rsplit('\nif __name__',1)[0]
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'source':['# Audit gộp dataset\nĐối chiếu từng trường với nguồn, không dùng regex y khoa. Chạy tại repo root; chỉ ghi vào output.']},
        {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':[code+'\nrun(Path.cwd(), Path("output/merge_audit_notebook"))\n']}]}
    (out/'merge_audit.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
    receipt={'schemaVersion':1,'items':[{'id':'merge-reconciliation','title':'Đối chiếu dữ liệu nguồn với bản gộp','queries':[{'id':'source-export','source':{'label':'dataset_local → dataset; kiểm tra bằng audit_dataset_merge.py','caveats':['So sánh văn bản sau strip; số dùng sai số tuyệt đối 1e-9. Không đánh giá tính đúng lâm sàng.','Bản gộp bỏ một số trường truy vết, cờ nguồn và bảng axial; xem báo cáo chi tiết.']},'columns':['check','compared','differences'],'rows':[{'check':k,**v} for k,v in summary['checks'].items()],'preview':{'kind':'aggregate','note':'Tổng hợp các phép đối chiếu theo khóa bệnh nhân và tầng.'},'methods':[{'language':'python','code':code}]}]}]}
    (out/'sources.json').write_text(json.dumps(receipt,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--output',type=Path,default=Path('output/merge_audit_2026-09-19')); args=ap.parse_args(); run(args.root.resolve(),args.output.resolve())
