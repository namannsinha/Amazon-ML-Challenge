import csv, os, sys, time, argparse, pickle, re, multiprocessing as mp
from collections import defaultdict, Counter
import joblib, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import preprocess_record
from src.features import compute_pair_features

PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
TEST_S1=os.path.join(PROJECT_ROOT,"dataset","test","test_source1.tsv")
MODEL_PATH=os.path.join(os.path.dirname(__file__),"generated","combined_gradient_boosting_model.joblib")
CACHE_DIR=os.path.join(os.path.dirname(__file__),"generated","test_index_v2")
OUTPUT_DIR=os.path.join(PROJECT_ROOT,"output")
THRESHOLD=.875
NAME_TOP_K=10; ADDRESS_TOP_K=10; BATCH_SIZE=5000

G_RECORDS=G_EXACT=G_NAME=G_ADDRESS=G_MODEL=G_FEATURES=None

def normalize_text(value):
    if value is None:return ""
    value=str(value).casefold(); value=re.sub(r"[^\w\s]"," ",value,flags=re.UNICODE); return re.sub(r"\s+"," ",value).strip()

def core_name(value):
    value=normalize_text(value); value=re.sub(r"\b(incorporated|inc|corporation|corp|limited|ltd|private|pvt|company|co|llc|llp)\b","",value); return re.sub(r"\s+"," ",value).strip()

def ngrams(text):
    text=normalize_text(text)
    if not text:return ()
    text="^"+text+"$"; out=set()
    for n in (3,4):
        if len(text)>=n: out.update(text[i:i+n] for i in range(len(text)-n+1))
    return tuple(out)

def paths():
    return {k:os.path.join(CACHE_DIR,v) for k,v in {"records":"records.pkl","exact":"exact.pkl","name":"name_index.pkl","address":"address_index.pkl"}.items()}

def load_index():
    ps=paths()
    with open(ps["records"],"rb") as f: records=pickle.load(f)
    with open(ps["exact"],"rb") as f: exact=pickle.load(f)
    with open(ps["name"],"rb") as f: name=pickle.load(f)
    with open(ps["address"],"rb") as f: address=pickle.load(f)
    return records,exact,name,address

def retrieve(postings,query,top_k):
    grams=ngrams(query)
    if not grams:return ()
    overlap=Counter()
    for g in grams:
        for rid in postings.get(g,()): overlap[rid]+=1
    return tuple(rid for rid,_ in overlap.most_common(top_k))

def candidates_for_row(row):
    exact=G_EXACT; name_index=G_NAME["postings"]; address_index=G_ADDRESS["postings"]
    name=normalize_text(row.get("business_name","")); address=normalize_text(row.get("business_address","")); core=core_name(name); out=set()
    if name: out.update(exact["name"].get(name,()))
    if core: out.update(exact["core"].get(core,()))
    if address: out.update(exact["address"].get(address,()))
    out.update(retrieve(name_index,name,NAME_TOP_K)); out.update(retrieve(address_index,address,ADDRESS_TOP_K))
    return out

def make_candidate_record(rid):
    r=G_RECORDS
    return preprocess_record(r["entity_ids"][rid],r["names"][rid],r["addresses"][rid],r["countries"][rid])

def process_batch(batch):
    candidate_map={}; feature_rows=[]; pair_keys=[]
    for row in batch:
        sid=row["entity_id"]; cands=candidates_for_row(row); candidate_map[sid]=cands
        s1p=preprocess_record(sid,row.get("business_name",""),row.get("business_address",""),row.get("country",""))
        for rid in cands:
            feat=compute_pair_features(s1p,make_candidate_record(rid)); feature_rows.append([feat[c] for c in G_FEATURES]); pair_keys.append((sid,rid))
    probs=G_MODEL.predict_proba(pd.DataFrame(feature_rows,columns=G_FEATURES))[:,1] if feature_rows else []
    predictions=defaultdict(list)
    for (sid,rid),p in zip(pair_keys,probs):
        if p>=THRESHOLD: predictions[sid].append(rid)
    return candidate_map,predictions

def init_worker(records,exact,name,address,model,features):
    global G_RECORDS,G_EXACT,G_NAME,G_ADDRESS,G_MODEL,G_FEATURES
    G_RECORDS=records; G_EXACT=exact; G_NAME=name; G_ADDRESS=address; G_MODEL=model; G_FEATURES=features

def read_s1(sample_s1=None):
    rows=[]
    with open(TEST_S1,"r",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            rows.append(row)
            if sample_s1 and len(rows)>=sample_s1: break
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--sample-s1",type=int,default=None); p.add_argument("--batch-size",type=int,default=BATCH_SIZE); p.add_argument("--workers",type=int,default=4); args=p.parse_args()
    start=time.time(); print("="*70); print("V4 PARALLEL TEST SUBMISSION GENERATOR"); print("="*70); print(f"Workers: {args.workers}"); print(f"Batch size: {args.batch_size}"); print(f"Threshold: {THRESHOLD}")
    records,exact,name,address=load_index(); print(f"Loaded index for {len(records['entity_ids']):,} records",flush=True)
    bundle=joblib.load(MODEL_PATH); model=bundle["model"]; features=bundle["feature_columns"]
    s1_rows=read_s1(args.sample_s1); print(f"S1 records: {len(s1_rows):,}",flush=True)
    batches=[s1_rows[i:i+args.batch_size] for i in range(0,len(s1_rows),args.batch_size)]
    os.makedirs(OUTPUT_DIR,exist_ok=True); cp=os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"); mpth=os.path.join(OUTPUT_DIR,"matching_results.tsv")
    total_candidates=total_matches=completed=0
    ctx=mp.get_context("fork"); workers=max(1,min(args.workers,len(batches)))
    with open(cp,"w",encoding="utf-8",newline="") as cf, open(mpth,"w",encoding="utf-8",newline="") as mf:
        cw=csv.writer(cf,delimiter="\t",lineterminator="\n"); mw=csv.writer(mf,delimiter="\t",lineterminator="\n"); cw.writerow(["source1_entity_id","candidate_entity_ids"]); mw.writerow(["source1_entity_id","matched_entity_ids"])
        with ctx.Pool(processes=workers,initializer=init_worker,initargs=(records,exact,name,address,model,features)) as pool:
            for cmap,preds in pool.imap(process_batch,batches):
                batch=s1_rows[completed*args.batch_size:(completed+1)*args.batch_size]
                for row in batch:
                    sid=row["entity_id"]; rids=cmap.get(sid,()); cids=sorted(records["entity_ids"][r] for r in rids); mids=sorted(set(records["entity_ids"][r] for r in preds.get(sid,())))
                    cw.writerow([sid,",".join(cids)]); mw.writerow([sid,",".join(mids)]); total_candidates+=len(cids); total_matches+=len(mids)
                completed+=1
                print(f"Completed {min(completed*args.batch_size,len(s1_rows)):,}/{len(s1_rows):,} S1 | candidates={total_candidates:,} | matches={total_matches:,} | elapsed={(time.time()-start)/60:.1f} min",flush=True)
    print("\n"+"="*70); print("COMPLETE"); print("="*70); print(f"S1: {len(s1_rows):,}"); print(f"Candidates: {total_candidates:,}"); print(f"Avg candidates/S1: {total_candidates/max(1,len(s1_rows)):.2f}"); print(f"Matches: {total_matches:,}"); print(f"Runtime: {(time.time()-start)/60:.2f} minutes"); print(cp); print(mpth)

if __name__=="__main__": main()
