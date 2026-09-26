import csv, os, sys, time, argparse, pickle, re
from collections import defaultdict, Counter
import joblib, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import preprocess_record
from src.features import compute_pair_features
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
TEST_S1=os.path.join(PROJECT_ROOT,"dataset","test","test_source1.tsv")
TEST_S2=os.path.join(PROJECT_ROOT,"dataset","test","test_source2.tsv")
TEST_S3=os.path.join(PROJECT_ROOT,"dataset","test","test_source3.tsv")
MODEL_PATH=os.path.join(os.path.dirname(__file__),"generated","combined_gradient_boosting_model.joblib")
CACHE_DIR=os.path.join(os.path.dirname(__file__),"generated","test_index_v2")
OUTPUT_DIR=os.path.join(PROJECT_ROOT,"output")
THRESHOLD=.875; NGRAM_SIZES=(3,4); NAME_MAX_POSTING=5000; ADDRESS_MAX_POSTING=2000; NAME_TOP_K=10; ADDRESS_TOP_K=10; BATCH_SIZE=5000; CACHE_VERSION="v2-int-index-1"
def normalize_text(value):
    if value is None:return ""
    value=str(value).casefold(); value=re.sub(r"[^\w\s]"," ",value,flags=re.UNICODE); return re.sub(r"\s+"," ",value).strip()
def core_name(value):
    value=normalize_text(value); value=re.sub(r"\b(incorporated|inc|corporation|corp|limited|ltd|private|pvt|company|co|llc|llp)\b","",value); return re.sub(r"\s+"," ",value).strip()
def ngrams(text):
    text=normalize_text(text)
    if not text:return ()
    text="^"+text+"$"; out=set()
    for n in NGRAM_SIZES:
        if len(text)>=n: out.update(text[i:i+n] for i in range(len(text)-n+1))
    return tuple(out)
def paths():
    return {k:os.path.join(CACHE_DIR,v) for k,v in {"meta":"meta.pkl","records":"records.pkl","exact":"exact.pkl","name":"name_index.pkl","address":"address_index.pkl"}.items()}
def cache_exists(): return all(os.path.exists(v) for v in paths().values())
def save_pickle(obj,path):
    tmp=path+".tmp"
    with open(tmp,"wb") as f: pickle.dump(obj,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,path)
def build_index():
    print("="*70); print("BUILDING V2 TEST INDEX"); print("="*70)
    os.makedirs(CACHE_DIR,exist_ok=True)
    for fn in os.listdir(CACHE_DIR): os.remove(os.path.join(CACHE_DIR,fn))
    entity_ids=[]; names=[]; addresses=[]; countries=[]
    exact_name=defaultdict(set); exact_core=defaultdict(set); exact_address=defaultdict(set)
    name_postings=defaultdict(list); address_postings=defaultdict(list)
    total=0; start=time.time()
    for path in (TEST_S2,TEST_S3):
        print("Reading",path,flush=True)
        with open(path,"r",encoding="utf-8",newline="") as f:
            for row in csv.DictReader(f,delimiter="\t"):
                eid=row["entity_id"]; name=normalize_text(row.get("business_name","")); address=normalize_text(row.get("business_address","")); country=normalize_text(row.get("country","")); core=core_name(name); rid=total
                entity_ids.append(eid); names.append(name); addresses.append(address); countries.append(country)
                if name: exact_name[name].add(rid)
                if core: exact_core[core].add(rid)
                if address: exact_address[address].add(rid)
                for g in ngrams(name): name_postings[g].append(rid)
                for g in ngrams(address): address_postings[g].append(rid)
                total+=1
                if total%500000==0: print(f"  {total:,} records | {(time.time()-start)/60:.1f} min",flush=True)
    name_postings={g:ids for g,ids in name_postings.items() if len(ids)<=NAME_MAX_POSTING}
    address_postings={g:ids for g,ids in address_postings.items() if len(ids)<=ADDRESS_MAX_POSTING}
    ps=paths(); print("Saving compact indexes...",flush=True)
    save_pickle({"version":CACHE_VERSION,"entity_ids":entity_ids,"names":names,"addresses":addresses,"countries":countries},ps["records"])
    save_pickle({"version":CACHE_VERSION,"name":dict(exact_name),"core":dict(exact_core),"address":dict(exact_address)},ps["exact"])
    save_pickle({"version":CACHE_VERSION,"postings":name_postings},ps["name"])
    save_pickle({"version":CACHE_VERSION,"postings":address_postings},ps["address"])
    save_pickle({"version":CACHE_VERSION,"count":total,"name_grams":len(name_postings),"address_grams":len(address_postings)},ps["meta"])
    print(f"Loaded {total:,} records"); print(f"Name grams: {len(name_postings):,}"); print(f"Address grams: {len(address_postings):,}"); print(f"Build time: {(time.time()-start)/60:.1f} min"); print("Index build complete.",flush=True)
def load_index():
    ps=paths()
    with open(ps["records"],"rb") as f: records=pickle.load(f)
    with open(ps["exact"],"rb") as f: exact=pickle.load(f)
    with open(ps["name"],"rb") as f: name=pickle.load(f)
    with open(ps["address"],"rb") as f: address=pickle.load(f)
    return records,exact,name,address
def retrieve(postings,query,top_k):
    grams=ngrams(query)
    if not grams:return []
    overlap=Counter()
    for g in grams:
        for rid in postings.get(g,()): overlap[rid]+=1
    return [rid for rid,_ in overlap.most_common(top_k)]
def candidates_for_row(row,exact,name_index,address_index):
    name=normalize_text(row.get("business_name","")); address=normalize_text(row.get("business_address","")); core=core_name(name); out=set()
    if name: out.update(exact["name"].get(name,()))
    if core: out.update(exact["core"].get(core,()))
    if address: out.update(exact["address"].get(address,()))
    out.update(retrieve(name_index["postings"],name,NAME_TOP_K)); out.update(retrieve(address_index["postings"],address,ADDRESS_TOP_K)); return out
def make_candidate_record(records,rid):
    return preprocess_record(records["entity_ids"][rid],records["names"][rid],records["addresses"][rid],records["countries"][rid])

def process_batch(batch, records, exact, name_index, address_index, model, feature_columns):
    t0 = time.perf_counter()

    candidate_map = {}
    feature_rows = []
    pair_keys = []

    candidate_time = 0.0
    feature_time = 0.0

    for row in batch:
        sid = row["entity_id"]

        # Candidate generation timing
        t = time.perf_counter()

        cands = candidates_for_row(
            row,
            exact,
            name_index,
            address_index
        )

        candidate_time += time.perf_counter() - t

        candidate_map[sid] = cands

        s1p = preprocess_record(
            sid,
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", "")
        )

        # Feature generation timing
        t = time.perf_counter()

        for rid in cands:
            candidate = make_candidate_record(records, rid)

            feat = compute_pair_features(
                s1p,
                candidate
            )

            feature_rows.append(
                [feat[c] for c in feature_columns]
            )

            pair_keys.append((sid, rid))

        feature_time += time.perf_counter() - t

    # Model timing
    t = time.perf_counter()

    if feature_rows:
        X = pd.DataFrame(
            feature_rows,
            columns=feature_columns
        )

        probs = model.predict_proba(X)[:, 1]
    else:
        probs = []

    model_time = time.perf_counter() - t

    predictions = defaultdict(list)

    for (sid, rid), p in zip(pair_keys, probs):
        if p >= THRESHOLD:
            predictions[sid].append(rid)

    total = time.perf_counter() - t0

    print(
        f"[PROFILE] "
        f"candidates={candidate_time:.2f}s | "
        f"features={feature_time:.2f}s | "
        f"model={model_time:.2f}s | "
        f"total={total:.2f}s",
        flush=True
    )

    return candidate_map, predictions

def read_s1(sample_s1=None):
    rows=[]
    with open(TEST_S1,"r",encoding="utf-8",newline="") as f:
        for row in csv.DictReader(f,delimiter="\t"):
            rows.append(row)
            if sample_s1 and len(rows)>=sample_s1: break
    return rows
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--sample-s1",type=int,default=None); parser.add_argument("--batch-size",type=int,default=BATCH_SIZE); parser.add_argument("--rebuild-index",action="store_true"); args=parser.parse_args()
    start=time.time(); print("="*70); print("V2 TEST SUBMISSION GENERATOR"); print("="*70); print(f"Batch size: {args.batch_size}"); print(f"Threshold: {THRESHOLD}")
    if args.rebuild_index or not cache_exists(): build_index()
    else: print("Using existing V2 test index",flush=True)
    records,exact,name_index,address_index=load_index(); print(f"Loaded index for {len(records['entity_ids']):,} records",flush=True)
    bundle=joblib.load(MODEL_PATH); model=bundle["model"]; feature_columns=bundle["feature_columns"]
    s1_rows=read_s1(args.sample_s1); print(f"S1 records: {len(s1_rows):,}",flush=True)
    os.makedirs(OUTPUT_DIR,exist_ok=True); cp=os.path.join(OUTPUT_DIR,"candidate_pairs.tsv"); mp=os.path.join(OUTPUT_DIR,"matching_results.tsv")
    total_candidates=total_matches=processed=0
    with open(cp,"w",encoding="utf-8",newline="") as cf,open(mp,"w",encoding="utf-8",newline="") as mf:
        cw=csv.writer(cf,delimiter="\t",lineterminator="\n"); mw=csv.writer(mf,delimiter="\t",lineterminator="\n"); cw.writerow(["source1_entity_id","candidate_entity_ids"]); mw.writerow(["source1_entity_id","matched_entity_ids"])
        for i in range(0,len(s1_rows),args.batch_size):
            batch=s1_rows[i:i+args.batch_size]; cmap,preds=process_batch(batch,records,exact,name_index,address_index,model,feature_columns)
            for row in batch:
                sid=row["entity_id"]; rids=cmap.get(sid,()); cids=sorted(records["entity_ids"][r] for r in rids); mids=sorted(set(records["entity_ids"][r] for r in preds.get(sid,())))
                cw.writerow([sid,",".join(cids)]); mw.writerow([sid,",".join(mids)]); total_candidates+=len(cids); total_matches+=len(mids)
            processed+=len(batch); print(f"Completed {processed:,}/{len(s1_rows):,} S1 | candidates={total_candidates:,} | matches={total_matches:,} | elapsed={(time.time()-start)/60:.1f} min",flush=True)
    print("\n"+"="*70); print("COMPLETE"); print("="*70); print(f"S1: {len(s1_rows):,}"); print(f"Candidates: {total_candidates:,}"); print(f"Avg candidates/S1: {total_candidates/max(1,len(s1_rows)):.2f}"); print(f"Matches: {total_matches:,}"); print(f"Runtime: {(time.time()-start)/60:.2f} minutes"); print(cp); print(mp)
if __name__=="__main__": main()
