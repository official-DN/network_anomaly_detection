"""
app.py — AI-Based Anomaly Detection in Network Traffic
Run: python app.py
Flask frontend + pure-numpy ML models — zero extra setup needed.
"""

import os, sys, math, time, random, threading, subprocess
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pathlib import Path

# ── Auto-install required packages ────────────────────────────────────────────
def _install(pkg):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", pkg],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

for _pkg in ["flask", "numpy", "scikit-learn"]:
    try:
        __import__(_pkg.replace("-","_"))
    except ImportError:
        print(f"[setup] Installing {_pkg} ...")
        _install(_pkg)

import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, jsonify, flash)

# ── Flask init ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "flask_app" / "templates"),
    static_folder=str(BASE_DIR / "flask_app" / "static"),
)
app.secret_key = "nad-flask-2024"

# ── Global stores ──────────────────────────────────────────────────────────────
EVENTS  = []
ALERTS  = []
MODELS  = {}
SCALERS = {}
METRICS = {}

PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1, "HTTP": 80, "DNS": 53}
ATTACK_TYPES = ["DDoS","Port Scan","Brute Force","Botnet","DoS","Infiltration","Web Attack"]
RISK_BANDS   = [(0,30,"LOW"),(30,60,"MEDIUM"),(60,80,"HIGH"),(80,101,"CRITICAL")]

# ══════════════════════════════════════════════════════════════════════════════
#  ML MODELS  (pure numpy)
# ══════════════════════════════════════════════════════════════════════════════

class RandomForestMini:
    def __init__(self, n_trees=25, seed=42):
        self.n_trees = n_trees
        self.trees   = []
        self.rng     = np.random.default_rng(seed)

    def _gini(self, y):
        if not len(y): return 0.0
        p = np.bincount(y, minlength=2) / len(y)
        return 1.0 - float(np.sum(p**2))

    def _best_split(self, X, y, feats):
        best = (None, None, float("inf"))
        for f in feats:
            vals = np.unique(X[:, f])
            if len(vals) < 2: continue
            for t in ((vals[:-1]+vals[1:])/2)[:8]:
                m = X[:,f] <= t
                g = (m.sum()*self._gini(y[m]) + (~m).sum()*self._gini(y[~m])) / len(y)
                if g < best[2]: best = (f, t, g)
        return best

    def _build(self, X, y, depth=0):
        if depth >= 6 or len(np.unique(y)) == 1 or len(y) < 5:
            v = np.bincount(y, minlength=2); return {"leaf":True,"prob":v/v.sum()}
        nf = max(1, int(np.sqrt(X.shape[1])))
        fs = self.rng.choice(X.shape[1], nf, replace=False)
        f, t, _ = self._best_split(X, y, fs)
        if f is None:
            v = np.bincount(y, minlength=2); return {"leaf":True,"prob":v/v.sum()}
        m = X[:,f] <= t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[m],y[m],depth+1),"r":self._build(X[~m],y[~m],depth+1)}

    def _pred(self, node, x):
        return node["prob"] if node["leaf"] else self._pred(node["l"] if x[node["f"]]<=node["t"] else node["r"], x)

    def fit(self, X, y):
        n = len(X)
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, n)
            self.trees.append(self._build(X[idx], y[idx]))
        return self

    def predict_proba(self, X):
        return np.array([[self._pred(t, x) for t in self.trees] for x in X]).mean(axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)


class IsolationForestMini:
    def __init__(self, n_trees=50, max_samples=256, contamination=0.15, seed=42):
        self.n_trees=n_trees; self.max_samples=max_samples
        self.contamination=contamination; self.rng=np.random.default_rng(seed)
        self.trees=[]; self.threshold_=0.0

    def _build(self, X, depth=0):
        n, d = X.shape
        if n<=1 or depth>=8: return {"leaf":True,"size":n}
        f=int(self.rng.integers(0,d))
        lo,hi=float(X[:,f].min()),float(X[:,f].max())
        if lo==hi: return {"leaf":True,"size":n}
        t=float(self.rng.uniform(lo,hi)); m=X[:,f]<t
        return {"leaf":False,"f":f,"t":t,"l":self._build(X[m],depth+1),"r":self._build(X[~m],depth+1)}

    def _path(self, node, x, d=0):
        if node["leaf"]:
            n=node["size"]; return d+(2*(math.log(n-1)+0.5772)-2*(n-1)/n if n>1 else 0)
        return self._path(node["l"] if x[node["f"]]<node["t"] else node["r"], x, d+1)

    def fit(self, X):
        n = min(self.max_samples, len(X))
        for _ in range(self.n_trees):
            idx=self.rng.choice(len(X),n,replace=False); self.trees.append(self._build(X[idx]))
        scores=self.score_samples(X); self.threshold_=float(np.percentile(scores,(1-self.contamination)*100))
        return self

    def score_samples(self, X):
        c=lambda n:2*(math.log(n-1)+0.5772)-2*(n-1)/n if n>1 else 1
        avg=np.array([[self._path(t,x) for t in self.trees] for x in X]).mean(axis=1)
        return -2**(-avg/c(self.max_samples))

    def predict_proba(self, X):
        s=-self.score_samples(X); mn,mx=s.min(),s.max(); s=(s-mn)/(mx-mn+1e-9)
        return np.column_stack([1-s,s])

    def predict(self, X):
        return np.where(self.score_samples(X)<self.threshold_,-1,1)


class AutoencoderMini:
    def __init__(self, input_dim, hidden=32, latent=8, seed=42):
        rng=np.random.default_rng(seed); s=np.sqrt(2/input_dim)
        self.W1=rng.normal(0,s,(input_dim,hidden)); self.b1=np.zeros(hidden)
        self.W2=rng.normal(0,s,(hidden,latent));    self.b2=np.zeros(latent)
        self.W3=rng.normal(0,s,(latent,hidden));    self.b3=np.zeros(hidden)
        self.W4=rng.normal(0,s,(hidden,input_dim)); self.b4=np.zeros(input_dim)
        self.threshold=0.0

    def _r(self, x): return np.maximum(0,x)

    def forward(self, X):
        h1=self._r(X@self.W1+self.b1); z=self._r(h1@self.W2+self.b2)
        h3=self._r(z@self.W3+self.b3); return h3@self.W4+self.b4

    def fit(self, X, epochs=25, lr=0.008, batch=128):
        for _ in range(epochs):
            idx=np.random.permutation(len(X))
            for i in range(0,len(X),batch):
                b=X[idx[i:i+batch]]
                h1=self._r(b@self.W1+self.b1); z=self._r(h1@self.W2+self.b2)
                h3=self._r(z@self.W3+self.b3); o=h3@self.W4+self.b4
                d=2*(o-b)/len(b)
                dW4=h3.T@d; db4=d.sum(0)
                d3=(d@self.W4.T)*(h3>0); dW3=z.T@d3; db3=d3.sum(0)
                dz=(d3@self.W3.T)*(z>0); dW2=h1.T@dz; db2=dz.sum(0)
                d1=(dz@self.W2.T)*(h1>0); dW1=b.T@d1; db1=d1.sum(0)
                for dw in [dW1,dW2,dW3,dW4]: np.clip(dw,-1,1,out=dw)
                self.W1-=lr*dW1; self.b1-=lr*db1; self.W2-=lr*dW2; self.b2-=lr*db2
                self.W3-=lr*dW3; self.b3-=lr*db3; self.W4-=lr*dW4; self.b4-=lr*db4
        err=np.mean((X-self.forward(X))**2,axis=1); self.threshold=float(np.percentile(err,95))
        return self

    def score_samples(self, X): return np.mean((X-self.forward(X))**2,axis=1)

    def predict_proba(self, X):
        s=np.clip(self.score_samples(X)/(self.threshold*2+1e-9),0,1)
        return np.column_stack([1-s,s])

    def predict(self, X): return (self.score_samples(X)>self.threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA + TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def _generate_data(n=6000):
    rng=np.random.default_rng(42); n0=int(n*.78); n1=n-n0
    normal=np.column_stack([rng.integers(1024,65535,n0),rng.choice([80,443,53,22,25,3306],n0),
        rng.choice([6,17,1],n0),rng.integers(2,100,n0),rng.integers(200,100_000,n0),
        np.abs(rng.normal(2.5,1.5,n0)),np.abs(rng.normal(30,15,n0)),np.abs(rng.normal(15000,8000,n0)),
        np.abs(rng.normal(500,200,n0)),rng.integers(0,3,n0),rng.integers(1,10,n0),rng.integers(0,2,n0),np.zeros(n0)])
    a1=int(n1*.5); a2=n1-a1
    ddos=np.column_stack([rng.integers(40000,60000,a1),np.full(a1,80),np.full(a1,6),
        rng.integers(5000,50000,a1),rng.integers(500_000,5_000_000,a1),np.abs(rng.normal(0.1,0.05,a1)),
        rng.integers(10000,100000,a1),rng.integers(1_000_000,10_000_000,a1),np.full(a1,100.0),
        rng.integers(4000,40000,a1),np.zeros(a1),np.zeros(a1),np.zeros(a1)])
    scan=np.column_stack([rng.integers(40000,60000,a2),np.arange(1,a2+1)%65535,np.full(a2,6),
        np.ones(a2),rng.integers(40,80,a2),np.full(a2,.001),np.full(a2,200.),np.full(a2,16000.),
        np.full(a2,60.),np.ones(a2),np.zeros(a2),np.zeros(a2),np.zeros(a2)])
    X=np.vstack([normal,ddos,scan]); y=np.array([0]*n0+[1]*a1+[1]*a2)
    idx=np.random.permutation(len(X)); return X[idx],y[idx]


def train_all():
    global MODELS, SCALERS, METRICS
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

    print("[ML] Generating training data …")
    X, y = _generate_data(6000)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    SCALERS["default"] = sc

    def _eval(name, model, t0):
        preds=model.predict(Xte_s); proba=model.predict_proba(Xte_s)[:,1]
        if name=="isolation_forest": preds=(preds==-1).astype(int)
        METRICS[name]={"accuracy":round(float(accuracy_score(yte,preds)),4),
            "precision":round(float(precision_score(yte,preds,zero_division=0)),4),
            "recall":round(float(recall_score(yte,preds,zero_division=0)),4),
            "f1_score":round(float(f1_score(yte,preds,zero_division=0)),4),
            "roc_auc":round(float(roc_auc_score(yte,proba)),4),
            "train_time_s":round(time.time()-t0,2)}
        m=METRICS[name]
        print(f"  [{name:20s}]  acc={m['accuracy']:.3f}  f1={m['f1_score']:.3f}  auc={m['roc_auc']:.3f}")

    for name, cls, kwargs in [
        ("random_forest",    lambda: RandomForestMini(n_trees=25),                          {}),
        ("isolation_forest", lambda: IsolationForestMini(n_trees=50,contamination=0.2),     {}),
        ("autoencoder",      lambda: AutoencoderMini(input_dim=Xtr_s.shape[1]),             {}),
    ]:
        print(f"[ML] Training {name} …")
        t0=time.time(); model=cls()
        if name=="random_forest":    model.fit(Xtr_s, ytr)
        elif name=="isolation_forest": model.fit(Xtr_s)
        else:                        model.fit(Xtr_s, epochs=25, lr=0.008)
        MODELS[name]=model; _eval(name, model, t0)

    print("[ML] All models ready.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE + HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _vec(r):
    dur=max(float(r.get("duration",0.001)),0.001); pkts=max(int(r.get("packets",1)),1); byt=max(int(r.get("bytes",0)),0)
    return np.array([[float(r.get("src_port",0)),float(r.get("dst_port",0)),
        float(PROTOCOL_MAP.get(str(r.get("protocol","TCP")).upper(),6)),
        float(pkts),float(byt),dur,pkts/dur,byt/dur,byt/pkts,
        float(r.get("syn_count",0)),float(r.get("ack_count",0)),
        float(r.get("fin_count",0)),float(r.get("rst_count",0))]])

def _risk(score):
    for lo,hi,label in RISK_BANDS:
        if lo<=score<hi: return label
    return "CRITICAL"

def predict(record, model_name="random_forest"):
    model=MODELS.get(model_name) or list(MODELS.values())[0]
    sc=SCALERS.get("default"); X=_vec(record)
    if sc: X=sc.transform(X)
    proba=model.predict_proba(X)[0]; asc=float(proba[1]); is_a=asc>=0.5; rs=int(asc*100)
    return {"prediction":"ANOMALY" if is_a else "NORMAL","attack_type":random.choice(ATTACK_TYPES) if is_a else None,
            "anomaly_score":round(asc,4),"confidence":round(float(max(proba)),4),
            "risk_level":_risk(rs) if is_a else "LOW","risk_score":rs if is_a else 0,"model_used":model_name}

def _now(): return datetime.now(timezone.utc).isoformat()

def _alert(record, result):
    if result["prediction"]=="ANOMALY" and result["risk_score"]>=30:
        ALERTS.append({"id":len(ALERTS)+1,"timestamp":_now(),"attack_type":result["attack_type"],
            "src_ip":record.get("src_ip","0.0.0.0"),"dst_ip":record.get("dst_ip","0.0.0.0"),
            "src_port":record.get("src_port",0),"dst_port":record.get("dst_port",0),
            "protocol":record.get("protocol","TCP"),"anomaly_score":result["anomaly_score"],
            "confidence":result["confidence"],"risk_level":result["risk_level"],
            "model_used":result["model_used"],"status":"NEW"})

def get_stats():
    total=len(EVENTS); anom=sum(1 for e in EVENTS if e.get("prediction")=="ANOMALY")
    pkts=sum(e.get("packets",0) for e in EVENTS); byt=sum(e.get("bytes",0) for e in EVENTS)
    pc=defaultdict(int); sc=defaultdict(int)
    for e in EVENTS: pc[e.get("protocol","?")]+=1; sc[e.get("src_ip","?")]+=1
    return {"total_events":total,"total_packets":pkts,"total_bytes":byt,"anomaly_count":anom,
        "normal_count":total-anom,"anomaly_percentage":round(anom/max(total,1)*100,1),
        "protocol_distribution":dict(pc),
        "top_src_ips":[{"ip":ip,"count":c} for ip,c in sorted(sc.items(),key=lambda x:-x[1])[:10]],
        "avg_packets_per_flow":round(pkts/max(total,1),1),"avg_bytes_per_flow":round(byt/max(total,1),1)}

def get_timeline(hours=24):
    now=datetime.now(timezone.utc)
    buckets={( now-timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00"):
        {"timestamp":(now-timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00"),"total":0,"anomalies":0,"normal":0}
        for h in range(hours)}
    for e in EVENTS:
        try:
            ts=datetime.fromisoformat(e.get("created_at","")).strftime("%Y-%m-%dT%H:00:00")
            if ts in buckets:
                buckets[ts]["total"]+=1; buckets[ts]["anomalies" if e.get("prediction")=="ANOMALY" else "normal"]+=1
        except: pass
    return sorted(buckets.values(),key=lambda x:x["timestamp"])

def get_threats():
    c=defaultdict(int)
    for e in EVENTS:
        if e.get("prediction")=="ANOMALY": c[e.get("attack_type","Unknown")]+=1
    return {"attack_distribution":dict(c),"total_anomalies":sum(c.values())}

def get_model_list():
    tmap={"random_forest":"Supervised","isolation_forest":"Unsupervised","autoencoder":"Unsupervised"}
    return [{"name":k.replace("_"," ").title(),"key":k,"type":tmap.get(k,"Unknown"),
             "status":"loaded" if k in MODELS else "not_found",
             **{f:METRICS.get(k,{}).get(f) for f in ["accuracy","precision","recall","f1_score","roc_auc","train_time_s"]}}
            for k in ["random_forest","isolation_forest","autoencoder"]]


# ── Demo traffic ───────────────────────────────────────────────────────────────
_NORM=[
    {"src_ip":"192.168.1.10","dst_ip":"8.8.8.8","src_port":52341,"dst_port":53,"protocol":"UDP","packets":2,"bytes":120,"duration":.05},
    {"src_ip":"192.168.1.20","dst_ip":"172.16.0.1","src_port":45231,"dst_port":443,"protocol":"TCP","packets":15,"bytes":8200,"duration":2.1},
    {"src_ip":"192.168.1.30","dst_ip":"10.0.0.5","src_port":33241,"dst_port":80,"protocol":"TCP","packets":8,"bytes":4500,"duration":.8},
    {"src_ip":"10.0.0.5","dst_ip":"192.168.1.100","src_port":3306,"dst_port":49200,"protocol":"TCP","packets":20,"bytes":15000,"duration":3.2},
]
_ANOM=[
    {"src_ip":"198.51.100.99","dst_ip":"192.168.1.1","src_port":54321,"dst_port":80,"protocol":"TCP","packets":50000,"bytes":2500000,"duration":.5,"syn_count":48000},
    {"src_ip":"203.0.113.50","dst_ip":"192.168.1.1","src_port":40000,"dst_port":22,"protocol":"TCP","packets":200,"bytes":12000,"duration":30.},
    {"src_ip":"192.0.2.10","dst_ip":"10.0.0.5","src_port":54321,"dst_port":1,"protocol":"TCP","packets":1,"bytes":60,"duration":.001},
]

def _demo():
    time.sleep(4)
    while True:
        flow=random.choice(_ANOM if random.random()<.3 else _NORM)
        res=predict(flow); EVENTS.append({**flow,**res,"created_at":_now()})
        _alert(flow,res)
        if len(EVENTS)>2000: EVENTS.pop(0)
        time.sleep(random.uniform(.8,2.5))


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES — PAGES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard",
        stats=get_stats(), alerts=list(reversed(ALERTS))[:10],
        timeline=get_timeline(12), threats=get_threats())

@app.route("/traffic")
def traffic():
    stats=get_stats()
    return render_template("traffic.html", active_page="traffic",
        stats=stats, events=list(reversed(EVENTS))[:60],
        proto_data=stats["protocol_distribution"])

@app.route("/detect")
def detect():
    return render_template("detect.html", active_page="detect")

@app.route("/alerts")
def alerts():
    rf=request.args.get("risk",""); sf=request.args.get("status","")
    data=list(reversed(ALERTS))
    if rf: data=[a for a in data if a["risk_level"]==rf]
    if sf: data=[a for a in data if a["status"]==sf]
    return render_template("alerts.html", active_page="alerts", alerts=data)

@app.route("/alerts/<int:alert_id>/update")
def update_alert(alert_id):
    ns=request.args.get("new_status","RESOLVED")
    for a in ALERTS:
        if a["id"]==alert_id: a["status"]=ns; flash(f"Alert #{alert_id} → {ns}","success"); break
    return redirect(url_for("alerts"))

@app.route("/models")
def models():
    return render_template("models.html", active_page="models", models=get_model_list())

@app.route("/analytics")
def analytics():
    return render_template("analytics.html", active_page="analytics",
        stats=get_stats(), timeline=get_timeline(24), threats=get_threats())


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES — JSON API (used by Jinja templates via fetch)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/detect", methods=["POST"])
def api_detect():
    f=request.form
    rec={"src_ip":f.get("src_ip","0.0.0.0"),"dst_ip":f.get("dst_ip","0.0.0.0"),
         "src_port":int(f.get("src_port",0)),"dst_port":int(f.get("dst_port",0)),
         "protocol":f.get("protocol","TCP"),"packets":int(f.get("packets",1)),
         "bytes":int(f.get("bytes",0)),"duration":float(f.get("duration",.001)),
         "syn_count":int(f.get("syn_count",0))}
    res=predict(rec, f.get("model","random_forest"))
    EVENTS.append({**rec,**res,"created_at":_now()})
    if len(EVENTS)>2000: EVENTS.pop(0)
    _alert(rec,res)
    return jsonify({"success":True,"result":res})

@app.route("/api/health")
def api_health():
    return jsonify({"status":"healthy","models":list(MODELS.keys()),"events":len(EVENTS),"alerts":len(ALERTS)})


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   AI-Based Anomaly Detection in Network Traffic       ║")
    print("║   Flask + Pure-Numpy ML — python app.py               ║")
    print("╠══════════════════════════════════════════════════════╣")

    t0=time.time()
    train_all()
    print(f"  Models trained in {time.time()-t0:.1f}s")

    threading.Thread(target=_demo, daemon=True).start()
    print("  Background traffic simulation started")

    print("╠══════════════════════════════════════════════════════╣")
    print(f"  Dashboard  →  http://localhost:{PORT}")
    print(f"  Health     →  http://localhost:{PORT}/api/health")
    print("  Press Ctrl+C to stop")
    print("╚══════════════════════════════════════════════════════╝\n")

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
