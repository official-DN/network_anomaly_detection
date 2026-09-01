"""
Generate synthetic traffic records and submit to the running API for demo.
Usage: python scripts/generate_demo_traffic.py --n 200
"""
import sys, os, time, argparse, random
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

API_BASE = "http://127.0.0.1:8000/api"

NORMAL_FLOWS = [
    {"src_ip":"192.168.1.10","dst_ip":"8.8.8.8","src_port":52341,"dst_port":53,"protocol":"UDP","packets":2,"bytes":120,"duration":0.05},
    {"src_ip":"192.168.1.20","dst_ip":"172.16.0.1","src_port":45231,"dst_port":443,"protocol":"TCP","packets":15,"bytes":8200,"duration":2.1},
    {"src_ip":"192.168.1.30","dst_ip":"10.0.0.5","src_port":33241,"dst_port":80,"protocol":"TCP","packets":8,"bytes":4500,"duration":0.8},
]
ANOMALY_FLOWS = [
    {"src_ip":"198.51.100.99","dst_ip":"192.168.1.1","src_port":54321,"dst_port":80,"protocol":"TCP","packets":50000,"bytes":2500000,"duration":0.5,"syn_count":48000,"ack_count":0},
    {"src_ip":"203.0.113.50","dst_ip":"192.168.1.1","src_port":40000,"dst_port":22,"protocol":"TCP","packets":200,"bytes":12000,"duration":30.0},
    {"src_ip":"192.0.2.10","dst_ip":"10.0.0.5","src_port":54321,"dst_port":1,"protocol":"TCP","packets":1,"bytes":60,"duration":0.001},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--model', default='random_forest')
    args = parser.parse_args()

    headers = {}
    try:
        res = requests.post(f"{API_BASE}/auth/login", data={"username":"admin","password":"admin123"})
        if res.ok:
            headers["Authorization"] = f"Bearer {res.json()['access_token']}"
            print("Authenticated as admin")
    except Exception:
        print("Auth failed — proceeding unauthenticated")

    sent = anomalies = 0
    for _ in range(args.n):
        if random.random() < 0.8:
            flow = random.choice(NORMAL_FLOWS)
        else:
            flow = random.choice(ANOMALY_FLOWS)
        try:
            res = requests.post(f"{API_BASE}/traffic/analyze?model={args.model}", json=flow, headers=headers, timeout=5)
            if res.ok:
                d = res.json().get("data", {})
                if d.get("prediction") == "ANOMALY":
                    anomalies += 1
                    print(f"  ANOMALY [{d.get('risk_level')}] score={d.get('anomaly_score',0):.2f} from {flow['src_ip']}")
                sent += 1
            time.sleep(0.05)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nSent {sent} flows | Anomalies detected: {anomalies} ({anomalies/max(sent,1)*100:.1f}%)")

if __name__ == "__main__":
    main()
