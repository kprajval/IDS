import joblib
import numpy as np
import argparse
import pandas as pd

model = joblib.load("ensemble_model.joblib")
label_encoder = joblib.load("label_encoder.joblib")

columns = [
    "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count",
    "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts",
    "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min"
]

attack_samples = {
    "syn_flood": [0, 12345, 0, 80, 6, 10000, 1000, 10, 4000, 100, 1500, 0, 1000, 300, 0, 0, 0, 0, 1000000, 100,
                  1, 0.1, 1, 0.1, 1, 0.1, 0.2, 1, 0.1, 1, 0.1, 0.2, 1, 0.1,
                  0, 0, 0, 0, 20, 20, 100, 1, 0, 1500, 800, 300, 100000,
                  0, 1000, 0, 0, 1000, 0, 0, 0, 1, 1500, 1000, 0, 10, 1, 0, 0, 0, 0,
                  10, 1500, 0, 0, 8192, 0, 10, 20, 1, 0.5, 1, 0.2, 0, 0, 0, 0],
    
    "brute_force": [0, 4444, 0, 22, 6, 2000, 40, 20, 1000, 200, 500, 100, 400, 100, 100, 10, 30, 10, 2000, 50,
                    5, 1, 10, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1,
                    0, 0, 0, 0, 20, 20, 50, 20, 100, 800, 450, 120, 14400,
                    0, 0, 0, 0, 20, 0, 0, 0, 1, 900, 500, 100, 10, 2, 0, 1, 1, 0,
                    40, 1000, 20, 200, 512, 1024, 40, 20, 2, 1, 2, 0.5, 0, 0, 0, 0],
    
    "arp_spoof": [0, 0, 0, 0, 1, 1000, 2, 2, 100, 100, 100, 100, 100, 0, 100, 100, 100, 0, 800, 4,
                  2, 0.5, 2, 0.5, 1, 0.5, 0.1, 1, 0.5, 1, 0.5, 0.1, 1, 0.5,
                  0, 0, 0, 0, 20, 20, 2, 2, 100, 100, 100, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 100, 50, 50, 10, 2, 0, 10, 2, 0,
                  2, 100, 2, 100, 1024, 1024, 2, 20, 1, 0.1, 1, 0.1, 0, 0, 0, 0],
    
    "dos": [0, 8888, 0, 80, 6, 50000, 100, 5, 5000, 500, 1000, 0, 500, 250, 0, 0, 0, 0, 500000, 200,
            1, 0.2, 1, 0.1, 1, 0.1, 0.2, 1, 0.1, 1, 0.1, 0.2, 1, 0.1,
            0, 0, 0, 0, 20, 20, 100, 5, 0, 1500, 1000, 300, 90000,
            0, 1, 0, 0, 1, 0, 0, 0, 1, 1400, 1000, 0, 10, 1, 0, 0, 0, 0,
            100, 5000, 5, 500, 4096, 1024, 100, 20, 1, 0.5, 1, 0.2, 0, 0, 0, 0],
    
    "benign": [0]*78
}

def simulate_attack(attack_type):
    if attack_type not in attack_samples:
        print(f"[!] Unknown attack type '{attack_type}'. Choose from: {list(attack_samples.keys())}")
        return

    sample = np.array(attack_samples[attack_type]).reshape(1, -1)
    df_sample = pd.DataFrame(sample, columns=columns)

    prediction = model.predict(df_sample)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    print(f"\n[+] Attack Type Selected: {attack_type.replace('_', ' ').title()}")
    print(f"[+] Predicted Label by Model: {predicted_label}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate and classify network attack using trained model.")
    parser.add_argument("attack", type=str, help="Attack type: syn_flood, brute_force, arp_spoof, dos, benign")
    args = parser.parse_args()

    simulate_attack(args.attack)


