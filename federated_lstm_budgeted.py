# ───────── Install if needed ───────── #
# !pip install tensorflow==2.18.0 numpy==1.26.4 pandas==2.2.2 datasets==2.18.0 pyarrow==14.0.2 --force-reinstall
# import os; os.kill(os.getpid(), 9)

import numpy as np
import tensorflow as tf
import gzip, io, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from datasets import load_dataset

# ───────── Load and preprocess dataset ───────── #
dataset = load_dataset("tulipa762/electricity_load_diagrams")
df = dataset["train"].to_pandas()

df["target_scalar"] = df["target"].apply(lambda x: x[0])
df["feat_static_cat"] = df["feat_static_cat"].apply(lambda x: x[0])
df["datetime"] = pd.to_datetime(df["start"])
df["hour"] = df["datetime"].dt.hour
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

cols_in = ['feat_static_cat', 'hour', 'dayofweek', 'month']
col_out = ['target_scalar']

X_raw = df[cols_in].astype("float32").to_numpy()
y_raw = df[col_out].astype("float32").to_numpy()

scaler_x = StandardScaler().fit(X_raw)
X = scaler_x.transform(X_raw)

scaler_y = StandardScaler().fit(y_raw)
y = scaler_y.transform(y_raw)

# ───────── Sequence generator for LSTM ───────── #
SEQUENCE_LENGTH = 10

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X, y, SEQUENCE_LENGTH)

# ───────── Train/val/test split ───────── #
x_train, x_tmp, y_train, y_tmp = train_test_split(X_seq, y_seq, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.50, random_state=42)

# ───────── Split into clients ───────── #
NUM_CLIENTS = 3
chunk = len(x_train) // NUM_CLIENTS
clients = [(x_train[i*chunk:(i+1)*chunk], y_train[i*chunk:(i+1)*chunk]) for i in range(NUM_CLIENTS)]
client_budgets = [0.3, 0.4, 0.8]  # in KB, per-client adaptive budget

# ───────── DeltaCompressor (with target_kb) ───────── #
class DeltaCompressor:
    def __init__(self):
        self.res = None

    @staticmethod
    def _flat(ws):
        flat, shp = [], []
        for w in ws:
            shp.append(w.shape)
            flat.append(w.reshape(-1))
        return np.concatenate(flat), shp

    @staticmethod
    def _unflat(vec, shp):
        out, idx = [], 0
        for s in shp:
            n = np.prod(s)
            out.append(vec[idx:idx+n].reshape(s))
            idx += n
        return out

    def compress(self, delta, target_kb=1.0):
        if self.res is not None:
            delta = [d + r for d, r in zip(delta, self.res)]

        vec, shp = self._flat(delta)
        abs_vec = np.abs(vec)
        lo, hi = abs_vec.min(), abs_vec.max()
        best_blob = None

        for _ in range(15):
            thresh = (lo + hi) / 2
            mask = abs_vec >= thresh
            idx = np.where(mask)[0].astype(np.int32)
            vals = vec[idx]
            sign = np.sign(vals).astype(np.int8)
            scale = np.float32(np.abs(vals).mean()) if vals.size else np.float32(0.0)

            recon = np.zeros_like(vec)
            recon[idx] = sign.astype(np.float32) * scale
            self.res = self._unflat(vec - recon, shp)

            payload = {"i": idx, "s": sign, "k": scale, "sh": shp}
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as f:
                f.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            blob = buf.getvalue()

            if len(blob) / 1024 <= target_kb:
                best_blob = blob
                hi = thresh
            else:
                lo = thresh

        return best_blob if best_blob is not None else blob

    def decompress(self, blob):
        p = pickle.loads(gzip.GzipFile(fileobj=io.BytesIO(blob)).read())
        vec = np.zeros(sum(np.prod(s) for s in p["sh"]), dtype=np.float32)
        vec[p["i"]] = p["s"].astype(np.float32) * p["k"]
        return self._unflat(vec, p["sh"])

# ───────── Build LSTM model ───────── #
def build_lstm(lr=1e-2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, len(cols_in))),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9), loss="mae")
    return model

# ───────── Federated Learning with adaptive compression ───────── #
ROUNDS, LOCAL_EPOCHS, BATCH = 100, 2, 64
comp = DeltaCompressor()
global_model = build_lstm()
global_w = global_model.get_weights()

results = []

for r in range(1, ROUNDS + 1):
    uplink_bytes, deltas = [], []

    for i, (cx, cy) in enumerate(clients):
        local = build_lstm()
        local.set_weights(global_w)
        local.fit(cx, cy, LOCAL_EPOCHS, BATCH,
                  validation_data=(x_val, y_val),
                  callbacks=[EarlyStopping("val_loss", patience=2, restore_best_weights=True, verbose=0)],
                  verbose=0)

        delta = [p - g for p, g in zip(local.get_weights(), global_w)]
        budget_kb = client_budgets[i]
        blob = comp.compress(delta, target_kb=budget_kb)
        uplink_bytes.append(len(blob))
        deltas.append(comp.decompress(blob))
        tf.keras.backend.clear_session()

    agg = [np.mean([d[i] for d in deltas], axis=0) for i in range(len(global_w))]
    global_w = [g + a for g, a in zip(global_w, agg)]
    global_model.set_weights(global_w)

    if r == 1 or r % 10 == 0:
        val_preds = global_model.predict(x_val, verbose=0)
        val_mae = np.mean(np.abs(scaler_y.inverse_transform(val_preds) - scaler_y.inverse_transform(y_val)))
        print(f"Round {r:3d} | upload {np.mean(uplink_bytes)/1024:.2f} KB | val MAE: {val_mae:.2f}")

    results.append({"round": r, "uplink_kb": np.mean(uplink_bytes)/1024})

# ───────── Final Evaluation ───────── #
test_preds = global_model.predict(x_test, verbose=0)
test_mae = np.mean(np.abs(scaler_y.inverse_transform(test_preds) - scaler_y.inverse_transform(y_test)))
print("\nFinal Test MAE:", test_mae)

# ───────── Save results ───────── #
pd.DataFrame(results).to_csv("federated_lstm_compression_stats.csv", index=False)
print("Saved compression stats.")

# ───────── Reproduced Results ───────── #
# Dataset : tulipa762/electricity_load_diagrams
# Clients : 3  |  Rounds : 100  |  Local epochs : 2  |  Batch : 64
# Sequence length : 10  |  Input : feat_static_cat + hour + dayofweek + month
# Compressor : DeltaCompressor with per-client budgets [0.3, 0.4, 0.8] KB
#
# Round   1 | upload 0.50 KB | val MAE: 3193.16
# Round  10 | upload 0.50 KB | val MAE: 3085.94
# Round  20 | upload 0.49 KB | val MAE: 3083.01
# Round  30 | upload 0.50 KB | val MAE: 3083.88
# Round  40 | upload 0.50 KB | val MAE: 3082.76
# Round  50 | upload 0.50 KB | val MAE: 3085.74
# Round  60 | upload 0.50 KB | val MAE: 3084.01
# Round  70 | upload 0.50 KB | val MAE: 3084.50
# Round  80 | upload 0.50 KB | val MAE: 3086.44
# Round  90 | upload 0.49 KB | val MAE: 3084.82
# Round 100 | upload 0.50 KB | val MAE: 3090.84
#
# Final Test MAE : 644.46
# ─────────────────────────────────────── #
