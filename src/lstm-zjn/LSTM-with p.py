import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================
INPUT_LEN = 30
PRED_LEN = 10


# ==================== 构造滑动窗口样本 ====================
def make_windows(a):
    X, y = [], []
    for i in range(len(a) - INPUT_LEN - PRED_LEN + 1):
        X.append(a[i:i + INPUT_LEN])
        y.append(a[i + INPUT_LEN:i + INPUT_LEN + PRED_LEN])
    return np.array(X)[..., None], np.array(y)

def make_multivar_windows(features, targets):
    X, y = [], []
    for i in range(len(targets) - INPUT_LEN - PRED_LEN + 1):
        X.append(features[i:i + INPUT_LEN])
        y.append(targets[i + INPUT_LEN:i + INPUT_LEN + PRED_LEN])
    return np.array(X), np.array(y)

# ==================== 主函数 ====================
if __name__ == "__main__":

    DATA_DIR = r"C:\Users\22177\Desktop\ERA5"
    SUBSET_SIZE = 100
    EPOCHS = 20
    BATCH_SIZE = 64
    SEED = 42

    # 设置随机种子以保证复现性
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # files = sorted(glob.glob(os.path.join(DATA_DIR, "daily_pixel_data_*.parquet")))[:5]

    files = sorted(glob.glob(os.path.join(DATA_DIR, "daily_pixel_data_*.parquet")))


    df_all = pd.concat([
        # pd.read_parquet(f, columns=["date", "latitude", "longitude", "swvl1_m3m3"])
        pd.read_parquet(f, columns=["date", "latitude", "longitude", "swvl1_m3m3", "tp_mm_day", "e_mm_day","ro_mm_day"])
        .assign(date=lambda d: pd.to_datetime(d["date"], format="%Y-%m-%d", errors="coerce"),
                year=lambda d: pd.to_datetime(d["date"], errors="coerce").dt.year)
        for f in files
    ], ignore_index=True).set_index("date")

    grps = df_all.groupby(["latitude", "longitude"], sort=False)
    print(f"准备聚类选择代表点...")

    # ==== 1. 提取统计特征（均值/方差/极差） ====
    agg_stats = grps["swvl1_m3m3"].agg(["mean", "std", lambda x: x.max() - x.min()])
    agg_stats.columns = ["mean", "std", "range"]
    agg_stats = agg_stats.dropna()

    # ==== 2. KMeans 聚类后选出中心最近点 ====
    k = SUBSET_SIZE  # 与原本SUBSET_SIZE一致
    X = agg_stats.values
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
    labels = kmeans.labels_

    selected_indices = []
    for i in range(k):
        idx = np.where(labels == i)[0]
        center = kmeans.cluster_centers_[i]
        dist = np.linalg.norm(X[idx] - center, axis=1)
        selected_indices.append(idx[np.argmin(dist)])

    selected_ids = agg_stats.index[selected_indices]  # index为 (lat, lon) tuple

    # 替换原来的 subset_keys
    subset_keys = selected_ids.tolist()

    print(f"已选出代表点数：{len(subset_keys)}")

    # subset_keys = list(grps.groups.keys())[:SUBSET_SIZE]

    # 存储所有网格点的评估结果
    results = []

    for lat, lon in subset_keys:
        try:
            df = grps.get_group((lat, lon))
            # arr = df["swvl1_m3m3"].fillna(method="ffill").values.astype("float32")
            #arr = df["swvl1_m3m3"].ffill().values.astype("float32")

            #yrs = df["year"].values

            #train_s = arr[np.isin(yrs, [2020, 2021])]
            #val_s = arr[yrs == 2022]
            #test_s = arr[yrs == 2023]

            #if any(len(x) < INPUT_LEN + PRED_LEN for x in (train_s, val_s, test_s)):
            #    print(f"[跳过] ({lat:.2f},{lon:.2f}) 数据不足")
            #    continue

            ## 归一化
            #scaler = MinMaxScaler()
            #train_sc = scaler.fit_transform(train_s.reshape(-1, 1)).flatten()
            #val_sc = scaler.transform(val_s.reshape(-1, 1)).flatten()
            #test_sc = scaler.transform(test_s.reshape(-1, 1)).flatten()

            ## 构造窗口
            #X_tr, y_tr = make_windows(train_sc)
            #X_va, y_va = make_windows(val_sc)
            #X_te, y_te = make_windows(test_sc)

            # 填充并提取四个变量（目标+3特征）
            vars_all = df[["swvl1_m3m3", "tp_mm_day", "e_mm_day", "ro_mm_day"]].ffill().values.astype("float32")
            yrs = df["year"].values

            # 拆分为输入和目标
            train_v = vars_all[np.isin(yrs, [2020, 2021])]
            val_v = vars_all[yrs == 2022]
            test_v = vars_all[yrs == 2023]

            if any(len(x) < INPUT_LEN + PRED_LEN for x in (train_v, val_v, test_v)):
                print(f"[跳过] ({lat:.2f},{lon:.2f}) 数据不足")
                continue

            # 特征包括 swvl1、tp、e、ro → 一起作为输入
            train_x, train_y = train_v[:, :], train_v[:, 0]
            val_x, val_y = val_v[:, :], val_v[:, 0]
            test_x, test_y = test_v[:, :], test_v[:, 0]

            # 归一化（每个点的4个输入特征和目标分别缩放）
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            train_x = scaler_x.fit_transform(train_x.reshape(-1, 4)).reshape(-1, 4)
            train_y = scaler_y.fit_transform(train_y.reshape(-1, 1)).flatten()
            val_x = scaler_x.transform(val_x.reshape(-1, 4)).reshape(-1, 4)
            val_y = scaler_y.transform(val_y.reshape(-1, 1)).flatten()
            test_x = scaler_x.transform(test_x.reshape(-1, 4)).reshape(-1, 4)
            test_y = scaler_y.transform(test_y.reshape(-1, 1)).flatten()

            # 构造时间窗
            X_tr, y_tr = make_multivar_windows(train_x, train_y)
            X_va, y_va = make_multivar_windows(val_x, val_y)
            X_te, y_te = make_multivar_windows(test_x, test_y)

            # 模型定义：输入 shape=(INPUT_LEN, 4)
            model = Sequential([
                Input(shape=(INPUT_LEN, 4)),
                LSTM(64),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dense(PRED_LEN)
            ])

            # 构建模型
            #model = Sequential([
            #    LSTM(64, input_shape=(INPUT_LEN, 1)),
            #    Dropout(0.2),
            #    Dense(32, activation="relu"),
            #    Dense(PRED_LEN)
            #])
            model.compile(optimizer="adam", loss="mse")

            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0
            )

            # 预测与评价
            y_pred = model.predict(X_te, verbose=0)
            #y_pred_inv = scaler.inverse_transform(y_pred)
            #y_true_inv = scaler.inverse_transform(y_te)

            y_pred_inv = scaler_y.inverse_transform(y_pred)
            y_true_inv = scaler_y.inverse_transform(y_te)

            rmse = np.sqrt(((y_pred_inv - y_true_inv) ** 2).mean())

            # 在 try 内部 RMSE 后添加：
            mae = mean_absolute_error(y_true_inv, y_pred_inv)
            r2 = r2_score(y_true_inv, y_pred_inv)

            results.append({
                "lat": lat,
                "lon": lon,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            })

            print(f"[✓] Grid ({lat:.2f},{lon:.2f}) → Test RMSE: {rmse:.4f}")

        except Exception as e:
            print(f"[错误] Grid ({lat:.2f},{lon:.2f}) 处理失败：{str(e)}")



    # 转换为 DataFrame
    results_df = pd.DataFrame(results)

    # --- 折线图展示每个点的 RMSE ---
    plt.figure(figsize=(10, 5))
    plt.plot(results_df["rmse"].values, marker="o")
    plt.title("RMSE of Each Grid Point")
    plt.xlabel("Grid Point Index")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 散点图：按地理坐标画 RMSE 热力图 ---
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(results_df["lon"], results_df["lat"], c=results_df["rmse"], cmap="coolwarm", s=60)
    plt.colorbar(sc, label="Test RMSE")
    plt.title("RMSE Distribution on Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Boxplot 展示误差分布 ---
    plt.figure(figsize=(10, 5))
    results_df[["rmse", "mae"]].plot(kind="box")
    plt.title("Distribution of RMSE and MAE Across Grid Points")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 打印统计摘要 ---
    print("评价指标统计：")
    print(results_df[["rmse", "mae", "r2"]].describe())