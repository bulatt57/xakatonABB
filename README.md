# xakatonABB
#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import time

start = time.perf_counter()
DATA_DIR = "selection"

# 1. Загрузка
train_x = pd.read_csv(f"{DATA_DIR}/x_train.csv")
train_y = pd.read_csv(f"{DATA_DIR}/y_train.csv")
test_x  = pd.read_csv(f"{DATA_DIR}/x_test.csv")
build   = pd.read_excel(f"{DATA_DIR}/Kazan_minzhkh.xlsx")

# 2. Merge данных
train = train_x.merge(train_y, on="uuid").merge(build, on="Короткий Адрес")
test  = test_x.merge(build,   on="Короткий Адрес")

# 3. Feature engineering
# 3.1 Relative floor
train["Этажей в Доме"] = train["Этажей в Доме"].replace(0, np.nan)
test ["Этажей в Доме"] = test ["Этажей в Доме"].replace(0, np.nan)
train["Относительный_этаж"] = train["Этаж Квартиры"] / train["Этажей в Доме"]
test ["Относительный_этаж"] = test ["Этаж Квартиры"] / test ["Этажей в Доме"]

# 3.2 Age and age^2
current_year = pd.Timestamp.now().year
train["Возраст_дома"] = current_year - train["Год Постройки"]
test ["Возраст_дома"] = current_year - test ["Год Постройки"]
train["age_squared"] = train["Возраст_дома"] ** 2
test ["age_squared"] = test ["Возраст_дома"] ** 2

# 3.3 Area per floor
train["area_per_floor"] = train["Площадь"] / train["Этаж Квартиры"]
test ["area_per_floor"] = test ["Площадь"] / test ["Этаж Квартиры"]

# 3.4 Mean price by district
district_mean = train.groupby("Район")["Цена за квадрат"].mean().rename("mean_price_by_district")
train = train.merge(district_mean, on="Район", how="left")
test  = test.merge(district_mean, on="Район", how="left")
median_district = train["mean_price_by_district"].median()
train["mean_price_by_district"].fillna(median_district, inplace=True)
test ["mean_price_by_district"].fillna(median_district, inplace=True)

# 3.5 Coord clusters
coords = train[["Широта","Долгота"]].dropna()
kmeans = KMeans(n_clusters=5, random_state=42).fit(coords)
train["coord_cluster"] = kmeans.predict(train[["Широта","Долгота"]].fillna(-1)).astype(str)
test ["coord_cluster"] = kmeans.predict(test [["Широта","Долгота"]].fillna(-1)).astype(str)

# 4. Признаки
features = [
    "Площадь","Кол-во Комнат","Этаж Квартиры","Этажей в Доме",
    "Относительный_этаж","Возраст_дома","age_squared","area_per_floor",
    "Материал Стен","Газ","Район","Ремонт","Экспозиция",
    "mean_price_by_district","coord_cluster"
]
cat_features = ["Материал Стен","Газ","Район","Ремонт","coord_cluster"]
num_features = [f for f in features if f not in cat_features]

# 5. Заполнение NaN
medians = train[num_features].median()
train[num_features] = train[num_features].fillna(medians)
test [num_features] = test [num_features].fillna(medians)
for c in cat_features:
    train[c] = train[c].fillna("missing").astype(str)
    test [c]  = test[c].fillna("missing").astype(str)

# 6. Подготовка X,y с логарифмом
X = train[features]
y = np.log1p(train["Цена за квадрат"])

# 7. CV для оценки
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    mdl = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=7,
        border_count=64,
        random_strength=1,
        verbose=0
    )
    mdl.fit(X_tr, y_tr, cat_features=cat_features)
    pred = np.expm1(mdl.predict(X_val))
    rmse = np.sqrt(mean_squared_error(train.loc[val_idx,"Цена за квадрат"], pred))
    scores.append(rmse)
print(f"Mean CV RMSE (лог-модель): {np.mean(scores):.2f}")

# 8. Финальное обучение с early stopping
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=7,
    border_count=64,
    random_strength=1,
    random_state=42,
    verbose=100
)
model.fit(X_tr, y_tr,
          eval_set=(X_val,y_val),
          cat_features=cat_features,
          early_stopping_rounds=100
)

# 9. Сохранение
pickle.dump(model, open("model.pkl","wb"))
with open("features.txt","w",encoding="utf-8") as f:
    for feat in features:
        f.write(feat+"\n")

# 10. Предсказание
X_test = test[features]
y_pred = np.expm1(model.predict(X_test))

test_uuid = test_x.uuid  # оригинальный порядок uuid из x_test
submission_file = pd.DataFrame(list(zip(test_uuid, y_pred.flatten())), columns=['uuid','Цена за квадрат'])
submission_file.to_csv('predictions.csv', index=False)

# Логирование
total_time = time.perf_counter() - start
print(f"Done in {total_time:.1f}s. Files: model.pkl, features.txt, predictions.csv")
print("Feature importances:")
print(model.get_feature_importance(prettified=True))
