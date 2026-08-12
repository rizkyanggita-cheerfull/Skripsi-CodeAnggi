import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier

DATA_DIR = Path('/mnt/data/fraud_data')
OUT_DIR = Path('/mnt/data/fraud_results')
OUT_DIR.mkdir(exist_ok=True)
CSV_FILES = [
    DATA_DIR / 'Fraudulent_E-Commerce_Transaction_Data.csv',
    DATA_DIR / 'Fraudulent_E-Commerce_Transaction_Data_2.csv',
]
RANDOM_STATE = 42
N_JOBS = 8


def mem(label=''):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / (1024**3)
    print(f'[MEM] {label}: {rss:.2f} GiB', flush=True)


def load_and_engineer():
    usecols = [
        'Transaction Amount', 'Transaction Date', 'Payment Method',
        'Product Category', 'Quantity', 'Customer Age', 'Customer Location',
        'Device Used', 'Shipping Address', 'Billing Address', 'Is Fraudulent',
        'Account Age Days', 'Transaction Hour'
    ]
    dtype = {
        'Transaction Amount': 'float32',
        'Payment Method': 'string',
        'Product Category': 'string',
        'Quantity': 'int16',
        'Customer Age': 'int16',
        'Customer Location': 'string',
        'Device Used': 'string',
        'Shipping Address': 'string',
        'Billing Address': 'string',
        'Is Fraudulent': 'int8',
        'Account Age Days': 'int16',
        'Transaction Hour': 'int8',
    }
    frames = []
    raw_rows = 0
    removed_negative = 0
    removed_fraud = 0
    removed_nonfraud = 0
    for fp in CSV_FILES:
        print(f'Loading {fp.name}', flush=True)
        for chunk in pd.read_csv(fp, usecols=usecols, dtype=dtype, chunksize=150_000):
            raw_rows += len(chunk)
            neg = chunk['Customer Age'] < 0
            if neg.any():
                removed_negative += int(neg.sum())
                removed_fraud += int(chunk.loc[neg, 'Is Fraudulent'].sum())
                removed_nonfraud += int(neg.sum() - chunk.loc[neg, 'Is Fraudulent'].sum())
                chunk = chunk.loc[~neg].copy()

            dt = pd.to_datetime(chunk['Transaction Date'], errors='raise')
            hour = chunk['Transaction Hour'].astype('int8')
            amount = chunk['Transaction Amount'].astype('float32')
            qty = chunk['Quantity'].astype('float32')
            amount_per_item = (amount / qty).astype('float32')
            dow = dt.dt.dayofweek.astype('int8')

            compact = pd.DataFrame({
                'date_ns': dt.astype('int64'),
                'Transaction Amount': amount,
                'Quantity': chunk['Quantity'].astype('int8'),
                'Customer Age': chunk['Customer Age'].astype('int16'),
                'Account Age Days': chunk['Account Age Days'].astype('int16'),
                'Transaction Hour': hour,
                'Customer Location': chunk['Customer Location'],
                'Payment Method': chunk['Payment Method'],
                'Product Category': chunk['Product Category'],
                'Device Used': chunk['Device Used'],
                'Address Match': (chunk['Shipping Address'] == chunk['Billing Address']).astype('int8'),
                'Transaction Month': dt.dt.month.astype('int8'),
                'Transaction Day': dt.dt.day.astype('int8'),
                'Transaction DayOfWeek': dow,
                'Is Weekend': (dow >= 5).astype('int8'),
                # Operational definition used in this execution: 00:00-05:59.
                'Is Night Transaction': (hour < 6).astype('int8'),
                'Transaction Hour Sin': np.sin(2 * np.pi * hour.astype('float32') / 24).astype('float32'),
                'Transaction Hour Cos': np.cos(2 * np.pi * hour.astype('float32') / 24).astype('float32'),
                'DayOfWeek Sin': np.sin(2 * np.pi * dow.astype('float32') / 7).astype('float32'),
                'DayOfWeek Cos': np.cos(2 * np.pi * dow.astype('float32') / 7).astype('float32'),
                'Amount per Item': amount_per_item,
                'Log Transaction Amount': np.log1p(amount).astype('float32'),
                'Log Amount per Item': np.log1p(amount_per_item).astype('float32'),
                'target': chunk['Is Fraudulent'].astype('int8'),
            })
            frames.append(compact)
            del chunk, compact, dt
            gc.collect()

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    # Use categories to reduce memory before sorting/splitting.
    for c in ['Customer Location', 'Payment Method', 'Product Category', 'Device Used']:
        df[c] = df[c].astype('category')
    df.sort_values('date_ns', kind='mergesort', inplace=True)
    df.reset_index(drop=True, inplace=True)

    metadata = {
        'raw_rows': raw_rows,
        'removed_negative_age': removed_negative,
        'removed_fraud': removed_fraud,
        'removed_nonfraud': removed_nonfraud,
        'clean_rows': len(df),
        'clean_fraud': int(df['target'].sum()),
        'clean_nonfraud': int(len(df) - df['target'].sum()),
        'date_min': str(pd.to_datetime(df['date_ns'].iloc[0])),
        'date_max': str(pd.to_datetime(df['date_ns'].iloc[-1])),
        'night_definition': 'Transaction Hour < 6 (00:00-05:59)',
    }
    return df, metadata


def metrics(y_true, prob, threshold=0.5):
    pred = (prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, prob)),
        'pr_auc': float(average_precision_score(y_true, prob)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'predicted_fraud': int(tp + fp),
        'fraud_alert_rate': float((tp + fp) / len(y_true)),
    }


def build_model(scale_pos_weight):
    return XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        tree_method='hist',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=15,
        subsample=0.80,
        colsample_bytree=0.70,
        gamma=1.00,
        reg_alpha=0.01,
        reg_lambda=5.00,
        max_delta_step=5,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        early_stopping_rounds=50,
        scale_pos_weight=scale_pos_weight,
        verbosity=1,
    )


def fit_eval(name, X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight=1.0):
    print(f'\n=== Training {name} | shape={X_train.shape} ===', flush=True)
    mem(name + ' before fit')
    model = build_model(scale_pos_weight)
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    fit_seconds = time.time() - t0
    prob_val = model.predict_proba(X_val)[:, 1].astype('float32')
    prob_test = model.predict_proba(X_test)[:, 1].astype('float32')
    m_val = metrics(y_val, prob_val, 0.5)
    m_test = metrics(y_test, prob_test, 0.5)
    best_iteration = getattr(model, 'best_iteration', None)
    print(name, 'best_iteration=', best_iteration, 'fit_sec=', round(fit_seconds, 1), flush=True)
    print('VAL', m_val, flush=True)
    result = {
        'scenario': name,
        'train_rows': int(len(y_train)),
        'train_fraud': int(np.sum(y_train)),
        'train_nonfraud': int(len(y_train) - np.sum(y_train)),
        'scale_pos_weight': float(scale_pos_weight),
        'best_iteration': None if best_iteration is None else int(best_iteration),
        'fit_seconds': float(fit_seconds),
        **{f'val_{k}': v for k, v in m_val.items()},
        **{f'test_{k}': v for k, v in m_test.items()},
    }
    del model
    gc.collect()
    return result, prob_val, prob_test


def main():
    t_all = time.time()
    df, metadata = load_and_engineer()
    mem('after load/engineering')
    print('Metadata:', metadata, flush=True)
    print('Category counts:', {c: int(df[c].nunique()) for c in ['Customer Location','Payment Method','Product Category','Device Used']}, flush=True)

    n = len(df)
    train_end = int(0.64 * n)
    val_end = int(0.80 * n)
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    del df
    gc.collect()

    metadata.update({
        'train_rows': len(train), 'val_rows': len(val), 'test_rows': len(test),
        'train_fraud': int(train['target'].sum()),
        'val_fraud': int(val['target'].sum()),
        'test_fraud': int(test['target'].sum()),
        'train_nonfraud': int(len(train)-train['target'].sum()),
        'val_nonfraud': int(len(val)-val['target'].sum()),
        'test_nonfraud': int(len(test)-test['target'].sum()),
        'train_start': str(pd.to_datetime(train['date_ns'].iloc[0])),
        'train_end_inclusive': str(pd.to_datetime(train['date_ns'].iloc[-1])),
        'validation_start': str(pd.to_datetime(val['date_ns'].iloc[0])),
        'validation_end_inclusive': str(pd.to_datetime(val['date_ns'].iloc[-1])),
        'test_start': str(pd.to_datetime(test['date_ns'].iloc[0])),
        'test_end_inclusive': str(pd.to_datetime(test['date_ns'].iloc[-1])),
    })
    print('Split metadata:', {k: metadata[k] for k in metadata if 'rows' in k or 'fraud' in k or 'start' in k or 'end_' in k}, flush=True)

    # Frequency encoding learned only from training.
    loc_freq = train['Customer Location'].value_counts(dropna=False) / len(train)
    for part in (train, val, test):
        part['Customer Location Frequency'] = part['Customer Location'].map(loc_freq).astype('float32').fillna(0.0)

    numeric_cols = [
        'Transaction Amount', 'Quantity', 'Customer Age', 'Account Age Days',
        'Transaction Hour', 'Customer Location Frequency', 'Address Match',
        'Transaction Month', 'Transaction Day', 'Transaction DayOfWeek',
        'Is Weekend', 'Is Night Transaction', 'Transaction Hour Sin',
        'Transaction Hour Cos', 'DayOfWeek Sin', 'DayOfWeek Cos',
        'Amount per Item', 'Log Transaction Amount', 'Log Amount per Item',
    ]
    cat_cols = ['Payment Method', 'Product Category', 'Device Used']

    # Ordinal codes for SMOTENC and consistent one-hot encoding.
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32)
    train_cat = ord_enc.fit_transform(train[cat_cols].astype('string'))
    val_cat = ord_enc.transform(val[cat_cols].astype('string'))
    test_cat = ord_enc.transform(test[cat_cols].astype('string'))

    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    train_cat_oh = onehot.fit_transform(train_cat)
    val_cat_oh = onehot.transform(val_cat)
    test_cat_oh = onehot.transform(test_cat)

    X_train_num = train[numeric_cols].to_numpy(dtype=np.float32, copy=True)
    X_val_num = val[numeric_cols].to_numpy(dtype=np.float32, copy=True)
    X_test_num = test[numeric_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = train['target'].to_numpy(dtype=np.int8, copy=True)
    y_val = val['target'].to_numpy(dtype=np.int8, copy=True)
    y_test = test['target'].to_numpy(dtype=np.int8, copy=True)

    X_train_enc = np.concatenate([X_train_num, train_cat_oh], axis=1)
    X_val_enc = np.concatenate([X_val_num, val_cat_oh], axis=1)
    X_test_enc = np.concatenate([X_test_num, test_cat_oh], axis=1)
    X_train_mixed = np.concatenate([X_train_num, train_cat], axis=1)
    X_val_mixed = np.concatenate([X_val_num, val_cat], axis=1)
    X_test_mixed = np.concatenate([X_test_num, test_cat], axis=1)

    metadata['numeric_feature_count'] = len(numeric_cols)
    metadata['onehot_feature_count'] = int(train_cat_oh.shape[1])
    metadata['final_feature_count'] = int(X_train_enc.shape[1])
    metadata['ordinal_categories'] = {cat_cols[i]: [str(x) for x in cats] for i, cats in enumerate(ord_enc.categories_)}

    # Free dataframes and intermediate OHE arrays.
    del train, val, test, train_cat_oh, val_cat_oh, test_cat_oh
    gc.collect()
    mem('arrays ready')
    print('Feature counts:', metadata['numeric_feature_count'], metadata['onehot_feature_count'], metadata['final_feature_count'], flush=True)

    results = []
    probs = {'y_val': y_val, 'y_test': y_test}

    # Baseline and weighted.
    for name, spw in [('S0_Baseline', 1.0), ('S1_Weighted', 23.67372)]:
        res, pv, pt = fit_eval(name, X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test, spw)
        results.append(res); probs[name + '_val'] = pv; probs[name + '_test'] = pt

    # SMOTENC scenarios.
    cat_indices = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))
    for ratio in [0.10, 0.20, 0.30, 0.50]:
        name = f'S2_SMOTENC_{ratio:.2f}'
        print(f'\n--- Resampling {name} ---', flush=True)
        t0 = time.time()
        sampler = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=ratio,
            random_state=RANDOM_STATE,
            k_neighbors=5,
        )
        X_res_mixed, y_res = sampler.fit_resample(X_train_mixed, y_train)
        sample_seconds = time.time() - t0
        # One-hot categorical code columns after SMOTENC.
        X_res_cat_oh = onehot.transform(X_res_mixed[:, len(numeric_cols):])
        X_res_enc = np.concatenate([X_res_mixed[:, :len(numeric_cols)].astype(np.float32, copy=False), X_res_cat_oh], axis=1)
        del X_res_cat_oh, X_res_mixed, sampler
        gc.collect()
        res, pv, pt = fit_eval(name, X_res_enc, y_res.astype(np.int8), X_val_enc, y_val, X_test_enc, y_test, 1.0)
        res['sampling_seconds'] = float(sample_seconds)
        results.append(res); probs[name + '_val'] = pv; probs[name + '_test'] = pt
        del X_res_enc, y_res
        gc.collect(); mem(name + ' after cleanup')

    # RandomUnderSampler scenarios, sampling indices only.
    dummy = np.arange(len(y_train), dtype=np.int32).reshape(-1, 1)
    for ratio in [0.10, 0.20, 0.30, 0.50]:
        name = f'S3_RUS_{ratio:.2f}'
        print(f'\n--- Resampling {name} ---', flush=True)
        t0 = time.time()
        rus = RandomUnderSampler(sampling_strategy=ratio, random_state=RANDOM_STATE)
        idx_res, y_res = rus.fit_resample(dummy, y_train)
        idx = idx_res.ravel()
        X_res_enc = X_train_enc[idx]
        sample_seconds = time.time() - t0
        del rus, idx_res, idx
        gc.collect()
        res, pv, pt = fit_eval(name, X_res_enc, y_res.astype(np.int8), X_val_enc, y_val, X_test_enc, y_test, 1.0)
        res['sampling_seconds'] = float(sample_seconds)
        results.append(res); probs[name + '_val'] = pv; probs[name + '_test'] = pt
        del X_res_enc, y_res
        gc.collect(); mem(name + ' after cleanup')

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / 'threshold_050_results.csv', index=False)
    np.savez_compressed(OUT_DIR / 'scenario_probabilities.npz', **probs)
    metadata['total_runtime_seconds'] = time.time() - t_all
    with open(OUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print('\n=== FINAL VALIDATION RESULTS ===', flush=True)
    show_cols = ['scenario','train_rows','train_fraud','train_nonfraud','best_iteration','fit_seconds',
                 'val_accuracy','val_precision','val_recall','val_f1_score','val_roc_auc','val_pr_auc',
                 'val_tp','val_tn','val_fp','val_fn','val_fraud_alert_rate']
    print(results_df[show_cols].to_string(index=False), flush=True)
    print(f'Outputs saved to {OUT_DIR}', flush=True)
    mem('final')


if __name__ == '__main__':
    main()
