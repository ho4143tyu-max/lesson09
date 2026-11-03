import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

st.set_page_config(page_title="채점표 | 뇌졸중 예측 실습실", page_icon="📝", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📝 채점표")
st.caption("테스트 데이터의 혼동행렬과 분류 성능을 확인합니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()


# 1. 입력 속성 선택
st.header("1. 입력 속성 선택")
selected_features = st.multiselect(
    "모델 학습에 사용할 속성을 선택하세요.",
    options=list(FEATURE_NAMES.keys()),
    default=["age", "avg_glucose_level", "hypertension", "heart_disease"],
    format_func=lambda feature: FEATURE_NAMES[feature]
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# 2. 데이터 정렬 및 훈련 / 테스트 분리
st.header("2. 데이터 분리 및 모델 학습")
df = df.sort_values("id").reset_index(drop=True)

# 10명 중 앞 3명은 테스트, 나머지 7명은 훈련
test_mask = np.arange(len(df)) % 10 < 3
train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

# BMI를 선택한 경우에만 훈련 데이터 중앙값으로 결측값 대체
bmi_median = None
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()
    if pd.isna(bmi_median):
        st.error("훈련 데이터에서 체질량지수 중앙값을 계산할 수 없습니다.")
        st.stop()
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("선택한 속성에 처리되지 않은 빈 값이 있습니다.")
    st.stop()

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "원본 데이터가 5,110행인지 확인해 주세요."
    )

if bmi_median is not None:
    st.info(f"체질량지수 결측값은 훈련 데이터 중앙값 {bmi_median:.2f}으로 대체했습니다.")

positive_test_count = int((y_test == 1).sum())
st.write(
    f"**테스트용 사람 수: {len(test_df):,}명 / "
    f"그중 실제 뇌졸중인 사람 수: {positive_test_count:,}명**"
)


# 3. 로지스틱 회귀 모델
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_scaled, y_train)
logistic_predictions = logistic_model.predict(X_test_scaled)


# 4. 의사결정트리 모델
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)


# 5. 훈련 데이터 다수 클래스만 답하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())
baseline_predictions = np.full(len(y_test), majority_class, dtype=int)


# 6. 혼동행렬과 지표 계산
def confusion_counts(actual, predicted):
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    return {
        "TN": int(((actual == 0) & (predicted == 0)).sum()),
        "FP": int(((actual == 0) & (predicted == 1)).sum()),
        "FN": int(((actual == 1) & (predicted == 0)).sum()),
        "TP": int(((actual == 1) & (predicted == 1)).sum())
    }


def calculate_metrics(counts):
    tn, fp, fn, tp = counts["TN"], counts["FP"], counts["FN"], counts["TP"]
    total = tn + fp + fn + tp
    actual_positive = tp + fn
    predicted_positive = tp + fp

    accuracy = (tp + tn) / total if total else 0.0
    recall = tp / actual_positive if actual_positive else 0.0

    if predicted_positive == 0:
        precision = None
        f1 = None
    else:
        precision = tp / predicted_positive
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if denominator > 0 else 0.0

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


def format_metric(value):
    return "계산할 수 없음" if value is None else f"{value * 100:.2f}%"


def make_confusion_table(counts):
    tn, fp, fn, tp = counts["TN"], counts["FP"], counts["FN"], counts["TP"]
    actual_no, actual_yes = tn + fp, fn + tp
    predicted_no, predicted_yes = tn + fn, fp + tp
    total = actual_no + actual_yes

    return pd.DataFrame(
        {
            "예측: 뇌졸중 아님 (0)": [
                f"{tn:,}명 (TN)", f"{fn:,}명 (FN)", f"{predicted_no:,}명"
            ],
            "예측: 뇌졸중 (1)": [
                f"{fp:,}명 (FP)", f"{tp:,}명 (TP)", f"{predicted_yes:,}명"
            ],
            "행 합계": [
                f"{actual_no:,}명", f"{actual_yes:,}명", f"{total:,}명"
            ]
        },
        index=["실제: 뇌졸중 아님 (0)", "실제: 뇌졸중 (1)", "열 합계"]
    )


# 7. 두 모델의 혼동행렬
st.divider()
st.header("3. 테스트 데이터 혼동행렬")
st.write("행은 실제 상태, 열은 모델의 예측입니다. 양성은 뇌졸중(stroke=1)입니다.")

logistic_counts = confusion_counts(y_test, logistic_predictions)
tree_counts = confusion_counts(y_test, tree_predictions)

model_col1, model_col2 = st.columns(2)
with model_col1:
    st.subheader("로지스틱 회귀(확률로 답하는 모델)")
    st.dataframe(make_confusion_table(logistic_counts), use_container_width=True)
with model_col2:
    st.subheader("의사결정트리(질문으로 답하는 모델)")
    st.dataframe(make_confusion_table(tree_counts), use_container_width=True)


# 8. 네 가지 성능 지표 비교
st.divider()
st.header("4. 성능 지표 비교")
baseline_counts = confusion_counts(y_test, baseline_predictions)

metric_rows = []
for model_name, counts in [
    ("로지스틱 회귀(확률로 답하는 모델)", logistic_counts),
    ("의사결정트리(질문으로 답하는 모델)", tree_counts),
    ("훈련용 다수 클래스 기준 모델", baseline_counts)
]:
    metrics = calculate_metrics(counts)
    metric_rows.append({
        "모델": model_name,
        "정확도": format_metric(metrics["정확도"]),
        "재현율": format_metric(metrics["재현율"]),
        "정밀도": format_metric(metrics["정밀도"]),
        "F1": format_metric(metrics["F1"])
    })

st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
st.caption(
    "정확도 = (TP + TN) / 전체, 재현율 = TP / (TP + FN), "
    "정밀도 = TP / (TP + FP), F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)


# 9. 놓친 환자와 헛짚은 사람 목록
st.divider()
st.header("5. 놓친 사람과 헛짚은 사람 확인")

model_names = [
    "로지스틱 회귀(확률로 답하는 모델)",
    "의사결정트리(질문으로 답하는 모델)"
]
selected_model = st.selectbox("목록을 확인할 모델을 선택하세요.", model_names)

if selected_model == model_names[0]:
    selected_predictions = logistic_predictions
    selected_counts = logistic_counts
else:
    selected_predictions = tree_predictions
    selected_counts = tree_counts

review_df = test_df[
    ["id", "age", "avg_glucose_level", "hypertension", "heart_disease", "stroke"]
].copy()
review_df["예측"] = selected_predictions
review_df["실제"] = y_test.to_numpy()


def make_people_list(data):
    result = data[
        ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
    ].copy()
    result = result.rename(columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부"
    })
    result["고혈압 여부"] = result["고혈압 여부"].map({0: "없음", 1: "있음"})
    result["심장병 여부"] = result["심장병 여부"].map({0: "없음", 1: "있음"})
    result = result.sort_values("나이", ascending=False, na_position="last")
    result["나이"] = result["나이"].round(1)
    result["평균 혈당"] = result["평균 혈당"].round(2)
    return result


missed_people = review_df[
    (review_df["실제"] == 1) & (review_df["예측"] == 0)
]
false_alarm_people = review_df[
    (review_df["실제"] == 0) & (review_df["예측"] == 1)
]

missed_table = make_people_list(missed_people)
false_alarm_table = make_people_list(false_alarm_people)

list_col1, list_col2 = st.columns(2)
with list_col1:
    st.subheader(f"놓친 환자: {len(missed_table):,}명 (FN)")
    st.caption("실제 뇌졸중이지만 모델이 뇌졸중이 아니라고 예측한 사람")
    st.dataframe(missed_table.head(20), hide_index=True, use_container_width=True)
    if len(missed_table) > 20:
        st.caption(f"전체 {len(missed_table):,}명 중 나이가 많은 앞 20명만 표시했습니다.")

with list_col2:
    st.subheader(f"헛짚은 사람: {len(false_alarm_table):,}명 (FP)")
    st.caption("실제 뇌졸중은 아니지만 모델이 뇌졸중이라고 예측한 사람")
    st.dataframe(false_alarm_table.head(20), hide_index=True, use_container_width=True)
    if len(false_alarm_table) > 20:
        st.caption(f"전체 {len(false_alarm_table):,}명 중 나이가 많은 앞 20명만 표시했습니다.")

st.info(
    f"목록 인원 확인 — 놓친 환자 {len(missed_table):,}명 = 혼동행렬 FN "
    f"{selected_counts['FN']:,}명, 헛짚은 사람 {len(false_alarm_table):,}명 = 혼동행렬 FP "
    f"{selected_counts['FP']:,}명"
)

st.caption(
    "※ 이 페이지는 머신러닝 분류 원리를 배우기 위한 교육용 실습입니다. "
    "모델의 예측은 실제 의료 진단이나 개인의 뇌졸중 위험 판정이 아닙니다."
)
