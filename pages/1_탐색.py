import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title="탐색 | 뇌졸중 예측 실습실", page_icon="🔎", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 데이터 탐색")
st.caption("나이, 평균 혈당, 질병 여부, 체질량지수, 흡연 상태와 뇌졸중의 관계를 살펴봅니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

for column in ["age", "avg_glucose_level", "bmi", "stroke"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["뇌졸중 여부"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})


# 1. 나이와 평균 혈당 분포
st.header("1. 나이와 평균 혈당의 분포")
st.write("전체 사람들의 나이와 평균 혈당이 어떤 범위에 많이 분포하는지 확인합니다.")

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        df.dropna(subset=["age"]), x="age", nbins=30,
        title="나이 분포", labels={"age": "나이", "count": "사람 수"},
        color_discrete_sequence=["#4C78A8"]
    )
    fig.update_layout(xaxis_title="나이", yaxis_title="사람 수", bargap=0.08)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        df.dropna(subset=["avg_glucose_level"]), x="avg_glucose_level", nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
        color_discrete_sequence=["#F58518"]
    )
    fig.update_layout(xaxis_title="평균 혈당", yaxis_title="사람 수", bargap=0.08)
    st.plotly_chart(fig, use_container_width=True)


# 2. 뇌졸중 유무에 따른 상자그림과 평균
st.divider()
st.header("2. 뇌졸중 유무에 따른 나이와 평균 혈당 비교")
st.write("상자그림으로 두 그룹의 분포를 비교하고, 표에서 각 그룹의 평균값을 확인합니다.")

col1, col2 = st.columns(2)
with col1:
    fig = px.box(
        df.dropna(subset=["age", "뇌졸중 여부"]),
        x="뇌졸중 여부", y="age", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]},
        title="뇌졸중 유무에 따른 나이",
        labels={"age": "나이", "뇌졸중 여부": "그룹"}, points="outliers"
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="나이")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        df.dropna(subset=["avg_glucose_level", "뇌졸중 여부"]),
        x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": ["뇌졸중 없음", "뇌졸중 있음"]},
        title="뇌졸중 유무에 따른 평균 혈당",
        labels={"avg_glucose_level": "평균 혈당", "뇌졸중 여부": "그룹"},
        points="outliers"
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="평균 혈당")
    st.plotly_chart(fig, use_container_width=True)

mean_table = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .rename(columns={"age": "평균 나이", "avg_glucose_level": "평균 혈당"})
    .round(2)
)
mean_table.index.name = "그룹"
st.markdown("#### 그룹별 평균값")
st.dataframe(mean_table, use_container_width=True)


# 3. 고혈압과 심장병 유무에 따른 뇌졸중 비율
st.divider()
st.header("3. 고혈압·심장병 유무에 따른 뇌졸중 비율")
st.write("각 질환이 있는 그룹과 없는 그룹에서 뇌졸중이 발생한 비율을 비교합니다.")


def make_rate_table(data, column, labels):
    valid = data.dropna(subset=[column, "stroke"]).copy()
    valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=[column])
    result = (
        valid.groupby(column)["stroke"]
        .agg(전체_사람수="count", 뇌졸중_사람수="sum", 뇌졸중_비율="mean")
        .reset_index()
    )
    result["구분"] = result[column].map(labels)
    result["뇌졸중 비율(%)"] = result["뇌졸중_비율"] * 100
    return result


col1, col2 = st.columns(2)
with col1:
    rates = make_rate_table(df, "hypertension", {0: "고혈압 없음", 1: "고혈압 있음"})
    fig = px.bar(
        rates, x="구분", y="뇌졸중 비율(%)",
        text=rates["뇌졸중 비율(%)"].map(lambda x: f"{x:.2f}%"),
        title="고혈압 유무에 따른 뇌졸중 비율", color="구분",
        hover_data={"전체_사람수": True, "뇌졸중_사람수": True, "뇌졸중 비율(%)": ":.2f"}
    )
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    rates = make_rate_table(df, "heart_disease", {0: "심장병 없음", 1: "심장병 있음"})
    fig = px.bar(
        rates, x="구분", y="뇌졸중 비율(%)",
        text=rates["뇌졸중 비율(%)"].map(lambda x: f"{x:.2f}%"),
        title="심장병 유무에 따른 뇌졸중 비율", color="구분",
        hover_data={"전체_사람수": True, "뇌졸중_사람수": True, "뇌졸중 비율(%)": ":.2f"}
    )
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig, use_container_width=True)


# 4. BMI 결측값 그룹과 전체의 뇌졸중 비율
st.divider()
st.header("4. BMI가 비어 있는 사람들의 뇌졸중 비율")
st.write("BMI 값이 없는 사람들만 따로 확인하고, 전체 데이터의 뇌졸중 비율과 비교합니다.")

missing_bmi = df[df["bmi"].isna()].dropna(subset=["stroke"])
overall = df.dropna(subset=["stroke"])
bmi_comparison = pd.DataFrame({
    "구분": ["BMI 빈 값인 사람", "전체 사람"],
    "사람 수": [len(missing_bmi), len(overall)],
    "뇌졸중인 사람 수": [int(missing_bmi["stroke"].eq(1).sum()),
                       int(overall["stroke"].eq(1).sum())],
    "뇌졸중 비율(%)": [
        round(missing_bmi["stroke"].eq(1).mean() * 100, 2) if len(missing_bmi) else 0,
        round(overall["stroke"].eq(1).mean() * 100, 2) if len(overall) else 0
    ]
})
st.dataframe(bmi_comparison, hide_index=True, use_container_width=True)
st.caption(
    f"BMI 빈 값은 {int(df['bmi'].isna().sum()):,}명입니다. "
    "비율은 stroke 값이 확인되는 사람을 기준으로 계산했습니다."
)


# 5. 흡연 상태별 사람 수
st.divider()
st.header("5. 흡연 상태별 사람 수")
st.write("흡연 상태에 기록된 값마다 사람 수를 세어 보여 줍니다. 빈 값도 별도로 표시합니다.")

smoking_counts = (
    df["smoking_status"].fillna("빈 값").astype(str)
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_counts, hide_index=True, use_container_width=True)

st.caption(
    "※ 이 페이지의 통계는 데이터 탐색을 위한 것입니다. "
    "인과관계나 개인의 의료적 위험을 확정하지 않습니다."
)
