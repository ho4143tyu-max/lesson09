import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "기록.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide"
)


# --------------------------------------------------
# 기록 파일 읽기 / 저장하기
# --------------------------------------------------
def load_records():
    """저장된 우리말 뜻과 데이터 출처를 불러온다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            records = json.load(file)

        if not isinstance(records, dict):
            return {"meanings": {}, "source": ""}

        return {
            "meanings": records.get("meanings", {}),
            "source": records.get("source", "")
        }
    except (json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_records(meanings, source):
    """입력한 내용을 JSON 파일로 저장한다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        "meanings": meanings,
        "source": source
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=4)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

# --------------------------------------------------
# 열 정보 만들기
# --------------------------------------------------
records = load_records()
saved_meanings = records.get("meanings", {})

column_info = []

for column in df.columns:
    series = df[column]
    unique_values = series.dropna().unique()
    unique_count = len(unique_values)

    if pd.api.types.is_numeric_dtype(series):
        if unique_count <= 12:
            value_kind = ", ".join(str(value) for value in sorted(unique_values))
        else:
            value_kind = f"수치형 (고유값 {unique_count:,}개)"
    else:
        value_kind = ", ".join(str(value) for value in sorted(unique_values))

    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_kind,
        "빈 값 개수": int(series.isna().sum())
    })

info_df = pd.DataFrame(column_info)


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고 뇌졸중 예측 모델을 이해하기 위한 실습 공간입니다.")

st.markdown(
    """
    ### 데이터 소개
    이 데이터는 사람들의 기본 정보와 건강 관련 정보를 담고 있으며,
    각 사람에게 뇌졸중이 발생했는지를 나타내는 `stroke` 열을 포함하고 있습니다.
    이번 실습에서는 데이터를 먼저 살펴보고, 이후 뇌졸중 예측에 활용할 수 있는
    정보를 이해하는 것을 목표로 합니다.
    """
)

stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (stroke_count / len(df) * 100) if len(df) else 0

st.markdown("### 데이터 한눈에 보기")
card1, card2, card3, card4 = st.columns(4)

card1.metric("전체 사람 수", f"{len(df):,}명")
card2.metric("열 개수", f"{len(df.columns):,}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")

st.divider()

st.markdown("### 열 정보 살펴보기")
st.write(
    "각 열의 이름과 값의 종류, 빈 값 개수를 확인할 수 있습니다. "
    "**우리말 뜻** 칸은 교재를 참고해 직접 입력해 주세요."
)

edited_info = st.data_editor(
    info_df,
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            width="large",
            help="교재를 참고해 각 열의 뜻을 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d개",
            width="small"
        )
    },
    key="column_info_editor"
)

st.divider()

st.markdown("### 데이터 미리보기")
st.write("원본 데이터의 처음 다섯 줄입니다. 데이터의 실제 값을 확인해 보세요.")
st.dataframe(df.head(5), hide_index=True, use_container_width=True)

st.divider()

st.markdown("### 데이터 출처 기록")
source_text = st.text_area(
    "교재에 제시된 데이터 출처를 입력하세요.",
    value=records.get("source", ""),
    height=110,
    placeholder="데이터 출처를 여기에 입력하세요."
)

if st.button("💾 우리말 뜻과 출처 저장", type="primary", use_container_width=True):
    meanings_to_save = {
        str(row["열 이름"]): (
            "" if pd.isna(row["우리말 뜻"]) else str(row["우리말 뜻"])
        )
        for _, row in edited_info.iterrows()
    }

    try:
        save_records(meanings_to_save, source_text)
        st.success(f"입력한 내용이 '{RECORD_FILE}' 파일에 저장되었습니다.")
    except OSError as error:
        st.error("기록 파일을 저장하지 못했습니다.")
        st.code(str(error))
