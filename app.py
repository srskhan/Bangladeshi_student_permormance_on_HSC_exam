import streamlit as st
import pandas as pd
import pickle



st.set_page_config(
    page_title="HSC Result Prediction",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    background-color: #4F46E5;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #4338CA;
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    text-align: center;
    color: #111827;
}

</style>
""", unsafe_allow_html=True)



with open("/Users/sadikurrahmankhan/Documents/Machine Learning Projects/Bangladeshi Student Permormance on HSC exam/lrmodel.pkl", "rb") as f:
    model = pickle.load(f)

with open("/Users/sadikurrahmankhan/Documents/Machine Learning Projects/Bangladeshi Student Permormance on HSC exam/lrscaler.pkl", "rb") as f:
    scaler = pickle.load(f)


st.title("🎓 HSC Result Prediction System")

st.markdown("###")


col1, col2 = st.columns(2)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        10,
        100,
        18
    )

    address = st.selectbox(
        "Address",
        ["Urban", "Rural"]
    )

    famsize = st.selectbox(
        "Family Size",
        ["LE3", "GT3"]
    )

    Pstatus = st.selectbox(
        "Parent Status",
        ["Together", "Apart"]
    )

    M_Edu = st.selectbox(
        "Mother Education",
        [0, 1, 2, 3, 4]
    )

    F_Edu = st.selectbox(
        "Father Education",
        [0, 1, 2, 3, 4]
    )


with col2:

    M_Job = st.selectbox(
        "Mother Job",
        ["Teacher", "Health", "Services", "At_Home", "Other"]
    )

    F_Job = st.selectbox(
        "Father Job",
        ["Teacher", "Health", "Services", "Farmer", "Other"]
    )

    relationship = st.selectbox(
        "Relationship",
        ["single", "relationship"]
    )

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    tuition_fee = st.number_input(
        "Tuition Fee",
        min_value=0,
        value=5000
    )

    time_friends = st.number_input(
        "Time with Friends",
        min_value=0,
        max_value=24,
        value=2
    )

    ssc_result = st.number_input(
        "SSC GPA",
        min_value=0.0,
        max_value=5.0,
        value=4.0
    )


input_data = pd.DataFrame({
    "gender": [gender],
    "age": [age],
    "address": [address],
    "famsize": [famsize],
    "Pstatus": [Pstatus],
    "M_Edu": [M_Edu],
    "F_Edu": [F_Edu],
    "M_Job": [M_Job],
    "F_Job": [F_Job],
    "relationship": [relationship],
    "smoker": [smoker],
    "tuition_fee": [tuition_fee],
    "time_friends": [time_friends],
    "ssc_result": [ssc_result]
})


input_data = pd.get_dummies(input_data)

expected_columns = scaler.feature_names_in_

for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_columns]


scaled_data = scaler.transform(input_data)


st.markdown("##")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:

    if st.button("Predict HSC Result"):

        prediction = model.predict(scaled_data)

        st.success(
            f"🎯 Predicted HSC GPA: {round(prediction[0], 2)}"
        )