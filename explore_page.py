from cProfile import label
from numpy import true_divide
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# if the number of samples is greater or equal than the cutoff value then we keep it, otherwise combine it to a diff category
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

# convert to float and make another cut off
def clean_experience(x):
    if x == 'Less than 1 year':
        return 0.5
    if x == 'More than 50 years':
        return 50
    return float(x)

# keeping bachelors and masters
# unifying professional and other doctoral as 'postgrad'
# everything else is assuming its less than a bachelors
def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelors degree'
    if 'Master’s degree' in x:
        return 'Masters degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

@st.cache       # when we have executed this one time, we cache it for next session
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    # cleaning the dataframe. we only want to keep a few columns
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]

    df = df[df["ConvertedCompYearly"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    # executing this function
    country_map = shorten_categories(df.Country.value_counts(), 400)
    df['Country'] = df['Country'].map(country_map)
    df = df[df["ConvertedCompYearly"] <= 250000]
    df = df[df["ConvertedCompYearly"] >= 10000]
    df = df[df["Country"] != "Other"]

    df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)
    df['EdLevel'] = df['EdLevel'].apply(clean_education)

    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)

    return df

df = load_data()

# streamlit displaying page
def show_explore_page():
    st.title("Explore Software Engineering Salaries")

    st.write("""### Stack Overflow Developer Survey 2021""")

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")       # equal makes the pie chart a circle

    # display the pie chart
    st.write("""#### Number of data from different countries""")
    st.pyplot(fig1)

    st.write("""### Mean salary based on country""")
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("""### Mean salary based on experience""")
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)
