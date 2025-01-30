import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit_app as st
import base64
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
#######################################
# PAGE SETUP
#######################################

st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Sales Dashboard Analysis")
st.markdown("_Prototype v0.4.1_")
df = pd.DataFrame()
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    if not df.empty:
        selected_cities = st.multiselect("Select Cities", df["City"].unique(), default=df["City"].unique())
    else:
        selected_cities = None

if uploaded_file is None:
    st.info(" Upload a file through config", icon="ℹ️")
    st.stop()

# Check if selected_cities is not None before filtering the DataFrame
if selected_cities:
    # Filter data based on the selected cities
    filtered_df = df[df["City"].isin(selected_cities)]
else:
    # If no cities are selected, show the total output without filtering
    filtered_df = df.copy()

    #######################################
    # DATA LOADING
    #######################################

    # Update the 'DateTime' column
filtered_df['DateTime'] = pd.to_datetime(filtered_df['Date'] + ' ' + filtered_df['Time'])

    # Drop the original "Date" and "Time" columns
filtered_df = filtered_df.drop(['Date', 'Time'], axis=1)

with st.expander("Data Preview"):
    st.dataframe(filtered_df)

#######################################
# VISUALIZATION METHODS
#######################################

def plot_metric(label, value, prefix="", suffix="", show_graph=False, color_graph=""):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,
            gauge={"axis": {"visible": False}},
            number={
                "prefix": prefix,
                "suffix": suffix,
                "font.size": 30,
            },
            title={
                "text": label,
                "font": {"size": 26},
            },
        )
    )

    if show_graph:
        fig.add_trace(
            go.Scatter(
                y=[50, 75, 40, 60, 80, 30, 70, 45, 65, 55, 85, 90, 20, 95, 25, 35, 15, 100, 10, 5, 55, 70, 40, 30, 80, 60, 90, 50, 75, 85],
                hoverinfo="skip",
                fill="tozeroy",
                fillcolor=color_graph,
                line={
                    "color": color_graph,
                },
            )
        )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        plot_bgcolor="white",
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_gauge(
    indicator_number, indicator_color, indicator_suffix, indicator_title, max_bound
):
    fig = go.Figure(
        go.Indicator(
            value=indicator_number,
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": indicator_suffix,
                "font.size": 26,
            },
            gauge={
                "axis": {"range": [0, max_bound], "tickwidth": 1},
                "bar": {"color": indicator_color},
            },
            title={
                "text": indicator_title,
                "font": {"size": 18},
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_top_right():
    sales_data = filtered_df.groupby(['Gender', 'Branch']).agg({'Total': 'sum'}).reset_index()

    fig = px.bar(
            sales_data,
            x="Branch",
            y="Total",
            color="Gender",
            barmode="group",
            text_auto=".2s",
            title=f"Total Sales by Gender and Branch",
            height=400,
        )
    fig.update_traces(
            textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
        )
    st.plotly_chart(fig, use_container_width=True)
    # Expander for Pie Chart data
    #with st.expander("Total Sales by Gender and Branch_viewdata", expanded=False):
         #st.write(sales_data)


    #st.download_button(label="Download Data", data=sales_data.to_csv() , file_name="sales_data.csv")

def plot_bottom_left():
        sales_data = filtered_df.groupby('Product line').agg({'Rating': 'mean'}).reset_index()

        fig = px.scatter(
            sales_data,
            x="Product line",
            y="Rating",
            title=f" Average Rating by Product Line",
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Average Rating by Product Line_viewdata", expanded=False):
         st.write(sales_data)


        st.download_button(label="Download Data", data=sales_data.to_csv() , file_name="sale_data.csv")

def plot_bottom_right():
        sales_data = filtered_df.groupby('City').agg({'Tax 5%': 'sum'}).reset_index()

        fig = px.pie(
            sales_data,
            names="City",
            values="Tax 5%",
            title="Total Tax Distribution by City",
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Total Tax Distribution by City", expanded=False):
         st.write(sales_data)


        st.download_button(label="Download Data", data=sales_data.to_csv() , file_name="saless_data.csv")
        
def plot_time_series():
    st.title("Time Series Analysis - Total Sales Over Time")

    # Filter data based on the selected time range
    start_date = st.date_input("Select Start Date", min_value=filtered_df['DateTime'].min(), max_value=filtered_df['DateTime'].max(), value=filtered_df['DateTime'].min())
    end_date = st.date_input("Select End Date", min_value=filtered_df['DateTime'].min(), max_value=filtered_df['DateTime'].max(), value=filtered_df['DateTime'].max())

    # Convert start_date and end_date to datetime64[ns]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    time_series_data = filtered_df[(filtered_df['DateTime'] >= start_date) & (filtered_df['DateTime'] <= end_date)].resample('D', on='DateTime').sum()

    fig = px.line(
        time_series_data,
        x=time_series_data.index,
        y="Total",
        title="Time Series Analysis - Total Sales Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Expander for Time Series data
    with st.expander("Time Series Analysis - Total Sales Over Time_viewdata", expanded=False):
        st.write(time_series_data)

    # Download button for Time Series data
    st.download_button(label="Download Data", data=time_series_data.to_csv(), file_name="time_series_data.csv")
def download_button(df, label, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download {label}</a>'
    st.markdown(href, unsafe_allow_html=True)

    #######################################
    # STREAMLIT LAYOUT
    #######################################

top_left_column, top_right_column = st.columns((2, 1))
bottom_left_column, bottom_right_column = st.columns(2)

with top_left_column:
        column_1, column_2, column_3, column_4 = st.columns(4)

        with column_1:
            total_sales_value = filtered_df["Total"].sum()
            st.info(f"Total Sales: ${total_sales_value:,.2f}")
            st.success("✅ Good performance")
            plot_gauge(filtered_df["Rating"].mean(), "#0068C9", "", "Average Rating", 10)

        with column_2:
            total_quantity_sold_value = filtered_df["Quantity"].sum()
            st.info(f"Total Quantity Sold: {total_quantity_sold_value:,}")
            st.success("✅ Good performance")
            plot_gauge(filtered_df["Total"].mean(), "#FF8700", "$", "Average Total", 1000)

        with column_3:
            total_tax_value = filtered_df["Tax 5%"].sum()
            st.info(f"Total Tax: ${total_tax_value:,.2f}")
            st.success("✅ Good performance")
            plot_gauge(filtered_df["Unit price"].mean(), "#FF2B2B", "$", "Average Unit Price", 150)

        with column_4:
            total_invoice_count = filtered_df["Invoice ID"].nunique()
            st.info(f"Total Invoice Count: {total_invoice_count}")
            st.success("✅ Great performance!")
            plot_gauge(filtered_df["Quantity"].mean(), "#29B09D", "", "Average Quantity", 10)

with top_right_column:
    plot_top_right()

with bottom_left_column:
    plot_bottom_left()

with bottom_right_column:
    plot_bottom_right()

plot_time_series()   

def plot_knn_regression(df, x_column, y_column, k_neighbors=5):
    st.title(" K-NN Regression Analysis")

    # Filter numeric columns for X and Y
    numeric_columns = df.select_dtypes(include='number').columns
    x_options = st.selectbox("Select Feature Column for K-NN Regression", numeric_columns, index=0)
    y_options = st.selectbox("Select Target Column for K-NN Regression", numeric_columns, index=1)

    # Split data into features (X) and target variable (y)
    X = df[[x_options]]
    y = df[y_options]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the k-NN regression model
    knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)
    knn_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn_model.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    st.info(f"Mean Squared Error (MSE): {mse:.2f}")

    # Plot the original data and the regression line
    fig = px.scatter(x=X_test[x_options], y=y_test, labels={x_options: f"{x_options} (Test Set)", y_options: y_options})
    fig.add_trace(go.Scatter(x=X_test[x_options], y=y_pred, mode='lines', name='Regression Line'))
    st.plotly_chart(fig, use_container_width=True)


# Add a section in your Streamlit app to call the k-NN regression function
if not filtered_df.empty:
    plot_knn_regression(filtered_df, x_column=None, y_column=None, k_neighbors=5)