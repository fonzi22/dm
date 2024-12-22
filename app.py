import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go

from utils.tab4 import plot_salary_distribution, plot_avg_salary, plot_avg_salary_by_tag, avg_salary_by_experience
from utils.job_recommendation import find_top_k_jobs
from utils.get_advice import get_job_advice
st.set_page_config(
    page_title="Job Market Insights", 
    page_icon="💼", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
    color: #2C3E50;
}
.highlight {
    background-color: #F1F8FF;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: #E8F4F8;
    color: #2C3E50;
    border-radius: 10px;
    font-weight: bold;
}
.stTabs [data-baseweb="tab"][data-selected="true"] {
    background-color: #3498DB;
    color: white;
}
                        
</style>
""", unsafe_allow_html=True)

def load_data():
    df = pd.read_csv('./data/preprocessed_P_unique_job.csv')
    return df

def create_job_distribution_plot(df, group_by, top_k=5):
    """Tạo biểu đồ phân phối công việc với Plotly"""
    if group_by != 'Job Category':
        job_category = st.selectbox(
            "Select Job Category for Tags Analysis", 
            ['All'] + list(df['Job Category'].unique())
        )
        if job_category == 'All': 
            df_filtered = df
        else: 
            df_filtered = df[df['Job Category'] == job_category]
        
        
        if group_by == 'tags':
            df_filtered['new_tags'] = df_filtered['new_tags'].str.split(';')
            df_exploded = df_filtered.explode('new_tags')
            counts = df_exploded['new_tags'].value_counts()
        else:
            counts = df_filtered[group_by].value_counts()
    else:
        counts = df[group_by].value_counts()

    # Chọn top k và nhóm "Khác"
    counts_top = counts.head(top_k)
    other_count = counts[top_k:].sum()
    counts_top["Others"] = other_count

    # Tạo biểu đồ Plotly
    fig = px.pie(
        values=counts_top.values, 
        names=counts_top.index, 
        title=f"Distribution of Jobs by {group_by}",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    return fig

def create_top_companies_plot(df, category=None, top_k=5):
    """Tạo biểu đồ các công ty hàng đầu theo danh mục (nếu có)"""
    if category:
        # Lọc dữ liệu theo danh mục
        df = df[df['Job Category'] == category]

    top_companies = df['Company Name'].value_counts().head(top_k)
    # print(top_companies)
    fig = px.bar(
        x=top_companies.index, 
        y=top_companies.values,
        title=f"Top {top_k} Companies by Job Postings" + (f" in {category}" if category else ""),
        labels={'x': 'Company Name', 'y': 'Number of Jobs'},
        color=top_companies.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        xaxis_title="Company Name", 
        yaxis_title="Number of Jobs"
    )
    return fig

def interactive_job_filter(df):
    
    # Tạo các cột để lọc
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        location = st.selectbox("Location", 
            ['All'] + list(df['Location'].unique()))
    
    with col2:
        experience = st.selectbox("Experience Level", 
            ['All'] + list(df['Experience'].unique()))
    
    with col3:
        category = st.selectbox("Job Category", 
            ['All'] + list(df['Job Category'].unique()))
    
    with col4:
        df['new_tags'] = df['new_tags'].fillna('').astype(str)
        tag = st.selectbox("Tags", 
            ['All'] + list(set(tag for tags in df['new_tags'].str.split(';') for tag in tags)))
    
    # Lọc dữ liệu
    filtered_df = df.copy()
    if location != 'All':
        filtered_df = filtered_df[filtered_df['Location'] == location]
    if experience != 'All':
        filtered_df = filtered_df[filtered_df['Experience'] == experience]
    if category != 'All':
        filtered_df = filtered_df[filtered_df['Job Category'] == category]
    if tag != 'All':
        filtered_df = filtered_df[filtered_df['new_tags'].str.contains(tag, na=False)]
    
    # Hiển thị kết quả
    st.write(f"📊 Found {len(filtered_df)} jobs")
    if not filtered_df.empty:
        st.dataframe(filtered_df)
    else:
        st.warning("No jobs found matching the selected filters.")
def truncate_text(text, max_length=200):
    """Truncate long text with ellipsis"""
    max_length = max(max_length, len(str(text)))
    return text[:max_length] + '...' if text and len(str(text)) > max_length else text
@st.cache_data
def load_salary_models(df, model="Linear Regression", transformation="None"):
    """Load the trained salary prediction models"""
    model = "ln" if model == "Linear Regression" else "rf"
    if transformation == "None":
        transformation = "none"
    elif transformation == "Log":
        transformation = "log"
    else:
        transformation = "boxcox"
    min_salary_model = joblib.load(f"./models/{model}_min_salary_{transformation}.pkl")
    max_salary_model = joblib.load(f"./models/{model}_max_salary_{transformation}.pkl")
    if model == "ln" and transformation == "none":
        one_hot_encoder = joblib.load("./models/ohe.pkl")
        multilabel_binarizer = joblib.load("./models/mlb.pkl")
    else:
        one_hot_encoder = joblib.load("./models/ohe_new.pkl")
        multilabel_binarizer = joblib.load("./models/mlb_new.pkl")
    min_pt, max_pt = None, None
    if transformation == "boxcox":
        min_pt = joblib.load("./models/min_pt.pkl")
        max_pt = joblib.load("./models/max_pt.pkl")
    salary_df = df[(df['min_salary'] > 0) & (df['max_salary'] > 0)].drop(['Job ID', 'Posted Date', 'Job Position'], axis=1)
    return salary_df, min_salary_model, max_salary_model, one_hot_encoder, multilabel_binarizer, min_pt, max_pt

def get_tags_for_category(df, category):
    """Get available tags for a specific job category"""
    category_df = df[df['Job Category'] == category]
    all_tags = set()
    for tags_str in category_df['new_tags'].dropna():
        tags = [tag.strip() for tag in tags_str.split(';')]
        all_tags.update(tags)
    return sorted(all_tags)

def prepare_input_data(input_features, one_hot_encoder, multilabel_binarizer):
    """Prepare input data for prediction"""
    # Create a DataFrame with the input features
    input_df = pd.DataFrame([input_features])
    categorical_features = input_df[['Location', 'Job Category']]
    categorical_encoded = one_hot_encoder.transform(categorical_features)
    categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=one_hot_encoder.get_feature_names_out(), index=input_df.index)
    tags_encoded = multilabel_binarizer.transform(input_df['new_tags'])
    tags_df = pd.DataFrame(tags_encoded, columns=multilabel_binarizer.classes_)
    return pd.concat([input_df['Experience'], categorical_encoded_df, tags_df], axis=1)

def format_salary_prediction(prediction):
    """Format salary prediction to VND with proper formatting"""
    return f"{prediction:,.1f} Million VND"


def main():
    # st.set_option('deprecation.showPyplotGlobalUse', False)

    st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🌐 Job Market Explorer</h1>", unsafe_allow_html=True)
    
    # Load dữ liệu
    df = load_data()
    # Load IT data
    df_IT = pd.read_csv('./data/IT_jobs_translated.csv')

    # Tạo tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Job Distribution", "🏢 Company Insights", "🔍 Job Search", "💸 Explore Salary", "🎯 Job Recommendation", "💰 Salary Prediction"])
    
    with tab1:
        st.subheader("📊 Job Distribution Analysis")

        col1, col2 = st.columns(2)
        with col1:
            group_by = st.selectbox("Group Jobs By", 
                ["Job Category", "Location", "Experience", "tags"])
        with col2:
            top_k = st.slider("Top Categories", 3, 50, 10)
        
        fig = create_job_distribution_plot(df, group_by, top_k)
        fig.update_layout(width=1200, height=800) 
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("🏢 Top Companies Insights")

        category = st.selectbox("Select a job category to view the top company", ['All'] + list(df['Job Category'].unique()))
        category = None if category == 'All' else category
        top_k = st.slider("Number of Top Companies", 3, 10, 5)
        fig_companies = create_top_companies_plot(df, category, top_k)
        st.plotly_chart(fig_companies, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        # st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.subheader("🔍 Interactive Job Filter")
        interactive_job_filter(df)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.subheader("💸 Explore Salary")

        sub_tab1 = st.selectbox("What do you want to explore", ["Salary distribution", "Average salary", "Average by YoE", "Average on tags"])
    
        if sub_tab1 == "Salary distribution":
            category = st.selectbox("Select a job category to view the salary distribution", list(df['Job Category'].unique()))
            fig = plot_salary_distribution(df, category=category)
            st.pyplot(fig)
        elif sub_tab1 == "Average salary":
            col1, col2, col3 = st.columns(3)
            with col1:
                feature = st.selectbox("Select a feature to view the average salary.", ["Job Category", "Location", "Experience"])
            with col2:
                sort_by = st.selectbox("Sort by.", ["Salary", "Feature"])
                sort_by = True if sort_by == 'Salary' else False
            with col3:
                topk = st.slider("Top k popular", 3, 10, 5)
            fig = plot_avg_salary(df, top_k=topk, category=feature, by_max_salary=sort_by)
            st.pyplot(fig)
        elif sub_tab1 == "Average by YoE":
            job_cat = st.selectbox("Select a job category to view the salary distribution", ['All'] + list(df['Job Category'].unique()))
            job_cat = None if job_cat == 'All' else job_cat
            fig = avg_salary_by_experience(df, job_cat)
            st.pyplot(fig)
        elif sub_tab1 ==  "Average on tags":
            col1, col2 = st.columns(2)
            with col1:
                job_cat = st.selectbox("Select a job category to view the salary distribution", ['All'] + list(df['Job Category'].unique()))
                job_cat = None if job_cat == 'All' else job_cat
            with col2:
                topk = st.slider("Select top k to show", 3, 10, 5)
            fig = plot_avg_salary_by_tag(df,top_k=topk, category=job_cat)
            st.pyplot(fig)


    with tab5:
        st.subheader("🎯 Job Recommendation")
        
        with st.form("job_search_form"):
            col1, col2 = st.columns(2)
            with col1:
                description_query = st.text_input(
                    "Job Description", 
                    value="triển khai các mô hình NLP, RAG, nghiên cứu phát triển sản phẩm mới để tích hợp vào hệ thống phần mềm", 
                    placeholder="Enter the job description you are looking for",
                )
                k = st.slider(
                    "Number of Results", 
                    min_value=1, 
                    max_value=10, 
                    value=5,
                    help="Select how many job recommendations to display"
                )
            with col2:
                experiences_query = st.text_input(
                    "Experiences", 
                    value="", 
                    placeholder="Enter the experiences and jobs you have done"
                )
                benefits_query = st.text_input(
                    "Job Benefits", 
                    value="", 
                    placeholder="Enter the benefits you want to receive or what you expect from the company"
                )

            search_button = st.form_submit_button("🔎Find Jobs", use_container_width=True)

        if search_button:

            st.session_state["job_results"] = None
            st.session_state["advice_buttons"] = {}

            result = find_top_k_jobs(df_IT, description_query, experiences_query, benefits_query, k)
            if not result.empty:
                st.session_state["job_results"] = result
            else:
                st.warning("No matching jobs found. Try different search terms.")

        result = st.session_state.get("job_results", None)
        if result is not None:
            # Use Streamlit's expander for cleaner UI
            st.markdown("### 📋 Top Job Matches")
            
            for idx, row in result.iterrows():
                
                # Create a unique key for each job's advice button
                advice_key = f"advice_button_{idx}"
                
                # Create columns for job title and advice button
                title_col, advice_col = st.columns([3, 2])
                
                with title_col:
                    with st.expander(row['Title']): 
                        col_left, col_right = st.columns([1, 2])
                        
                        with col_left:
                            st.write("**Company**", row['Company'])
                            st.write("**Salary**", row['Salary'])
                            st.write("**Experience**", row['Experience'])
                        
                        with col_right:
                            st.write("**Description:**")
                            st.write(truncate_text(row['Description']))
                            
                            st.write("**Requirements:**")
                            st.write(truncate_text(row['Requirements']))
                            
                            st.write("**Benefits:**")
                            st.write(truncate_text(row['Benefits']))
                # Initialize session state for advice buttons
                if "advice_buttons" not in st.session_state:
                    st.session_state["advice_buttons"] = {}

                advice_key = f"advice_button_{idx}"

                # Initialize advice state for this button if not already done
                if advice_key not in st.session_state["advice_buttons"]:
                    st.session_state["advice_buttons"][advice_key] = None

                with advice_col:
                    advise_button = st.button("🤝 Get Advice", key=advice_key, use_container_width=True)

                    if advise_button:
                        try:
                            # Generate advice and save it in session state
                            advice = get_job_advice(row['Requirements'], experiences_query)
                            st.session_state["advice_buttons"][advice_key] = advice
                        except Exception as e:
                            st.error(f"An error occurred while generating advice: {str(e)}")
                    
                    if st.session_state["advice_buttons"].get(advice_key):
                        with st.expander('Advice for you'):
                            st.markdown(st.session_state["advice_buttons"][advice_key], unsafe_allow_html=True)

    with tab6:
        st.subheader("💰 Salary Prediction")
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])
        with col2:
            transformation = st.selectbox("Transformation method", ["None", "Log", "Boxcox"])
        # Load the models
        salary_df, min_salary_model, max_salary_model, one_hot_encoder, multilabel_binarizer, min_pt, max_pt = load_salary_models(df, model, transformation)
        
        if None in (min_salary_model, max_salary_model, one_hot_encoder, multilabel_binarizer):
            st.error("Unable to load salary prediction models. Please check if model files exist.")
            return
        
        # Create input form
        # with st.form("salary_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Job Category selection outside the form
            job_category = st.selectbox(
                "Job Category",
                options=salary_df['Job Category'].unique(),
            )
            
            # Get and display tags for selected category
            available_tags = get_tags_for_category(salary_df, job_category)
            tags = st.multiselect(
                "Skills/Technologies",
                options=available_tags,
            )
                
        with col2:
            location = st.selectbox(
                "Location",
                options=salary_df['Location'].unique()
            )
            
            experience = st.selectbox(
                "Experience Level",
                options=sorted(salary_df['Experience'].unique())
            )
        
        predict_button = st.button("Predict Salary Range", use_container_width=True)
        
        if predict_button:
            try:
                # Prepare input features
                input_features = {
                    'Location': location,
                    'Job Category': job_category,
                    'Experience': experience,
                    'new_tags': tags  # Using number of tags as a feature
                }
                
                # Prepare input data
                input_df = prepare_input_data(input_features, one_hot_encoder, multilabel_binarizer)
                
                # Make predictions
                min_salary_pred = min_salary_model.predict(input_df)[0]
                max_salary_pred = max_salary_model.predict(input_df)[0]
                if transformation == "Log":
                    min_salary_pred = np.exp(min_salary_pred)
                    max_salary_pred = np.exp(max_salary_pred)
                elif transformation == "Boxcox":
                    min_salary_pred = min_pt.inverse_transform(np.array([[min_salary_pred]]))[0][0]
                    max_salary_pred = max_pt.inverse_transform(np.array([[max_salary_pred]]))[0][0]
                if min_salary_pred > max_salary_pred:
                    min_salary_pred = max_salary_pred
                
                # Display predictions
                st.markdown("---")
                st.markdown("### Predicted Salary Range")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Minimum Salary",
                        value=format_salary_prediction(min_salary_pred)
                    )
                
                with col2:
                    st.metric(
                        label="Maximum Salary",
                        value=format_salary_prediction(max_salary_pred)
                    )
                
                # Display additional insights
                st.markdown("---")
                st.markdown("### 📊 Salary Insights")
                
                # Calculate average actual salaries for comparison
                similar_jobs = df[
                    (df['Job Category'] == job_category) &
                    (df['Location'] == location) &
                    (df['Experience'] == experience)
                ]
                similar_jobs.drop(['Job ID', 'Posted Date', 'Job Position'], axis=1, inplace=True)
                similar_jobs['min_salary'] = similar_jobs['min_salary'].apply(lambda x: x if x != -1 else "Thỏa thuận")
                similar_jobs['max_salary'] = similar_jobs['max_salary'].apply(lambda x: x if x != -1 else "Thỏa thuận")
                similar_jobs = similar_jobs[similar_jobs['new_tags'].apply(lambda x: all(tag in str(x).split(";") for tag in tags))]
                if not similar_jobs.empty:
                    st.write("#### Similar Jobs Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Average Min Salary",
                            format_salary_prediction(similar_jobs[(similar_jobs['min_salary'] != "Thỏa thuận") & (similar_jobs['max_salary'] != "Thỏa thuận")]['min_salary'].mean())
                        )
                    with col2:
                        st.metric(
                            "Average Max Salary",
                            format_salary_prediction(similar_jobs[(similar_jobs['min_salary'] != "Thỏa thuận") & (similar_jobs['max_salary'] != "Thỏa thuận")]['max_salary'].mean())
                        )
                    with col3:
                        st.metric(
                            "Number of Similar Jobs",
                            len(similar_jobs)
                        )
                    st.dataframe(similar_jobs)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        

if __name__ == '__main__':
    main()