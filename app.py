import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
from io import StringIO
import re

# Set page config
st.set_page_config(
    page_title="ITAM Graduates Analysis",
    page_icon="🎓",
    layout="wide",
    #initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00723F;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.8rem;
        color: #00723F;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #00723F;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    /* Card styling removed to eliminate boxes */
    .filter-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 25px;
    }
    .section-divider {
        border-top: 1px solid #ddd;
        margin: 40px 0;
    }
    /* Enable hover tooltips */
    .js-plotly-plot .plotly .hoverlayer {
        display: block !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("ITAM Graduates Analysis")
st.markdown("<h6 style='text-align: left; color: #666; font-style: italic; margin-bottom: 30px;'>Non-official dashboard made with publicly available information, # of graduates (titulados).</h6>", unsafe_allow_html=True)


# Function to extract first name from full name
def extract_first_name(full_name):
    try:
        # Split the name by spaces
        parts = full_name.split()
        
        # Common connectors in Spanish names that aren't last names
        connectors = ['DE', 'LA', 'LAS', 'LOS', 'DEL', 'Y']
        
        # Assume the first two parts are last names, unless they're connectors
        # If we have at least 3 parts (2 last names + 1 first name)
        if len(parts) >= 3:
            # Check if the third part is a connector
            if len(parts) > 3 and parts[2] in connectors:
                # If so, assume first name starts at position 4 or later
                first_name_start = 3
                # Check for additional connectors
                while first_name_start < len(parts) and parts[first_name_start] in connectors:
                    first_name_start += 1
                # If we've gone through all parts, default to the last part
                if first_name_start >= len(parts):
                    return parts[-1]
                return ' '.join(parts[first_name_start:])
            else:
                # Otherwise, assume first name starts at position 3
                return ' '.join(parts[2:])
        elif len(parts) == 2:
            # If only two parts, assume second part is the first name
            return parts[1]
        else:
            # If only one part, return it
            return parts[0]
    except:
        return ""

# Function to check if a person has multiple degrees
def find_people_with_multiple_degrees(df):
    # Count occurrences of each name
    name_counts = df['Student'].value_counts()
    
    # Filter to only include names that appear more than once
    multiple_degrees = name_counts[name_counts > 1]
    
    # Create a DataFrame with people who have multiple degrees
    if len(multiple_degrees) > 0:
        people_with_multiple = df[df['Student'].isin(multiple_degrees.index)]
        return people_with_multiple, multiple_degrees
    else:
        return pd.DataFrame(), pd.Series()

# Use default file
file_path = "250314_titulados-ITAM.csv"

# Load data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    
    # Add first name column
    data['First Name'] = data['Student'].apply(extract_first_name)
    
    # Handle NaN values in Year column before converting to integer
    if data['Year'].isna().any():
        # Count how many NaN values
        nan_count = data['Year'].isna().sum()
        # Fill NaN values with the median year
        median_year = data['Year'].median()
        data['Year'] = data['Year'].fillna(median_year)
    
    # Convert Year to integer to avoid decimal display
    data['Year'] = data['Year'].astype(int)
    
    # Clean up bachelor program names - remove "none" values and NaN values
    data = data[~data['Alternative name of bachelor'].astype(str).str.lower().isin(['none', 'nan'])]
    data = data[~data['Alternative name of bachelor'].isna()]
    
    return data

# Load the data
data = load_data(file_path)

# ===== FILTERS SECTION =====
with st.expander("Filters", expanded=False):
    # Year range filter
    years = sorted(data['Year'].unique())
    min_year = int(min(years))
    max_year = int(max(years))

    year_range = st.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    # Bachelor filter - changed to selectbox (dropdown)
    # Use Alternative name of bachelor for display
    # Convert all values to strings before sorting to avoid comparison between float and str
    bachelor_options = sorted([str(x) for x in data['Alternative name of bachelor'].unique()])
    selected_bachelors = st.selectbox(
        "Select Bachelor Programs",
        options=["All"] + bachelor_options
    )

    # If "All" is selected, use all bachelor options
    if selected_bachelors == "All":
        selected_bachelors = bachelor_options

# Apply filters
filtered_data = data[
    (data['Year'] >= year_range[0]) & 
    (data['Year'] <= year_range[1])
]

# Apply bachelor filter
if isinstance(selected_bachelors, list):
    # If it's a list (All option was selected)
    filtered_data = filtered_data[filtered_data['Alternative name of bachelor'].astype(str).isin(selected_bachelors)]
else:
    # If it's a single value
    filtered_data = filtered_data[filtered_data['Alternative name of bachelor'].astype(str) == selected_bachelors]

# ===== OVERVIEW SECTION =====
st.markdown("<h2 class='section-header'>Overview of ITAM Graduates</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Graduates", f"{len(filtered_data):,}")
    
with col2:
    # Count unique programs correctly using Alternative name of bachelor
    unique_programs = filtered_data['Alternative name of bachelor'].astype(str).nunique()
    st.metric("Number of Programs", f"{unique_programs:,}")
    st.markdown("<p style='font-size: 0.8rem; color: #666;'>* names of old programs have been consolidated for simplicity</p>", unsafe_allow_html=True)
    #   st.markdown("<p style='font-size: 0.8rem; color: #666;'>* 0 graduates from Artificial Intelligence Engineering program</p>", unsafe_allow_html=True)
    
with col3:
    st.metric("Years Covered", f"{min(filtered_data['Year'])} - {max(filtered_data['Year'])}")

# Calculate bachelor counts for use in multiple sections
bachelor_counts = filtered_data['Alternative name of bachelor'].astype(str).value_counts().reset_index()
bachelor_counts.columns = ['Bachelor', 'Count']

# Remove any remaining "none" or NaN values
bachelor_counts = bachelor_counts[~bachelor_counts['Bachelor'].str.lower().isin(['none', 'nan'])]

# Calculate percentage
total_graduates = bachelor_counts['Count'].sum()
bachelor_counts['Percentage'] = (bachelor_counts['Count'] / total_graduates * 100).round(2)

# ===== GRADUATION TRENDS SECTION =====
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Graduation Trends Over Time</h2>", unsafe_allow_html=True)

# Graduates per year with Plotly
year_counts = filtered_data.groupby('Year').size().reset_index(name='Count')

# Number of Graduates per Year chart
fig = px.line(
    year_counts,
    x='Year',
    y='Count',
    markers=True,
    line_shape='linear'
)
fig.update_traces(line=dict(color='#004A93', width=3), marker=dict(size=8))
fig.update_layout(
    title='Number of Graduates per Year',
    xaxis_title='Year',
    yaxis_title='Number of Graduates',
    height=400
)
# Add hover information
fig.update_traces(
    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Graduates: %{y:,}<extra></extra>'
)
# Enable hover information
fig.update_layout(hovermode='closest')
st.plotly_chart(fig, use_container_width=True)

# Year-over-year growth rate chart
if len(year_counts) > 1:
    year_counts['Growth'] = year_counts['Count'].pct_change() * 100
    year_counts['Growth'] = year_counts['Growth'].fillna(0)
    
    # Create a color list based on growth values
    colors = ['#FF4B4B' if x < 0 else '#00723F' for x in year_counts['Growth'][1:]]
    
    fig = px.bar(
        year_counts[1:],  # Skip first year as it has no growth rate
        x='Year',
        y='Growth',
        color_discrete_sequence=['#00723F']  # This will be overridden by the marker colors
    )
    
    # Update marker colors individually
    for i, color in enumerate(colors):
        fig.data[0].marker.color = colors
    
    fig.update_layout(
        title='Year-over-Year Growth Rate (%)',
        xaxis_title='Year',
        yaxis_title='Growth Rate (%)',
        height=400,
        hovermode='closest'
    )
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>Year: %{x}</b><br>Growth Rate: %{y:.2f}%<extra></extra>'
    )
    # Enable hover information
    fig.update_layout(hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough data to calculate growth rates.")

# ===== BACHELOR PROGRAM ANALYSIS SECTION =====
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Analysis by Bachelor Program</h2>", unsafe_allow_html=True)

# Bachelor program distribution with Plotly
fig = px.bar(
    bachelor_counts,
    x='Count',
    y='Bachelor',
    orientation='h',
    color_discrete_sequence=['#00723F']  # Use ITAM green for all bars
)
fig.update_layout(
    title='Number of Graduates by Bachelor Program',
    xaxis_title='Number of Graduates',
    yaxis_title='Bachelor Program',
    height=600,
    yaxis={'categoryorder': 'total ascending'},
    hovermode='closest'
)
# Add hover information with comma separators and percentage
fig.update_traces(
    hovertemplate='<b>%{y}</b><br>Graduates: %{x:,}<br>Percentage: %{customdata:.2f}%<extra></extra>',
    customdata=bachelor_counts['Percentage']
)
# Add text showing count and percentage
fig.update_traces(
    text=[f"{count:,} ({pct:.1f}%)" for count, pct in zip(bachelor_counts['Count'], bachelor_counts['Percentage'])],
    textposition='outside'
)
st.plotly_chart(fig, use_container_width=True)

# Add analysis of bachelor program distribution by year
st.markdown("<h3 class='subsection-header'>Bachelor Program Distribution by Year</h3>", unsafe_allow_html=True)
st.write("This analysis shows how the composition of bachelor programs has changed over time.")

# Let user select top N programs to display
top_n = st.slider("Select number of top programs to display", min_value=3, max_value=15, value=5)

# Get top N programs by total count
top_n_bachelors = bachelor_counts.sort_values('Count', ascending=False)['Bachelor'].head(top_n).tolist()

# Create a dataframe with counts by year and program, focusing on top N programs
yearly_counts = pd.crosstab(
    filtered_data['Year'], 
    filtered_data['Alternative name of bachelor'].astype(str)
)

# Filter to only include top N programs
yearly_counts_top = yearly_counts[top_n_bachelors]

# Calculate percentage for each year
yearly_distribution = yearly_counts_top.div(yearly_counts_top.sum(axis=1), axis=0) * 100

# Convert to long format for plotting
yearly_dist_long = yearly_distribution.reset_index().melt(
    id_vars=['Year'],
    var_name='Bachelor',
    value_name='Percentage'
)

# Create a stacked area chart
fig = px.area(
    yearly_dist_long,
    x='Year',
    y='Percentage',
    color='Bachelor',
    groupnorm='percent'  # Normalize to 100%
)

fig.update_layout(
    title=f'Relative Distribution of Top {top_n} Bachelor Programs by Year (%)',
    xaxis_title='Year',
    yaxis_title='Percentage',
    height=600,
    legend_title='Bachelor Program',
    hovermode='x unified'
)

# Add hover information with dynamic tooltips
fig.update_traces(
    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Percentage: %{y:.1f}%<extra></extra>'
)

# Enable hover information
fig.update_layout(hovermode='closest')

st.plotly_chart(fig, use_container_width=True)

# Also add a table showing the top programs for selected years
st.markdown("<h3 class='subsection-header'>Top Programs by Selected Years</h3>", unsafe_allow_html=True)

# Let user select specific years to compare
available_years = sorted(filtered_data['Year'].unique())

# Define the desired default years
desired_default_years = [1984, 1994, 2004, 2014, 2024]
# Filter to only include years that exist in the available data
default_years = [year for year in desired_default_years if year in available_years]
# If none of the desired years are available, fall back to min and max
if not default_years:
    default_years = [min(available_years), max(available_years)]

selected_years_for_comparison = st.multiselect(
    "Select years to compare",
    options=available_years,
    default=default_years
)

if selected_years_for_comparison:
    # Create a table with the top programs for each selected year
    comparison_data = []
    
    for year in selected_years_for_comparison:
        year_data = filtered_data[filtered_data['Year'] == year]
        program_counts = year_data['Alternative name of bachelor'].astype(str).value_counts()
        total = program_counts.sum()
        
        # Get top 5 programs
        top_programs = program_counts.head(5)
        
        for program, count in top_programs.items():
            comparison_data.append({
                'Year': year,
                'Program': program,
                'Count': count
            })
    
    # Convert to dataframe and display
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format the Count column with commas
    comparison_df['Count'] = comparison_df['Count'].apply(lambda x: f"{x:,}")
    
    # Pivot the table to show years as columns
    pivot_comparison = comparison_df.pivot(index='Program', columns='Year', values='Count')
    
    st.dataframe(pivot_comparison, use_container_width=True)

# ===== BACHELOR PROGRAM TRENDS SECTION =====
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Bachelor Program Trends</h2>", unsafe_allow_html=True)

# Get top 10 bachelor programs by total count
top_bachelors = bachelor_counts.sort_values('Count', ascending=False)['Bachelor'].head(10).tolist()

# Add dropdown for selecting specific programs
selected_programs = st.multiselect(
    "Select specific programs to display",
    options=["All"] + top_bachelors,
    default=["All"]
)

# Filter data based on selection
if "All" in selected_programs:
    display_bachelors = top_bachelors
else:
    display_bachelors = selected_programs

top_bachelors_data = filtered_data[filtered_data['Alternative name of bachelor'].astype(str).isin(display_bachelors)]

# Create pivot table
pivot_data = pd.crosstab(top_bachelors_data['Year'], top_bachelors_data['Alternative name of bachelor'].astype(str))

# Convert to long format for Plotly
pivot_long = pivot_data.reset_index().melt(
    id_vars=['Year'],
    var_name='Bachelor',
    value_name='Count'
)

fig = px.line(
    pivot_long,
    x='Year',
    y='Count',
    color='Bachelor',
    markers=True,
    line_shape='linear'
)
fig.update_layout(
    title='Graduates by Bachelor Program Over Time (Top 10 Programs)',
    xaxis_title='Year',
    yaxis_title='Number of Graduates',
    height=500,
    legend_title='Bachelor Program'
)
# Add hover information
fig.update_traces(
    hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Graduates: %{y:,}<extra></extra>'
)
# Enable hover information
fig.update_layout(hovermode='closest')
st.plotly_chart(fig, use_container_width=True)

# Heatmap of graduates by year and bachelor with Plotly
st.markdown("<h3 class='subsection-header'>Heatmap: Graduates by Year and Bachelor</h3>", unsafe_allow_html=True)

# Create pivot table for all programs
pivot_all = pd.crosstab(filtered_data['Year'], filtered_data['Alternative name of bachelor'].astype(str))

fig = px.imshow(
    pivot_all,
    labels=dict(x="Bachelor Program", y="Year", color="Graduates"),
    color_continuous_scale=[[0, 'white'], [0.01, '#f0f9f1'], [1, '#00723F']],  # Custom scale from white to ITAM green
    aspect="auto",
    zmin=0  # Start scale at 0
)
fig.update_layout(
    title='Heatmap of Graduates by Year and Bachelor Program',
    height=800,  # Make it bigger
    xaxis={'tickangle': 45}
)
# Add hover information with better formatting
fig.update_traces(
    hovertemplate='<b>%{x}</b><br>Year: %{y}<br>Graduates: %{z:,}<extra></extra>'
)
# Enable hover information
fig.update_layout(hovermode='closest')
st.plotly_chart(fig, use_container_width=True)

# ===== MULTIPLE DEGREES ANALYSIS SECTION =====
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Multiple Degrees Analysis</h2>", unsafe_allow_html=True)

# Find people with multiple degrees
multiple_degrees_data, name_counts = find_people_with_multiple_degrees(filtered_data)

if len(multiple_degrees_data) > 0:
    total_students = filtered_data['Student'].nunique()
    multiple_degree_students = len(name_counts)
    percentage = (multiple_degree_students / total_students) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("People with Multiple Degrees", f"{multiple_degree_students:,}")
    
    with col2:
        st.metric("Total Unique Students", f"{total_students:,}")
    
    with col3:
        st.metric("Percentage with Multiple Degrees", f"{percentage:.2f}%")
    
    # Distribution of number of degrees
    st.markdown("<h3 class='subsection-header'>Distribution of Number of Degrees per Person</h3>", unsafe_allow_html=True)
    
    # Count how many degrees each person has
    degree_counts = name_counts.reset_index()
    degree_counts.columns = ['Student', 'Number of Degrees']
    degree_distribution = degree_counts['Number of Degrees'].value_counts().sort_index().reset_index()
    degree_distribution.columns = ['Number of Degrees', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            degree_distribution,
            x='Number of Degrees',
            y='Count',
            color_discrete_sequence=['#00723F']  # Use ITAM green for all bars
        )
        fig.update_layout(
            title='Number of People by Degree Count',
            xaxis_title='Number of Degrees',
            yaxis_title='Count of People',
            height=400,
            hovermode='closest'
        )
        # Format x-axis to show integers (no decimals)
        fig.update_xaxes(tickformat='d')
        # Format y-axis to show comma separators
        fig.update_yaxes(tickformat=',')
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>Degrees: %{x}</b><br>People: %{y:,}<br>Percentage: %{customdata:.2f}%<extra></extra>',
            customdata=degree_distribution['Number of Degrees'].map(
                lambda x: (degree_distribution.loc[degree_distribution['Number of Degrees'] == x, 'Count'].iloc[0] / 
                          degree_distribution['Count'].astype(float).sum() * 100)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate percentages
        degree_distribution['Percentage'] = (degree_distribution['Count'] / degree_distribution['Count'].sum() * 100).round(2)
        degree_distribution['Percentage'] = degree_distribution['Percentage'].apply(lambda x: f"{x:.2f}%")
        degree_distribution['Count'] = degree_distribution['Count'].apply(lambda x: f"{x:,}")
        
        # Drop the 'Number of Degrees' column for display
        display_distribution = degree_distribution[['Count', 'Percentage']]
        
        st.write("Distribution of Degrees per Person")
        st.dataframe(display_distribution, use_container_width=True)
    
    # Most common degree combinations
    st.markdown("<h3 class='subsection-header'>Most Common Degree Combinations</h3>", unsafe_allow_html=True)
    
    combinations = []
    for name in name_counts.index:
        degrees = sorted(multiple_degrees_data[multiple_degrees_data['Student'] == name]['Alternative name of bachelor'].astype(str).tolist())
        combinations.append(' + '.join(degrees))
    
    combination_counter = Counter(combinations)
    
    # Visualize top combinations
    top_combinations = pd.DataFrame(combination_counter.most_common(10), columns=['Combination', 'Count'])
    
    # Calculate percentage
    top_combinations['Percentage'] = (top_combinations['Count'] / multiple_degree_students * 100).round(2)
    
    fig = px.bar(
        top_combinations,
        x='Count',
        y='Combination',
        orientation='h',
        color_discrete_sequence=['#00723F']  # Use ITAM green for all bars
    )
    fig.update_layout(
        title='Top 10 Most Common Degree Combinations',
        xaxis_title='Count',
        yaxis_title='Degree Combination',
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest'
    )
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Count: %{x:,}<br>Percentage: %{customdata:.2f}%<extra></extra>',
        customdata=top_combinations['Percentage']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show table with more combinations
    common_combinations = pd.DataFrame(combination_counter.most_common(20), columns=['Combination', 'Count'])
    
    # Add percentage column
    common_combinations['Percentage'] = (common_combinations['Count'] / multiple_degree_students * 100).round(2)
    common_combinations['Percentage'] = common_combinations['Percentage'].apply(lambda x: f"{x:.2f}%")
    common_combinations['Count'] = common_combinations['Count'].apply(lambda x: f"{x:,}")
    
    st.write("Top 20 Most Common Degree Combinations:")
    st.dataframe(common_combinations)
else:
    st.info("No people with multiple degrees found in the filtered data.")

# Add footer with attribution
st.markdown("---")
st.markdown(
    'Made by **[Valentin Mendez](https://personal-landing-page-vm.lovable.app/)** using publicly available info from '
    '**[ITAM](https://escolar1.rhon.itam.mx/titulacion/programas.asp)**'
) 