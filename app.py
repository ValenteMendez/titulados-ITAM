import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
from io import StringIO
import re
import os
import string

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
    .section-header-compact {
        font-size: 1.8rem;
        color: #00723F;
        margin-top: 10px;
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
st.markdown("<h6 style='text-align: left; color: #666; font-style: italic; margin-bottom: 30px;'>Unofficial dashboard created using publicly available data on the number of graduates ('titulados'); includes only bachelor ('licenciatura') degrees and covers 1951 - 2024.</h6>", unsafe_allow_html=True)


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

# Use default files
file_path = "250314_titulados-ITAM-with-proxy.csv"
# gender_file_path is no longer needed since gender data is now in the main file

# Load data
@st.cache_data
def load_data(file_path):
    # Load the data with gender information already included
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
    
    # Fill any missing gender values
    data['Gender'] = data['Gender'].fillna('Unknown')
    
    return data

# Load the data
data = load_data(file_path)

# ===== FILTERS SECTION =====
# with st.expander("Filters", expanded=False):
#     # Year range filter
#     years = sorted(data['Year'].unique())
#     min_year = int(min(years))
#     max_year = int(max(years))

#     year_range = st.slider(
#         "Select Year Range",
#         min_value=min_year,
#         max_value=max_year,
#         value=(min_year, max_year)
#     )

#     # Bachelor filter - changed to selectbox (dropdown)
#     # Use Alternative name of bachelor for display
#     # Convert all values to strings before sorting to avoid comparison between float and str
#     bachelor_options = sorted([str(x) for x in data['Alternative name of bachelor'].unique()])
#     selected_bachelors = st.selectbox(
#         "Select Bachelor Programs",
#         options=["All"] + bachelor_options
#     )

#     # If "All" is selected, use all bachelor options
#     if selected_bachelors == "All":
#         selected_bachelors = bachelor_options

# Apply filters
# filtered_data = data[
#     (data['Year'] >= year_range[0]) & 
#     (data['Year'] <= year_range[1])
# ]

# # Apply bachelor filter
# if isinstance(selected_bachelors, list):
#     # If it's a list (All option was selected)
#     filtered_data = filtered_data[filtered_data['Alternative name of bachelor'].astype(str).isin(selected_bachelors)]
# else:
#     # If it's a single value
#     filtered_data = filtered_data[filtered_data['Alternative name of bachelor'].astype(str) == selected_bachelors]

# Since filters are commented out, use all data
filtered_data = data

# ===== OVERVIEW SECTION =====
st.markdown("<h2 class='section-header-compact'>Overview of ITAM Graduates</h2>", unsafe_allow_html=True)

# Calculate gender stats
gender_counts = filtered_data['Gender'].value_counts()
male_count = gender_counts.get('Male', 0)
female_count = gender_counts.get('Female', 0)
total_graduates = len(filtered_data)
male_percentage = male_count / total_graduates * 100 if total_graduates > 0 else 0
female_percentage = female_count / total_graduates * 100 if total_graduates > 0 else 0

# Create simple columns with only total graduates and gender distribution
col1, col2 = st.columns(2)

with col1:
    st.metric("Total graduates", f"{total_graduates:,}")
    st.markdown("""
    <div style='font-size: 0.7rem; color: #888; font-style: italic; text-align: left;'>
        * includes cases of people with multiple degrees; counted separately
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.metric("Gender distribution", f"M: {male_percentage:.1f}% | F: {female_percentage:.1f}%")
    st.markdown(f"""
    <div style='font-size: 0.8rem; color: #666; margin-top: 0; text-align: left;'>
        <span style='color: #004A93;'>Male: {male_count:,}</span> | <span style='color: #FF6600;'>Female: {female_count:,}</span>
    </div>
    <div style='font-size: 0.7rem; color: #888; font-style: italic; text-align: left;'>
        * using names as proxy for gender
    </div>
    """, unsafe_allow_html=True)

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

# Always show the main charts first (without gender breakdown)
# Original chart without gender breakdown
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
    # Original growth chart (total, without gender breakdown)
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

# Add option to show gender breakdown AFTER the main charts
show_gender_breakdown = st.checkbox("*show gender breakdown by year*", key="gender_trends_checkbox")

# Show gender analysis charts only if checkbox is selected
if show_gender_breakdown:
    st.markdown("<h3 class='subsection-header'>Gender Analysis in Graduation Trends</h3>", unsafe_allow_html=True)
    
    # Group by Year and Gender
    year_gender_counts = filtered_data.groupby(['Year', 'Gender']).size().reset_index(name='Count')
    
    # Number of Graduates per Year by Gender chart
    fig = px.line(
        year_gender_counts,
        x='Year',
        y='Count',
        color='Gender',
        markers=True,
        line_shape='linear',
        color_discrete_map={
            'Male': '#004A93',  # ITAM blue
            'Female': '#FF6600',  # Orange
            'Unknown': '#999999'  # Gray
        }
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(
        title='Number of Graduates per Year by Gender',
        xaxis_title='Year',
        yaxis_title='Number of Graduates',
        height=400,
        legend_title='Gender'
    )
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Graduates: %{y:,}<extra></extra>'
    )
    
    # Add a stacked bar version too
    fig_stacked = px.bar(
        year_gender_counts,
        x='Year',
        y='Count',
        color='Gender',
        barmode='stack',
        color_discrete_map={
            'Male': '#004A93',  # ITAM blue
            'Female': '#FF6600',  # Orange
            'Unknown': '#999999'  # Gray
        }
    )
    fig_stacked.update_layout(
        title='Number of Graduates per Year by Gender (Stacked)',
        xaxis_title='Year',
        yaxis_title='Number of Graduates',
        height=400,
        legend_title='Gender'
    )
    # Add hover information
    fig_stacked.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Graduates: %{y:,}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_stacked, use_container_width=True)
    
    # Also calculate gender percentage per year
    year_gender_pct = year_gender_counts.pivot_table(
        index='Year', 
        columns='Gender', 
        values='Count', 
        aggfunc='sum'
    ).fillna(0)
    
    # Calculate percentages
    year_gender_pct['Total'] = year_gender_pct.sum(axis=1)
    for gender in year_gender_pct.columns:
        if gender != 'Total':
            year_gender_pct[f'{gender} %'] = (year_gender_pct[gender] / year_gender_pct['Total'] * 100).round(1)
    
    # Create a percentage chart
    gender_pct_data = []
    for year in year_gender_pct.index:
        for gender in ['Male', 'Female']:
            if gender in year_gender_pct.columns:
                gender_pct_data.append({
                    'Year': year,
                    'Gender': gender,
                    'Percentage': year_gender_pct.at[year, f'{gender} %']
                })
    
    gender_pct_df = pd.DataFrame(gender_pct_data)
    
    # Only create this chart if we have both Male and Female data
    if 'Male' in gender_pct_df['Gender'].unique() and 'Female' in gender_pct_df['Gender'].unique():
        # Create a 100% stacked bar chart using a workaround since barnorm is not supported in px.bar
        # First create a regular stacked bar chart
        fig_pct = px.bar(
            gender_pct_df,
            x='Year',
            y='Percentage',
            color='Gender',
            barmode='stack',
            color_discrete_map={
                'Male': '#004A93',  # ITAM blue
                'Female': '#FF6600'  # Orange
            },
            text=None  # Remove text parameter that might be causing labels
        )
        
        # Since we already calculated percentages, we don't need to normalize here
        # Just ensure the y-axis shows percentages properly
        
        fig_pct.update_layout(
            title='Gender Distribution by Year (%)',
            xaxis_title='Year',
            yaxis_title='Percentage (%)',
            height=400,
            legend_title='Gender',
            # Set y axis properly for percentages
            yaxis=dict(
                range=[0, 100],
                title='Percentage (%)'
            )
        )
        
        # Explicitly remove text labels but keep hover information
        fig_pct.update_traces(
            texttemplate='',  # Empty string for no text
            textposition='none',  # Don't show text position
            hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Percentage: %{y:.1f}%<extra></extra>'
        )
        
        st.plotly_chart(fig_pct, use_container_width=True)
    
    # For each gender, calculate year-over-year growth
    if len(year_counts) > 1:
        genders_to_show = ['Male', 'Female']
        growth_data = []
        
        for gender in genders_to_show:
            if gender in year_gender_counts['Gender'].unique():
                gender_data = year_gender_counts[year_gender_counts['Gender'] == gender]
                gender_data = gender_data.sort_values('Year')
                
                if len(gender_data) > 1:
                    # Calculate year-over-year growth
                    gender_data['Growth'] = gender_data['Count'].pct_change() * 100
                    gender_data['Growth'] = gender_data['Growth'].fillna(0)
                    
                    # Add to growth data
                    for _, row in gender_data[1:].iterrows():  # Skip first year
                        growth_data.append({
                            'Year': row['Year'],
                            'Gender': gender,
                            'Growth': row['Growth']
                        })
        
        # Create growth dataframe
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            
            # Create bar chart
            fig = px.bar(
                growth_df,
                x='Year',
                y='Growth',
                color='Gender',
                barmode='group',
                color_discrete_map={
                    'Male': '#004A93',  # ITAM blue
                    'Female': '#FF6600'  # Orange
                }
            )
            
            fig.update_layout(
                title='Year-over-Year Growth Rate by Gender (%)',
                xaxis_title='Year',
                yaxis_title='Growth Rate (%)',
                height=400,
                hovermode='closest',
                legend_title='Gender'
            )
            
            # Add hover information
            fig.update_traces(
                hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Growth Rate: %{y:.2f}%<extra></extra>'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ===== BACHELOR PROGRAM ANALYSIS SECTION =====
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Analysis by Bachelor Program</h2>", unsafe_allow_html=True)

# Bachelor program distribution with Plotly (original)
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

# Option to show gender breakdown AFTER the main chart
show_gender_by_program = st.checkbox("*show gender breakdown by program*", key="gender_program_checkbox")

if show_gender_by_program:
    st.markdown("<h3 class='subsection-header'>Gender Analysis by Bachelor Program</h3>", unsafe_allow_html=True)
    
    # Calculate bachelor counts by gender
    bachelor_gender_counts = filtered_data.groupby(['Alternative name of bachelor', 'Gender']).size().reset_index()
    bachelor_gender_counts.columns = ['Bachelor', 'Gender', 'Count']
    
    # Calculate total for each bachelor to get percentages
    bachelor_totals = bachelor_gender_counts.groupby('Bachelor')['Count'].sum().reset_index()
    bachelor_totals.columns = ['Bachelor', 'Total']
    
    # Merge to get percentages
    bachelor_gender_counts = pd.merge(bachelor_gender_counts, bachelor_totals, on='Bachelor')
    bachelor_gender_counts['Percentage'] = (bachelor_gender_counts['Count'] / bachelor_gender_counts['Total'] * 100).round(2)
    
    # Sort by total count
    bachelor_order = bachelor_totals.sort_values('Total', ascending=True)['Bachelor'].tolist()
    bachelor_gender_counts['Bachelor'] = pd.Categorical(bachelor_gender_counts['Bachelor'], categories=bachelor_order, ordered=True)
    bachelor_gender_counts = bachelor_gender_counts.sort_values(['Bachelor', 'Gender'])
    
    # Filter to only include Male and Female (exclude Unknown)
    bachelor_gender_counts = bachelor_gender_counts[bachelor_gender_counts['Gender'].isin(['Male', 'Female'])]
    
    # Create stacked bar chart
    fig = px.bar(
        bachelor_gender_counts,
        x='Count',
        y='Bachelor',
        color='Gender',
        orientation='h',
        barmode='stack',
        color_discrete_map={
            'Male': '#004A93',  # ITAM blue
            'Female': '#FF6600'  # Orange
        },
        hover_data=['Percentage']
    )
    
    fig.update_layout(
        title='Number of Graduates by Bachelor Program and Gender',
        xaxis_title='Number of Graduates',
        yaxis_title='Bachelor Program',
        height=800,  # Increase height for better visibility of all programs
        yaxis={'categoryorder': 'array', 'categoryarray': bachelor_order},
        hovermode='closest',
        legend_title='Gender'
    )
    
    # Add hover information with comma separators and percentage
    fig.update_traces(
        hovertemplate='<b>%{y}</b> - %{fullData.name}<br>Graduates: %{x:,}<br>Percentage: %{customdata[0]:.2f}%<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create a percentage chart to show gender composition by program
    fig_pct = px.bar(
        bachelor_gender_counts,
        x='Percentage',
        y='Bachelor',
        color='Gender',
        orientation='h',
        color_discrete_map={
            'Male': '#004A93',  # ITAM blue
            'Female': '#FF6600'  # Orange
        }
    )
    
    fig_pct.update_layout(
        title='Gender Distribution by Bachelor Program (%)',
        xaxis_title='Percentage (%)',
        yaxis_title='Bachelor Program',
        height=800,  # Increase height for better visibility of all programs
        yaxis={'categoryorder': 'array', 'categoryarray': bachelor_order},
        hovermode='closest',
        legend_title='Gender'
    )
    
    # Add hover information
    fig_pct.update_traces(
        hovertemplate='<b>%{y}</b> - %{fullData.name}<br>Percentage: %{x:.2f}%<br>Count: %{customdata:,}<extra></extra>',
        customdata=bachelor_gender_counts['Count']
    )
    
    st.plotly_chart(fig_pct, use_container_width=True)

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

# Option to show gender distribution over time for a specific program
show_gender_trends = st.checkbox("*show gender trends for a specific program*", key="program_gender_trends_checkbox")

if show_gender_trends:
    st.markdown("<h4 class='subsection-header'>Gender Trends by Program</h4>", unsafe_allow_html=True)
    
    # Get list of all bachelor programs
    all_programs = sorted(filtered_data['Alternative name of bachelor'].astype(str).unique())
    
    # Let user select a program
    selected_program = st.selectbox(
        "Select a program to see gender trends over time",
        options=all_programs
    )
    
    # Filter data for selected program
    program_data = filtered_data[filtered_data['Alternative name of bachelor'].astype(str) == selected_program]
    
    # Group by year and gender
    program_gender_years = program_data.groupby(['Year', 'Gender']).size().reset_index(name='Count')
    
    # Calculate totals per year
    program_years_total = program_gender_years.groupby('Year')['Count'].sum().reset_index()
    program_years_total.columns = ['Year', 'Total']
    
    # Merge to get percentages
    program_gender_years = pd.merge(program_gender_years, program_years_total, on='Year')
    program_gender_years['Percentage'] = (program_gender_years['Count'] / program_gender_years['Total'] * 100).round(1)
    
    # Filter to only include Male and Female
    program_gender_years = program_gender_years[program_gender_years['Gender'].isin(['Male', 'Female'])]
    
    # Create line chart for absolute numbers
    fig_count = px.line(
        program_gender_years,
        x='Year',
        y='Count',
        color='Gender',
        markers=True,
        color_discrete_map={
            'Male': '#004A93',  # ITAM blue
            'Female': '#FF6600'  # Orange
        }
    )
    
    fig_count.update_layout(
        title=f'Number of Graduates by Gender for {selected_program}',
        xaxis_title='Year',
        yaxis_title='Number of Graduates',
        height=400,
        legend_title='Gender'
    )
    
    # Add hover information
    fig_count.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Graduates: %{y:,}<extra></extra>'
    )
    
    st.plotly_chart(fig_count, use_container_width=True)
    
    # Create area chart for percentages
    fig_pct = px.area(
        program_gender_years,
        x='Year',
        y='Percentage',
        color='Gender',
        color_discrete_map={
            'Male': '#004A93',  # ITAM blue
            'Female': '#FF6600'  # Orange
        }
    )
    
    fig_pct.update_layout(
        title=f'Gender Distribution for {selected_program} Over Time (%)',
        xaxis_title='Year',
        yaxis_title='Percentage (%)',
        height=400,
        legend_title='Gender'
    )
    
    # Add hover information
    fig_pct.update_traces(
        hovertemplate='<b>%{fullData.name}</b><br>Year: %{x}<br>Percentage: %{y:.1f}%<extra></extra>'
    )
    
    st.plotly_chart(fig_pct, use_container_width=True)

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
#st.markdown("<h3 class='subsection-header'>Heatmap: Graduates by Year and Bachelor</h3>", unsafe_allow_html=True)

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
    
    # Create 3 columns for metrics
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
    
    # No debug info needed
    
    degree_distribution = degree_counts['Number of Degrees'].value_counts().sort_index().reset_index()
    degree_distribution.columns = ['Number of Degrees', 'Count']
    
    # Calculate percentages
    degree_distribution['Percentage'] = (degree_distribution['Count'] / degree_distribution['Count'].sum() * 100).round(2)
    
    # Create labels for the pie chart
    degree_distribution['Label'] = degree_distribution['Number of Degrees'].apply(
        lambda x: f"{int(x)} Degrees: {degree_distribution.loc[degree_distribution['Number of Degrees'] == x, 'Count'].iloc[0]:,} ({degree_distribution.loc[degree_distribution['Number of Degrees'] == x, 'Percentage'].iloc[0]:.2f}%)"
    )
    
    # Create a donut pie chart
    fig = go.Figure(data=[go.Pie(
        labels=degree_distribution['Label'],
        values=degree_distribution['Count'],
        hole=0.6,  # Larger hole for better aesthetics
        textinfo='label',
        textposition='outside',
        # Use blue and orange color scheme with 3 degrees specifically as orange
        marker=dict(
            # Assign colors based on the number of degrees
            colors=[
                '#004A93' if x == 2 else  # Blue for 2 degrees
                '#FF6600' if x == 3 else  # Orange for 3 degrees
                '#0066CC' if x == 4 else  # Light blue for 4 degrees
                '#FF9933'                 # Light orange for any other number
                for x in degree_distribution['Number of Degrees']
            ],
            line=dict(color='#FFFFFF', width=2)
        ),
        pull=[0.05] * len(degree_distribution),  # Slightly pull all slices for better visibility
        rotation=90  # Start from the top
    )])
    
    fig.update_layout(
        #title='Distribution of People by Number of Degrees',
        height=500,
        width=700,  # Set a specific width to make it less wide
        margin=dict(l=50, r=50, t=80, b=50),  # Adjust margins for better centering
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        # Remove the middle annotation
        showlegend=True
    )
    
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
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

# ===== OTHER RANDOM STATS SECTION =====
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("<h2 class='section-header'>Other Random Stats</h2>", unsafe_allow_html=True)
#st.markdown("<p>Exploring interesting patterns in last names and their distribution across years and programs.</p>", unsafe_allow_html=True)

# Extract the last name (first word in the Student column)
filtered_data['Last_Name'] = filtered_data['Student'].str.split().str[0]

# ===== VISUALIZATION 3: BUBBLE CHART OF LAST NAME INITIALS BY YEAR =====
st.markdown("<h3 class='subsection-header'>Last Name Initial Letters by Year</h3>", unsafe_allow_html=True)

# Get the first letter of the last name and convert to uppercase
filtered_data['First_Letter'] = filtered_data['Last_Name'].str[0].str.upper()

# Filter out any non-alphabetic first letters
letter_data = filtered_data[filtered_data['First_Letter'].str.match('[A-Z]')]

# Create a cross-tabulation of years and first letters
crosstab_data = pd.crosstab(letter_data['First_Letter'], letter_data['Year'])

# Convert to long format for Plotly
bubble_data = []
for letter in crosstab_data.index:
    for year in crosstab_data.columns:
        count = crosstab_data.loc[letter, year]
        if count > 0:
            bubble_data.append({
                'Letter': letter,
                'Year': year,
                'Count': count
            })

bubble_df = pd.DataFrame(bubble_data)

# Create a list of grey shades from dark to light
# Map each letter to a shade of grey based on its position in the alphabet
alphabet = string.ascii_uppercase
grey_shades = []
for i, letter in enumerate(alphabet):
    # Calculate a hex value from dark grey (#333333) to light grey (#DDDDDD)
    # based on position in alphabet
    intensity = 51 + int((i / 25) * 170)  # Maps to range ~51-221
    hex_color = f'#{intensity:02x}{intensity:02x}{intensity:02x}'
    grey_shades.append(hex_color)

# Create bubble chart with Plotly
fig = px.scatter(
    bubble_df,
    x='Year',
    y='Letter',
    size='Count',
    color='Letter',
    color_discrete_map={letter: grey_shades[i] for i, letter in enumerate(alphabet)},
    hover_name='Letter',
    size_max=50,  # Maximum bubble size
)
fig.update_layout(
    title='Distribution of Last Name Initial Letters by Graduation Year',
    xaxis_title='Graduation Year',
    yaxis_title='First Letter of Last Name',
    height=700,
    yaxis={'categoryorder': 'category descending'},  # Sort A-Z from top to bottom
    hovermode='closest',
    showlegend=False,  # Remove the letter labels
    # Set x-axis range to match the actual data range
    xaxis=dict(
        range=[bubble_df['Year'].min() - 1, bubble_df['Year'].max() + 1]
    )
)
# Add hover information
fig.update_traces(
    hovertemplate='<b>Letter %{hovertext}</b><br>Year: %{x}<br>Count: %{marker.size:,}<extra></extra>'
)
st.plotly_chart(fig, use_container_width=True)

# Add a checkbox to show detailed data
show_letter_data = st.checkbox("*show detailed data for last name initials*", key="letter_data_checkbox")

if show_letter_data:
    # Create a heatmap of the letter distribution by year
    fig = px.imshow(
        crosstab_data,
        labels=dict(x="Year", y="First Letter", color="Count"),
        color_continuous_scale=[[0, 'white'], [0.01, '#f0f9f1'], [1, '#00723F']],  # Custom scale from white to ITAM green
        aspect="auto"
    )
    
    # Get all letters in alphabetical order for y-axis (A to Z)
    all_letters = sorted(crosstab_data.index.tolist())
    
    fig.update_layout(
        title='Heatmap of Last Name Initial Letters by Year',
        height=700,
        # Explicitly set the y-axis categories to have A at top and Z at bottom
        yaxis={'categoryorder': 'array', 'categoryarray': all_letters, 'autorange': 'reversed'},
        hovermode='closest'
    )
    # Add hover information
    fig.update_traces(
        hovertemplate='<b>Letter: %{y}</b><br>Year: %{x}<br>Count: %{z:,}<extra></extra>'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show a table with the most common letters overall
    letter_counts = letter_data['First_Letter'].value_counts().reset_index()
    letter_counts.columns = ['Letter', 'Count']
    letter_counts['Percentage'] = (letter_counts['Count'] / letter_counts['Count'].sum() * 100).round(2)
    
    # Add a Rank column starting from 1
    letter_counts.insert(0, 'Rank', range(1, len(letter_counts) + 1))
    
    st.write("Distribution of Last Name Initial Letters:")
    
    # Format the Count and Percentage columns
    letter_counts['Count'] = letter_counts['Count'].apply(lambda x: f"{x:,}")
    letter_counts['Percentage'] = letter_counts['Percentage'].apply(lambda x: f"{x:.2f}%")
    
    # Display the dataframe without the index by setting it as None
    letter_counts = letter_counts.set_index('Rank')
    st.dataframe(letter_counts, use_container_width=True)

# ===== VISUALIZATION 1: TOP 20 LAST NAMES =====
st.markdown("<h3 class='subsection-header'>Most Common Last Names</h3>", unsafe_allow_html=True)

# Count the frequency of each last name
last_name_counts = filtered_data['Last_Name'].value_counts().reset_index()
last_name_counts.columns = ['Last_Name', 'Count']

# Get the top 20 last names
top_20_last_names = last_name_counts.head(20)

# Create a horizontal bar chart
fig = px.bar(
    top_20_last_names,
    x='Count',
    y='Last_Name',
    orientation='h',
    color_discrete_sequence=['#00723F']  # Use ITAM green for all bars
)
fig.update_layout(
    title='Top 20 Most Common Last Names Among ITAM Graduates',
    xaxis_title='Number of Graduates',
    yaxis_title='Last Name',
    height=600,
    yaxis={'categoryorder': 'total ascending'},
    hovermode='closest'
)
# Add hover information
fig.update_traces(
    hovertemplate='<b>%{y}</b><br>Count: %{x:,}<extra></extra>'
)
# Add text showing count
fig.update_traces(
    text=top_20_last_names['Count'],
    textposition='outside'
)

# Calculate what percentage of total graduates these top 20 names represent
top_20_count = top_20_last_names['Count'].sum()
total_count = filtered_data.shape[0]
top_20_percentage = (top_20_count / total_count) * 100

st.plotly_chart(fig, use_container_width=True)

# Add a note below the chart
st.markdown(f"""
<div style='font-size: 0.9rem; color: #666; font-style: italic; text-align: left; margin-top: -15px; margin-bottom: 20px;'>
    These top 20 last names represent <b>{top_20_percentage:.2f}%</b> of all ITAM graduates. 
    'DE' is kept separately as an indicator of compound last names.
</div>
""", unsafe_allow_html=True)

# ===== VISUALIZATION 2: TOP 20 LAST NAMES VS BACHELOR DEGREES =====
#st.markdown("<h3 class='subsection-header'>Last Names Distribution Across Programs</h3>", unsafe_allow_html=True)

# Get the top 20 last names list in order of frequency
top_20_last_names_list = top_20_last_names['Last_Name'].tolist()

# Filter data for only the top 20 last names
top_20_data = filtered_data[filtered_data['Last_Name'].isin(top_20_last_names_list)]

# Create a cross-tabulation of last names and alternative bachelor names
heatmap_data = pd.crosstab(top_20_data['Last_Name'], top_20_data['Alternative name of bachelor'])

# Sort the index to match the order of top 20 last names by frequency
heatmap_data = heatmap_data.reindex(top_20_last_names_list)

# Convert to long format for Plotly
heatmap_long = heatmap_data.reset_index().melt(
    id_vars=['Last_Name'],
    var_name='Bachelor',
    value_name='Count'
)

# Create heatmap with Plotly
fig = px.density_heatmap(
    heatmap_long,
    x='Bachelor',
    y='Last_Name',
    z='Count',
    color_continuous_scale=[[0, 'white'], [0.01, '#f0f9f1'], [1, '#00723F']],  # Custom scale from white to ITAM green
)

# Reverse the order of last names to have most frequent at the top
reversed_last_names = top_20_last_names_list.copy()
reversed_last_names.reverse()

fig.update_layout(
    title='Distribution of Top 20 Last Names Across Bachelor Degrees',
    xaxis_title='Bachelor Program',
    yaxis_title='Last Name',
    height=700,
    xaxis={'tickangle': 45},
    # Set the y-axis to display last names in order of frequency (most frequent at top)
    yaxis={'categoryorder': 'array', 'categoryarray': reversed_last_names},
    hovermode='closest'
)
# Add hover information
fig.update_traces(
    hovertemplate='<b>%{y}</b> - <b>%{x}</b><br>Count: %{z:,}<extra></extra>'
)
st.plotly_chart(fig, use_container_width=True)

# Add footer with attribution
st.markdown("---")
st.markdown(
    'Made by **[Valentin Mendez](https://personal-landing-page-vm.lovable.app/)** using publicly available info from '
    '**[ITAM](https://escolar1.rhon.itam.mx/titulacion/programas.asp)**'
) 