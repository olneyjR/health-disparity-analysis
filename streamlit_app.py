"""
Health Disparity Analysis Dashboard
Interactive visualization of health disparities across demographic and geographic dimensions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Health Disparity Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Content width limiter */
    .main .block-container {
        max-width: 1400px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Custom header - Changed to teal/cyan gradient */
    .custom-header {
        background: linear-gradient(135deg, #0891b2 0%, #0e7490 100%);
        padding: 1.5rem 2rem 2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .custom-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0891b2;
        line-height: 1.2;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        color: #10b981;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0891b2;
        display: inline-block;
    }
    
    .section-subheader {
        font-size: 1.1rem;
        font-weight: 500;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling - Changed to teal gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0891b2 0%, #0e7490 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: #0891b2;
        color: white !important;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #0891b2;
        margin: 1rem 0;
    }
    
    .info-box-title {
        font-weight: 600;
        color: #0c4a6e;
        margin-bottom: 0.5rem;
    }
    
    .info-box-text {
        color: #0e7490;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load all disparity analysis results"""
    metro_gaps = pd.read_csv('metro_nonmetro_gaps.csv')
    racial_gaps = pd.read_csv('racial_disparities.csv')
    income_gaps = pd.read_csv('income_disparities.csv')
    composite = pd.read_csv('composite_disparity_index.csv')
    
    return metro_gaps, racial_gaps, income_gaps, composite


def create_modern_choropleth(data, value_col, title, colorscale="Reds"):
    """Create modern choropleth map"""
    fig = px.choropleth(
        data,
        locations='State',
        locationmode="USA-states",
        color=value_col,
        scope="usa",
        color_continuous_scale=colorscale,
        title=title
    )
    
    fig.update_layout(
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            lakecolor='#e5e7eb',
            landcolor='#f9fafb',
            projection_type='albers usa',
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        title_font=dict(size=20, family="Inter", color="#1f2937"),
        font=dict(family="Inter"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_colorbar=dict(
            title=dict(font=dict(size=12)),
            thickness=15,
            len=0.7,
        )
    )
    
    return fig


def create_modern_bar(data, x, y, title, orientation='v'):
    """Create modern bar chart"""
    if orientation == 'h':
        fig = px.bar(data, x=x, y=y, title=title, orientation='h')
    else:
        fig = px.bar(data, x=x, y=y, title=title)
    
    fig.update_layout(
        title_font=dict(size=18, family="Inter", color="#1f2937"),
        font=dict(family="Inter"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#f3f4f6'),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False
    )
    
    fig.update_traces(
        marker_color='#667eea',
        marker_line_color='#764ba2',
        marker_line_width=1.5
    )
    
    return fig


def create_gauge_chart(value, title, max_value=100):
    """Create modern gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, max_value/3], 'color': '#d1fae5'},
                {'range': [max_value/3, 2*max_value/3], 'color': '#fed7aa'},
                {'range': [2*max_value/3, max_value], 'color': '#fecaca'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def main():
    # Custom header
    st.markdown("""
    <div class="custom-header">
        <h1>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 12px;">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"></path>
            </svg>
            US Health Disparity Analysis
        </h1>
        <p>Quantifying health inequities across demographic and geographic dimensions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    metro_gaps, racial_gaps, income_gaps, composite = load_data()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Select View",
            ["Overview", "Geographic Analysis", "Racial Disparities", "Income Disparities", "State Deep Dive"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard analyzes health disparities using data from America's Health Rankings 2025.
        
        **Data Coverage:**
        - 51 states + DC
        - 1,578 health measures
        - Multiple demographic dimensions
        """)
    
    # Route to appropriate page
    if page == "Overview":
        show_overview(metro_gaps, racial_gaps, income_gaps, composite)
    elif page == "Geographic Analysis":
        show_geographic_analysis(metro_gaps, composite)
    elif page == "Racial Disparities":
        show_racial_analysis(racial_gaps, composite)
    elif page == "Income Disparities":
        show_income_analysis(income_gaps, composite)
    elif page == "State Deep Dive":
        show_state_analysis(metro_gaps, racial_gaps, income_gaps, composite)


def show_overview(metro_gaps, racial_gaps, income_gaps, composite):
    """Overview page"""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#0891b2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                <polyline points="9 22 9 12 15 12 15 22"></polyline>
            </svg>
            <div class="metric-label">States Analyzed</div>
            <div class="metric-value">{len(composite)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
                <circle cx="12" cy="10" r="3"></circle>
                <path d="M12 21.7C17.3 17 20 13 20 10a8 8 0 1 0-16 0c0 3 2.7 6.9 8 11.7z"></path>
            </svg>
            <div class="metric-label">Metro-Nonmetro Gaps</div>
            <div class="metric-value">{len(metro_gaps):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
            <div class="metric-label">Racial Disparities</div>
            <div class="metric-value">{len(racial_gaps)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1d4ed8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 8px;">
                <line x1="12" y1="1" x2="12" y2="23"></line>
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
            </svg>
            <div class="metric-label">Income Disparities</div>
            <div class="metric-value">{len(income_gaps)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Composite Index Map
    st.markdown('<div class="section-header">Composite Disparity Index</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">States ranked by overall health inequity across all dimensions</div>', unsafe_allow_html=True)
    
    fig_composite = create_modern_choropleth(
        composite, 
        'Composite_Disparity_Index',
        'Overall Health Disparity by State',
        colorscale="RdYlGn_r"
    )
    st.plotly_chart(fig_composite, use_container_width=True)
    
    # Top/Bottom states in tabs
    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Highest Disparity", "Lowest Disparity"])
    
    with tab1:
        top_states = composite.nlargest(10, 'Composite_Disparity_Index')[
            ['State', 'Composite_Disparity_Index', 'Disparity_Rank', 'Avg_Metro_Gap', 'Avg_Racial_Gap', 'Avg_Income_Gap']
        ].copy()
        top_states.columns = ['State', 'Overall Index', 'Rank', 'Geographic Gap', 'Racial Gap', 'Income Gap']
        top_states['Overall Index'] = top_states['Overall Index'].round(2)
        top_states['Rank'] = top_states['Rank'].astype(int)
        
        # Apply heatmap styling to Overall Index column
        def highlight_index(val):
            if pd.isna(val):
                return ''
            color = f'background-color: rgba(239, 68, 68, {min(val/2, 0.8)})'
            return color
        
        styled_top = top_states.style.applymap(highlight_index, subset=['Overall Index'])
        st.dataframe(styled_top, use_container_width=True, hide_index=True)
    
    with tab2:
        bottom_states = composite.nsmallest(10, 'Composite_Disparity_Index')[
            ['State', 'Composite_Disparity_Index', 'Disparity_Rank', 'Avg_Metro_Gap', 'Avg_Racial_Gap', 'Avg_Income_Gap']
        ].copy()
        bottom_states.columns = ['State', 'Overall Index', 'Rank', 'Geographic Gap', 'Racial Gap', 'Income Gap']
        bottom_states['Overall Index'] = bottom_states['Overall Index'].round(2)
        bottom_states['Rank'] = bottom_states['Rank'].astype(int)
        
        # Apply heatmap styling to Overall Index column (green for low disparity)
        def highlight_index_low(val):
            if pd.isna(val):
                return ''
            # Get min and max from this subset for proper scaling
            min_val = bottom_states['Overall Index'].min()
            max_val = bottom_states['Overall Index'].max()
            # Normalize within this range
            if max_val - min_val > 0:
                normalized = (val - min_val) / (max_val - min_val)
            else:
                normalized = 0.5
            # Higher values in this group get darker green (they're still low overall)
            color = f'background-color: rgba(16, 185, 129, {0.2 + normalized * 0.6})'
            return color
        
        styled_bottom = bottom_states.style.applymap(highlight_index_low, subset=['Overall Index'])
        st.dataframe(styled_bottom, use_container_width=True, hide_index=True)
    
    # Key insights
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <div class="info-box-title">Key Findings</div>
        <div class="info-box-text">
        States in the South and Midwest show the highest composite disparity scores, driven primarily by 
        significant gaps in rural health outcomes and income-based disparities. Geographic location appears 
        to be a strong predictor of health inequity patterns.
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_geographic_analysis(metro_gaps, composite):
    """Geographic analysis page"""
    
    st.markdown('<div class="section-header">Geographic Disparity Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Metro vs Nonmetro health outcome differences</div>', unsafe_allow_html=True)
    
    metro_state_avg = metro_gaps.groupby('State')['Gap'].mean().reset_index()
    metro_state_avg.columns = ['State', 'Avg_Gap']
    
    fig_metro = create_modern_choropleth(
        metro_state_avg,
        'Avg_Gap',
        'Average Metro-Nonmetro Health Gap by State',
        colorscale="Oranges"
    )
    st.plotly_chart(fig_metro, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Measures with Largest Geographic Gaps")
        top_metro_measures = metro_gaps.groupby('Measure')['Gap'].mean().nlargest(10).reset_index()
        top_metro_measures.columns = ['Health Measure', 'Average Gap']
        
        # Orange bars to match map
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_metro_measures['Health Measure'],
            x=top_metro_measures['Average Gap'],
            orientation='h',
            marker=dict(
                color=top_metro_measures['Average Gap'],
                colorscale='Oranges',
                line=dict(color='#ea580c', width=1.5)
            ),
            text=top_metro_measures['Average Gap'].round(1),
            textposition='outside',
            textfont=dict(size=12, family='Inter')
        ))
        
        fig.update_layout(
            title='Top 10 Health Measures',
            title_font=dict(size=18, family="Inter", color="#1f2937"),
            font=dict(family="Inter"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True, 
                gridcolor='#f3f4f6',
                title='Average Gap'
            ),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=60, b=20),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### States with Largest Geographic Disparities")
        top_states_metro = metro_state_avg.nlargest(10, 'Avg_Gap')
        
        # Orange bars to match map
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_states_metro['State'],
            y=top_states_metro['Avg_Gap'],
            marker=dict(
                color=top_states_metro['Avg_Gap'],
                colorscale='Oranges',
                line=dict(color='#ea580c', width=1.5)
            ),
            text=top_states_metro['Avg_Gap'].round(1),
            textposition='outside',
            textfont=dict(size=12, family='Inter')
        ))
        
        fig.update_layout(
            title='Top 10 States by Geographic Gap',
            title_font=dict(size=18, family="Inter", color="#1f2937"),
            font=dict(family="Inter"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#f3f4f6',
                title='Average Gap'
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_racial_analysis(racial_gaps, composite):
    """Racial disparity analysis page"""
    
    st.markdown('<div class="section-header">Racial Health Disparities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Health outcome gaps across racial and ethnic groups</div>', unsafe_allow_html=True)
    
    racial_state_avg = racial_gaps.groupby('State')['Racial_Gap'].mean().reset_index()
    racial_state_avg.columns = ['State', 'Avg_Racial_Gap']
    
    fig_racial = create_modern_choropleth(
        racial_state_avg,
        'Avg_Racial_Gap',
        'Average Racial Health Disparity by State',
        colorscale="Purples"
    )
    st.plotly_chart(fig_racial, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Separate measures by scale
    st.markdown("#### Health Measures with Largest Racial Gaps")
    
    # Create tabs for different measure types
    tab1, tab2 = st.tabs(["Chronic Conditions & Behaviors", "Mortality"])
    
    with tab1:
        # Percentage-based measures (excluding Premature Death)
        other_measures = racial_gaps[racial_gaps['Measure'] != 'Premature Death']
        
        if len(other_measures) > 0:
            measure_avg = other_measures.groupby('Measure')['Racial_Gap'].mean().reset_index()
            measure_avg.columns = ['Health Measure', 'Average Racial Gap']
            measure_avg = measure_avg.sort_values('Average Racial Gap', ascending=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=measure_avg['Health Measure'],
                x=measure_avg['Average Racial Gap'],
                orientation='h',
                marker=dict(
                    color=measure_avg['Average Racial Gap'],
                    colorscale='Purples',
                    line=dict(color='#764ba2', width=1.5)
                ),
                text=measure_avg['Average Racial Gap'].round(1),
                textposition='outside',
                textfont=dict(size=12, family='Inter')
            ))
            
            fig.update_layout(
                title='Racial Disparity in Health Conditions (Percentage Points)',
                title_font=dict(size=18, family="Inter", color="#1f2937"),
                font=dict(family="Inter"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='#f3f4f6',
                    title='Average Gap (Percentage Points)'
                ),
                yaxis=dict(showgrid=False),
                margin=dict(l=20, r=20, t=60, b=20),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show insights
            max_gap = measure_avg.iloc[-1]
            st.markdown(f"""
            <div class="info-box">
                <div class="info-box-title">Key Insight</div>
                <div class="info-box-text">
                {max_gap['Health Measure']} shows the largest racial gap at {max_gap['Average Racial Gap']:.1f} percentage points, 
                indicating significant disparities in health outcomes across racial groups.
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        # Premature Death (different scale)
        premature_death_data = racial_gaps[racial_gaps['Measure'] == 'Premature Death']
        
        if len(premature_death_data) > 0:
            # Show state-by-state premature death gaps
            pd_by_state = premature_death_data.nlargest(15, 'Racial_Gap')[['State', 'Racial_Gap', 'Min_Value', 'Max_Value']]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=pd_by_state['State'],
                y=pd_by_state['Racial_Gap'],
                marker=dict(
                    color=pd_by_state['Racial_Gap'],
                    colorscale='Reds',
                    line=dict(color='#dc2626', width=1.5)
                ),
                text=pd_by_state['Racial_Gap'].apply(lambda x: f'{x:,.0f}'),
                textposition='outside',
                textfont=dict(size=11, family='Inter')
            ))
            
            fig.update_layout(
                title='Premature Death Racial Disparity by State (Deaths per 100k)',
                title_font=dict(size=18, family="Inter", color="#1f2937"),
                font=dict(family="Inter"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    showgrid=False,
                    title='State'
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='#f3f4f6',
                    title='Racial Gap (Deaths per 100k)'
                ),
                margin=dict(l=20, r=20, t=60, b=20),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary stats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_gap = premature_death_data['Racial_Gap'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Gap</div>
                    <div class="metric-value">{avg_gap:,.0f}</div>
                    <div class="metric-delta">deaths per 100k</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                max_gap = premature_death_data['Racial_Gap'].max()
                max_state = premature_death_data.loc[premature_death_data['Racial_Gap'].idxmax(), 'State']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Largest Gap</div>
                    <div class="metric-value">{max_gap:,.0f}</div>
                    <div class="metric-delta">{max_state}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                min_gap = premature_death_data['Racial_Gap'].min()
                min_state = premature_death_data.loc[premature_death_data['Racial_Gap'].idxmin(), 'State']
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Smallest Gap</div>
                    <div class="metric-value">{min_gap:,.0f}</div>
                    <div class="metric-delta">{min_state}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # States with largest overall disparities
    st.markdown("#### States with Largest Overall Racial Disparities")
    
    top_states_racial = racial_state_avg.nlargest(10, 'Avg_Racial_Gap')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_states_racial['State'],
        y=top_states_racial['Avg_Racial_Gap'],
        marker=dict(
            color=top_states_racial['Avg_Racial_Gap'],
            colorscale='Purples',
            line=dict(color='#7c3aed', width=1.5)
        ),
        text=top_states_racial['Avg_Racial_Gap'].round(1),
        textposition='outside',
        textfont=dict(size=12, family='Inter')
    ))
    
    fig.update_layout(
        title='Top 10 States by Average Racial Health Gap',
        title_font=dict(size=18, family="Inter", color="#1f2937"),
        font=dict(family="Inter"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#f3f4f6',
            title='Average Racial Gap'
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_income_analysis(income_gaps, composite):
    """Income disparity analysis page"""
    
    st.markdown('<div class="section-header">Income-Based Health Disparities</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Health outcome differences across income levels</div>', unsafe_allow_html=True)
    
    income_state_avg = income_gaps.groupby('State')['Income_Gap'].mean().reset_index()
    income_state_avg.columns = ['State', 'Avg_Income_Gap']
    
    fig_income = create_modern_choropleth(
        income_state_avg,
        'Avg_Income_Gap',
        'Average Income-Based Health Disparity by State',
        colorscale="Blues"
    )
    st.plotly_chart(fig_income, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Measures with Largest Income Gaps")
        top_income_measures = income_gaps.groupby('Measure')['Income_Gap'].mean().nlargest(10).reset_index()
        top_income_measures.columns = ['Health Measure', 'Average Income Gap']
        
        # Blue bars to match map
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_income_measures['Health Measure'],
            x=top_income_measures['Average Income Gap'],
            orientation='h',
            marker=dict(
                color=top_income_measures['Average Income Gap'],
                colorscale='Blues',
                line=dict(color='#1d4ed8', width=1.5)
            ),
            text=top_income_measures['Average Income Gap'].round(1),
            textposition='outside',
            textfont=dict(size=12, family='Inter')
        ))
        
        fig.update_layout(
            title='Top 10 Health Measures',
            title_font=dict(size=18, family="Inter", color="#1f2937"),
            font=dict(family="Inter"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True, 
                gridcolor='#f3f4f6',
                title='Average Income Gap'
            ),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=60, b=20),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### States with Largest Income Disparities")
        top_states_income = income_state_avg.nlargest(10, 'Avg_Income_Gap')
        
        # Blue bars to match map
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_states_income['State'],
            y=top_states_income['Avg_Income_Gap'],
            marker=dict(
                color=top_states_income['Avg_Income_Gap'],
                colorscale='Blues',
                line=dict(color='#1d4ed8', width=1.5)
            ),
            text=top_states_income['Avg_Income_Gap'].round(1),
            textposition='outside',
            textfont=dict(size=12, family='Inter')
        ))
        
        fig.update_layout(
            title='Top 10 States by Income Gap',
            title_font=dict(size=18, family="Inter", color="#1f2937"),
            font=dict(family="Inter"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True, 
                gridcolor='#f3f4f6',
                title='Average Income Gap'
            ),
            margin=dict(l=20, r=20, t=60, b=20),
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_state_analysis(metro_gaps, racial_gaps, income_gaps, composite):
    """State deep dive page"""
    
    st.markdown('<div class="section-header">State-Level Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Detailed disparity analysis for individual states</div>', unsafe_allow_html=True)
    
    selected_state = st.selectbox("Select a state", sorted(composite['State'].unique()))
    
    if selected_state:
        state_composite = composite[composite['State'] == selected_state].iloc[0]
        
        # State metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rank = int(state_composite['Disparity_Rank'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">National Rank</div>
                <div class="metric-value">#{rank}</div>
                <div class="metric-delta">of {len(composite)} states</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            index_val = state_composite['Composite_Disparity_Index']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Disparity Index</div>
                <div class="metric-value">{index_val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            percentile = (1 - (state_composite['Disparity_Rank'] / len(composite))) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Percentile</div>
                <div class="metric-value">{percentile:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if rank <= 10:
                severity = "High"
                color = "#ef4444"
            elif rank <= 25:
                severity = "Moderate"
                color = "#f59e0b"
            else:
                severity = "Low"
                color = "#10b981"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Disparity Level</div>
                <div class="metric-value" style="color: {color};">{severity}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Dimension breakdown
        st.markdown("#### Disparity Dimensions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if pd.notna(state_composite.get('Avg_Metro_Gap')):
                fig = create_gauge_chart(
                    state_composite['Avg_Metro_Gap'],
                    "Geographic Gap",
                    max_value=composite['Avg_Metro_Gap'].max()
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if pd.notna(state_composite.get('Avg_Racial_Gap')):
                fig = create_gauge_chart(
                    state_composite['Avg_Racial_Gap'],
                    "Racial Gap",
                    max_value=composite['Avg_Racial_Gap'].max()
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if pd.notna(state_composite.get('Avg_Income_Gap')):
                fig = create_gauge_chart(
                    state_composite['Avg_Income_Gap'],
                    "Income Gap",
                    max_value=composite['Avg_Income_Gap'].max()
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top disparities for this state
        st.markdown(f"#### Top Health Disparities in {selected_state}")
        
        tab1, tab2, tab3 = st.tabs(["Geographic", "Racial", "Income"])
        
        with tab1:
            state_metro = metro_gaps[metro_gaps['State'] == selected_state]
            if len(state_metro) > 0:
                top_metro = state_metro.nlargest(10, 'Gap')[['Measure', 'Gap', 'Metro_Value', 'Nonmetro_Value']]
                top_metro.columns = ['Health Measure', 'Gap', 'Metro Value', 'Nonmetro Value']
                st.dataframe(top_metro, use_container_width=True, hide_index=True)
        
        with tab2:
            state_racial = racial_gaps[racial_gaps['State'] == selected_state]
            if len(state_racial) > 0:
                top_racial = state_racial.nlargest(10, 'Racial_Gap')[['Measure', 'Racial_Gap', 'Min_Value', 'Max_Value']]
                top_racial.columns = ['Health Measure', 'Racial Gap', 'Min Value', 'Max Value']
                st.dataframe(top_racial, use_container_width=True, hide_index=True)
        
        with tab3:
            state_income = income_gaps[income_gaps['State'] == selected_state]
            if len(state_income) > 0:
                top_income = state_income.nlargest(10, 'Income_Gap')[['Measure', 'Income_Gap', 'Lowest_Income_Value', 'Highest_Income_Value']]
                top_income.columns = ['Health Measure', 'Income Gap', 'Lowest Income', 'Highest Income']
                st.dataframe(top_income, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
