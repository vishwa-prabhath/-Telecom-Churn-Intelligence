from pathlib import Path

# Use workspace-relative paths (not hard-coded user paths)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# ── Data loading ────────────────────────────────────────────
df = pd.read_parquet(DATA_DIR / 'scored_customers.parquet')
cohort_stats = pd.read_csv(DATA_DIR / 'cohort_stats.csv')
contract_stats = pd.read_csv(DATA_DIR / 'contract_stats.csv')
shap_importance = pd.read_csv(MODELS_DIR / 'shap_importance.csv')
with open(MODELS_DIR / 'metrics.json') as f:
    metrics = json.load(f)

# ── Colour palette ──────────────────────────────────────────
COLORS = {
    'primary':   '#3B82F6',
    'danger':    '#EF4444',
    'success':   '#22C55E',
    'warning':   '#F59E0B',
    'muted':     '#6B7280',
    'bg':        '#F9FAFB',
    'card':      '#FFFFFF',
    'border':    '#E5E7EB',
    'churn':     '#EF4444',
    'stay':      '#22C55E',
    'high_risk': '#EF4444',
    'med_risk':  '#F59E0B',
    'low_risk':  '#22C55E',
}

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', size=12, color='#374151'),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=[COLORS['primary'], COLORS['danger'], COLORS['success'],
              COLORS['warning'], '#8B5CF6', '#EC4899'],
)

# ── KPI helpers ──────────────────────────────────────────────
total_customers = len(df)
churn_customers = df['ChurnBinary'].sum()
churn_rate = df['ChurnBinary'].mean()
monthly_revenue = df['MonthlyCharges'].sum()
revenue_at_risk = df.loc[df['RiskTier'] == 'High', 'MonthlyCharges'].sum()
high_risk_count = (df['RiskTier'] == 'High').sum()

# ── App setup ────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap'
    ],
    suppress_callback_exceptions=True
)
app.title = 'Telecom Churn Intelligence Dashboard'

# ── Reusable components ──────────────────────────────────────
def kpi_card(title, value, subtitle, color='#3B82F6', icon='📊'):
    return html.Div([
        html.Div([
            html.Span(icon, style={'fontSize': '24px'}),
        ], style={'marginBottom': '8px'}),
        html.Div(value, style={
            'fontSize': '28px', 'fontWeight': '600', 'color': color, 'lineHeight': '1'
        }),
        html.Div(title, style={
            'fontSize': '13px', 'fontWeight': '500', 'color': '#374151', 'marginTop': '4px'
        }),
        html.Div(subtitle, style={'fontSize': '11px', 'color': '#9CA3AF', 'marginTop': '2px'}),
    ], style={
        'background': '#FFFFFF', 'borderRadius': '12px',
        'padding': '20px', 'border': '1px solid #E5E7EB',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.06)'
    })

def section_header(title, subtitle=''):
    return html.Div([
        html.H5(title, style={'fontWeight': '600', 'color': '#111827', 'marginBottom': '2px'}),
        html.P(subtitle, style={'color': '#6B7280', 'fontSize': '13px', 'margin': '0'})
    ], style={'marginBottom': '16px'})

# ── KPI Row ──────────────────────────────────────────────────
kpi_row = dbc.Row([
    dbc.Col(kpi_card('Total Customers', f'{total_customers:,}', 'Active base', '#3B82F6', '👥'), md=2),
    dbc.Col(kpi_card('Churned', f'{churn_customers:,}', f'{churn_rate:.1%} of base', '#EF4444', '📉'), md=2),
    dbc.Col(kpi_card('Monthly Revenue', f'${monthly_revenue:,.0f}', 'All customers', '#22C55E', '💰'), md=2),
    dbc.Col(kpi_card('Revenue at Risk', f'${revenue_at_risk:,.0f}', 'High-risk segment', '#F59E0B', '⚠️'), md=2),
    dbc.Col(kpi_card('High Risk', f'{high_risk_count:,}', 'Churn prob > 60%', '#EF4444', '🔴'), md=2),
    dbc.Col(kpi_card('Best Model AUC', f'{metrics["XGBoost"]["auc"]:.3f}', 'XGBoost ROC-AUC', '#8B5CF6', '🤖'), md=2),
], className='g-3 mb-4')

# ── TAB 1 — EDA ──────────────────────────────────────────────
def build_eda_tab():
    # Churn by contract
    fig_contract = px.bar(
        contract_stats, x='Contract', y='churn_rate',
        color='churn_rate', color_continuous_scale='RdYlGn_r',
        text=contract_stats['churn_rate'].map(lambda x: f'{x:.1%}'),
        title='Churn Rate by Contract Type'
    )
    fig_contract.update_traces(textposition='outside')
    fig_contract.update_layout(**CHART_LAYOUT, showlegend=False,
                                coloraxis_showscale=False)
    fig_contract.update_yaxes(tickformat='.0%', title='')
    fig_contract.update_xaxes(title='')

    # Churn by internet service
    inet_stats = df.groupby('InternetService')['ChurnBinary'].mean().reset_index()
    fig_inet = px.bar(inet_stats, x='InternetService', y='ChurnBinary',
        text=inet_stats['ChurnBinary'].map(lambda x: f'{x:.1%}'),
        color='ChurnBinary', color_continuous_scale='RdYlGn_r',
        title='Churn Rate by Internet Service')
    fig_inet.update_traces(textposition='outside')
    fig_inet.update_layout(**CHART_LAYOUT, showlegend=False, coloraxis_showscale=False)
    fig_inet.update_yaxes(tickformat='.0%', title='')
    fig_inet.update_xaxes(title='')

    # Monthly charges distribution
    fig_charges = go.Figure()
    fig_charges.add_trace(go.Histogram(
        x=df.loc[df['Churn'] == 'No', 'MonthlyCharges'],
        name='Retained', opacity=0.7,
        marker_color=COLORS['success'], nbinsx=30))
    fig_charges.add_trace(go.Histogram(
        x=df.loc[df['Churn'] == 'Yes', 'MonthlyCharges'],
        name='Churned', opacity=0.7,
        marker_color=COLORS['danger'], nbinsx=30))
    fig_charges.update_layout(**CHART_LAYOUT, barmode='overlay',
        title='Monthly Charges: Churned vs Retained',
        xaxis_title='Monthly Charges ($)', yaxis_title='Count')

    # Payment method
    pay_stats = df.groupby('PaymentMethod')['ChurnBinary'].mean().reset_index()
    pay_stats = pay_stats.sort_values('ChurnBinary', ascending=True)
    fig_pay = px.bar(pay_stats, x='ChurnBinary', y='PaymentMethod',
        orientation='h', text=pay_stats['ChurnBinary'].map(lambda x: f'{x:.1%}'),
        color='ChurnBinary', color_continuous_scale='RdYlGn_r',
        title='Churn Rate by Payment Method')
    fig_pay.update_traces(textposition='outside')
    fig_pay.update_layout(**CHART_LAYOUT, showlegend=False, coloraxis_showscale=False)
    fig_pay.update_xaxes(tickformat='.0%', title='')
    fig_pay.update_yaxes(title='')

    return dbc.Container([
        section_header('Exploratory Data Analysis',
                       'Understanding which customer segments drive churn'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_contract, config={'displayModeBar': False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_inet, config={'displayModeBar': False}), md=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_charges, config={'displayModeBar': False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_pay, config={'displayModeBar': False}), md=6),
        ]),
    ], fluid=True)

# ── TAB 2 — Cohort Analysis ───────────────────────────────────
def build_cohort_tab():
    fig_cohort_churn = px.bar(
        cohort_stats, x='TenureCohort', y='churn_rate',
        text=cohort_stats['churn_rate'].map(lambda x: f'{x:.1%}'),
        color='churn_rate', color_continuous_scale='RdYlGn_r',
        title='Churn Rate by Tenure Cohort'
    )
    fig_cohort_churn.update_traces(textposition='outside')
    fig_cohort_churn.update_layout(**CHART_LAYOUT, showlegend=False, coloraxis_showscale=False)
    fig_cohort_churn.update_yaxes(tickformat='.0%', title='')

    fig_cohort_rev = px.bar(
        cohort_stats, x='TenureCohort', y='revenue_at_risk',
        text=cohort_stats['revenue_at_risk'].map(lambda x: f'${x:,.0f}'),
        color='revenue_at_risk', color_continuous_scale='Oranges',
        title='Revenue at Risk by Tenure Cohort'
    )
    fig_cohort_rev.update_traces(textposition='outside')
    fig_cohort_rev.update_layout(**CHART_LAYOUT, showlegend=False, coloraxis_showscale=False)
    fig_cohort_rev.update_yaxes(tickformat='$,.0f', title='')

    # Scatter — tenure vs monthly charges coloured by churn prob
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x='tenure', y='MonthlyCharges',
        color='ChurnProbability',
        color_continuous_scale='RdYlGn_r',
        opacity=0.6, size_max=8,
        title='Tenure vs Monthly Charges (colour = churn probability)',
        labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'}
    )
    fig_scatter.update_layout(**CHART_LAYOUT)

    # Service count vs churn
    svc_stats = df.groupby('ServiceCount')['ChurnBinary'].agg(['mean', 'count']).reset_index()
    svc_stats.columns = ['ServiceCount', 'churn_rate', 'customers']
    fig_svc = px.line(svc_stats, x='ServiceCount', y='churn_rate',
        markers=True, title='Churn Rate by Number of Add-on Services',
        labels={'churn_rate': 'Churn Rate', 'ServiceCount': 'Number of Services'})
    fig_svc.update_traces(line_color=COLORS['danger'], marker_color=COLORS['danger'])
    fig_svc.update_layout(**CHART_LAYOUT)
    fig_svc.update_yaxes(tickformat='.0%')

    return dbc.Container([
        section_header('Cohort & Segment Analysis',
                       'How churn and revenue risk vary across customer groups'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_cohort_churn, config={'displayModeBar': False}), md=6),
            dbc.Col(dcc.Graph(figure=fig_cohort_rev, config={'displayModeBar': False}), md=6),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_scatter, config={'displayModeBar': False}), md=8),
            dbc.Col(dcc.Graph(figure=fig_svc, config={'displayModeBar': False}), md=4),
        ]),
    ], fluid=True)

# ── TAB 3 — Model Performance ────────────────────────────────
def build_model_tab():
    model_df = pd.DataFrame(metrics).T.reset_index()
    model_df.columns = ['Model', 'AUC', 'Avg Precision', 'Precision', 'Recall', 'F1', 'Accuracy']

    fig_compare = go.Figure()
    metrics_to_show = ['AUC', 'F1', 'Recall', 'Precision']
    colors = [COLORS['primary'], COLORS['danger'], COLORS['success'], COLORS['warning']]
    for i, (m, c) in enumerate(zip(metrics_to_show, colors)):
        fig_compare.add_trace(go.Bar(
            name=m, x=model_df['Model'],
            y=model_df[m].astype(float),
            marker_color=c, opacity=0.85,
            text=model_df[m].astype(float).map(lambda x: f'{x:.3f}'),
            textposition='outside'
        ))
    fig_compare.update_layout(**CHART_LAYOUT,
        title='Model Comparison — Key Metrics',
        barmode='group', yaxis=dict(range=[0, 1.1]))

    # Risk distribution
    risk_counts = df['RiskTier'].value_counts().reset_index()
    risk_counts.columns = ['Risk', 'Count']
    fig_risk = px.pie(risk_counts, names='Risk', values='Count',
        color='Risk',
        color_discrete_map={'High': COLORS['danger'], 'Medium': COLORS['warning'], 'Low': COLORS['success']},
        title='Customer Risk Distribution',
        hole=0.4)
    fig_risk.update_layout(**CHART_LAYOUT)

    # Churn probability histogram
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Histogram(
        x=df.loc[df['Churn'] == 'No', 'ChurnProbability'],
        name='Retained', opacity=0.7,
        marker_color=COLORS['success'], nbinsx=30))
    fig_prob.add_trace(go.Histogram(
        x=df.loc[df['Churn'] == 'Yes', 'ChurnProbability'],
        name='Churned', opacity=0.7,
        marker_color=COLORS['danger'], nbinsx=30))
    fig_prob.update_layout(**CHART_LAYOUT, barmode='overlay',
        title='Predicted Churn Probability Distribution',
        xaxis_title='Churn Probability', yaxis_title='Count')

    return dbc.Container([
        section_header('Model Performance',
                       'XGBoost, LightGBM, and Random Forest comparison'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_compare, config={'displayModeBar': False}), md=8),
            dbc.Col(dcc.Graph(figure=fig_risk, config={'displayModeBar': False}), md=4),
        ], className='mb-3'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_prob, config={'displayModeBar': False}), md=12),
        ]),
    ], fluid=True)

# ── TAB 4 — SHAP / Feature Importance ────────────────────────
def build_shap_tab():
    top_n = shap_importance.head(12)

    fig_shap = px.bar(
        top_n.sort_values('shap_importance'),
        x='shap_importance', y='feature',
        orientation='h',
        color='shap_importance', color_continuous_scale='Blues',
        title='Top 12 Features — Mean |SHAP| Value (XGBoost)',
        text=top_n.sort_values('shap_importance')['shap_importance'].map(lambda x: f'{x:.3f}')
    )
    fig_shap.update_traces(textposition='outside')
    fig_shap.update_layout(**CHART_LAYOUT, showlegend=False, coloraxis_showscale=False)
    fig_shap.update_xaxes(title='Mean |SHAP| Value')
    fig_shap.update_yaxes(title='')

    # What SHAP means — business interpretation
    interpretations = [
        ('Contract', 'Month-to-month customers are 3x more likely to churn. Moving them to annual contracts is the #1 retention lever.'),
        ('Tenure', 'New customers (< 12 months) churn at 2x the rate of long-term customers. Onboarding experience is critical.'),
        ('TechSupport', 'Customers without tech support churn significantly more. Bundling support reduces churn by ~12%.'),
        ('PaymentMethod', 'Electronic check users churn most. Auto-payment (bank transfer/credit card) correlates with ~15% lower churn.'),
        ('OnlineSecurity', 'Customers without online security are high-risk. Offering a free security trial is a proven retention tactic.'),
        ('MonthlyCharges', 'High charge customers ($80+) on monthly contracts are the highest-value churn risk. Prioritise for retention calls.'),
    ]

    insight_cards = dbc.Row([
        dbc.Col(html.Div([
            html.Div(feat, style={'fontWeight': '600', 'fontSize': '13px', 'color': '#1D4ED8', 'marginBottom': '4px'}),
            html.Div(desc, style={'fontSize': '12px', 'color': '#374151', 'lineHeight': '1.5'})
        ], style={
            'background': '#EFF6FF', 'borderRadius': '8px',
            'padding': '14px', 'border': '1px solid #BFDBFE', 'height': '100%'
        }), md=4, className='mb-3')
        for feat, desc in interpretations
    ])

    return dbc.Container([
        section_header('Feature Importance & SHAP Analysis',
                       'Why the model predicts churn — and what it means for the business'),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_shap, config={'displayModeBar': False}), md=12),
        ], className='mb-4'),
        html.H6('Business Interpretation of Top Features',
                style={'fontWeight': '600', 'color': '#111827', 'marginBottom': '12px'}),
        insight_cards,
    ], fluid=True)

# ── TAB 5 — Retention Recommendations ────────────────────────
def build_retention_tab():
    # High risk customers — top 20 by revenue
    high_risk = df[df['RiskTier'] == 'High'].nlargest(20, 'MonthlyCharges')[
        ['customerID', 'Contract', 'InternetService', 'tenure',
         'MonthlyCharges', 'ChurnProbability', 'ServiceCount']
    ].copy()
    high_risk['ChurnProbability'] = (high_risk['ChurnProbability'] * 100).round(1)
    high_risk['MonthlyCharges'] = high_risk['MonthlyCharges'].round(2)

    recommendations = [
        {
            'segment': 'Month-to-month, < 6 months tenure',
            'customers': len(df[(df['Contract'] == 'Month-to-month') & (df['tenure'] < 6)]),
            'avg_churn_prob': df[(df['Contract'] == 'Month-to-month') & (df['tenure'] < 6)]['ChurnProbability'].mean(),
            'action': 'Offer 20% discount on annual contract within 30 days of signup',
            'revenue_impact': df[(df['Contract'] == 'Month-to-month') & (df['tenure'] < 6)]['MonthlyCharges'].sum(),
            'priority': 'Critical',
            'color': '#FEE2E2'
        },
        {
            'segment': 'Fiber optic, no tech support, monthly contract',
            'customers': len(df[(df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No') & (df['Contract'] == 'Month-to-month')]),
            'avg_churn_prob': df[(df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No') & (df['Contract'] == 'Month-to-month')]['ChurnProbability'].mean(),
            'action': 'Offer 3-month free tech support bundle + loyalty reward',
            'revenue_impact': df[(df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No') & (df['Contract'] == 'Month-to-month')]['MonthlyCharges'].sum(),
            'priority': 'High',
            'color': '#FEF3C7'
        },
        {
            'segment': 'Electronic check payers, monthly contract',
            'customers': len(df[(df['PaymentMethod'] == 'Electronic check') & (df['Contract'] == 'Month-to-month')]),
            'avg_churn_prob': df[(df['PaymentMethod'] == 'Electronic check') & (df['Contract'] == 'Month-to-month')]['ChurnProbability'].mean(),
            'action': 'Incentivise switch to auto-payment with $5/month bill credit',
            'revenue_impact': df[(df['PaymentMethod'] == 'Electronic check') & (df['Contract'] == 'Month-to-month')]['MonthlyCharges'].sum(),
            'priority': 'High',
            'color': '#FEF3C7'
        },
        {
            'segment': 'Senior citizens on high monthly charges',
            'customers': len(df[(df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > 70)]),
            'avg_churn_prob': df[(df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > 70)]['ChurnProbability'].mean(),
            'action': 'Proactive outreach with senior plan — simplified billing & dedicated support line',
            'revenue_impact': df[(df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > 70)]['MonthlyCharges'].sum(),
            'priority': 'Medium',
            'color': '#ECFDF5'
        },
    ]

    rec_cards = []
    for r in recommendations:
        priority_colors = {'Critical': '#DC2626', 'High': '#D97706', 'Medium': '#059669'}
        rec_cards.append(dbc.Col(html.Div([
            html.Div([
                html.Span(r['segment'], style={'fontWeight': '600', 'fontSize': '13px', 'color': '#111827'}),
                html.Span(r['priority'], style={
                    'fontSize': '11px', 'fontWeight': '600',
                    'color': priority_colors[r['priority']],
                    'background': r['color'], 'padding': '2px 8px',
                    'borderRadius': '10px', 'marginLeft': '8px'
                })
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span(f'{r["customers"]:,} customers', style={'fontSize': '12px', 'color': '#6B7280', 'marginRight': '12px'}),
                html.Span(f'Avg churn prob: {r["avg_churn_prob"]:.0%}', style={'fontSize': '12px', 'color': COLORS["danger"], 'marginRight': '12px'}),
                html.Span(f'Revenue: ${r["revenue_impact"]:,.0f}/mo', style={'fontSize': '12px', 'color': COLORS["success"]}),
            ], style={'marginBottom': '10px'}),
            html.Div('Recommended action:', style={'fontSize': '11px', 'fontWeight': '600', 'color': '#374151', 'marginBottom': '4px'}),
            html.Div(r['action'], style={'fontSize': '12px', 'color': '#1D4ED8', 'lineHeight': '1.5'}),
        ], style={
            'background': r['color'], 'borderRadius': '10px',
            'padding': '16px', 'height': '100%',
            'border': '1px solid rgba(0,0,0,0.06)'
        }), md=6, className='mb-3'))

    # High value at-risk table
    table = dash_table.DataTable(
        data=high_risk.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in high_risk.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'fontSize': '12px', 'padding': '8px 12px', 'fontFamily': 'Inter, sans-serif'},
        style_header={'fontWeight': '600', 'background': '#F3F4F6', 'borderBottom': '2px solid #E5E7EB'},
        style_data_conditional=[
            {'if': {'filter_query': '{ChurnProbability} > 80', 'column_id': 'ChurnProbability'},
             'color': '#DC2626', 'fontWeight': '600'},
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#F9FAFB'}
        ],
        page_size=10, sort_action='native',
    )

    return dbc.Container([
        section_header('Retention Recommendations',
                       'Actionable strategies ranked by revenue impact'),
        dbc.Row(rec_cards, className='mb-4'),
        html.H6('Top 20 High-Value Customers at Risk — Prioritise for Outreach',
                style={'fontWeight': '600', 'color': '#111827', 'marginBottom': '12px'}),
        table,
    ], fluid=True)

# ── TAB 6 — Customer Lookup ───────────────────────────────────
def build_lookup_tab():
    return dbc.Container([
        section_header('Customer Risk Lookup',
                       'Search any customer to see their churn risk and profile'),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='customer-dropdown',
                    options=[{'label': cid, 'value': cid} for cid in df['customerID'].head(200)],
                    placeholder='Select a customer ID...',
                    style={'fontSize': '13px'}
                )
            ], md=4),
        ], className='mb-4'),
        html.Div(id='customer-detail')
    ], fluid=True)

# ── Layout ────────────────────────────────────────────────────
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div([
                html.H3('📡 Telecom Churn Intelligence', style={
                    'fontWeight': '700', 'color': '#111827', 'margin': '0', 'fontSize': '22px'
                }),
                html.P('XGBoost · LightGBM · SHAP · Plotly Dash',
                       style={'color': '#6B7280', 'fontSize': '12px', 'margin': '0'})
            ]),
            html.Div([
                html.Span('● Live', style={'color': '#22C55E', 'fontWeight': '600', 'fontSize': '13px'}),
                html.Span(' | Built by Vishwa Gunathilake',
                          style={'color': '#6B7280', 'fontSize': '12px', 'marginLeft': '8px'})
            ])
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'})
    ], style={
        'background': '#FFFFFF', 'padding': '16px 24px',
        'borderBottom': '1px solid #E5E7EB',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.05)'
    }),

    # Main content
    html.Div([
        # KPIs
        html.Div(kpi_row, style={'marginBottom': '8px'}),

        # Tabs
        dcc.Tabs(id='main-tabs', value='eda', children=[
            dcc.Tab(label='📊 EDA', value='eda',
                style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                selected_style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px',
                                'fontWeight': '600', 'borderTop': '3px solid #3B82F6'}),
            dcc.Tab(label='🔍 Cohort Analysis', value='cohort',
                style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                selected_style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px',
                                'fontWeight': '600', 'borderTop': '3px solid #3B82F6'}),
            dcc.Tab(label='🤖 Model Performance', value='model',
                style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                selected_style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px',
                                'fontWeight': '600', 'borderTop': '3px solid #3B82F6'}),
            dcc.Tab(label='🔬 SHAP Analysis', value='shap',
                style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                selected_style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px',
                                'fontWeight': '600', 'borderTop': '3px solid #3B82F6'}),
            dcc.Tab(label='💡 Retention Actions', value='retention',
                style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                selected_style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px',
                                'fontWeight': '600', 'borderTop': '3px solid #3B82F6'}),
            dcc.Tab(label='👤 Customer Lookup', value='lookup',
                style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px'},
                selected_style={'fontFamily': 'Inter, sans-serif', 'fontSize': '13px',
                                'fontWeight': '600', 'borderTop': '3px solid #3B82F6'}),
        ], style={'marginBottom': '20px'}),

        html.Div(id='tab-content'),
    ], style={'padding': '24px', 'background': '#F9FAFB', 'minHeight': 'calc(100vh - 72px)'})
], style={'fontFamily': 'Inter, sans-serif'})

# ── Callbacks ────────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab(tab):
    if tab == 'eda':       return build_eda_tab()
    if tab == 'cohort':    return build_cohort_tab()
    if tab == 'model':     return build_model_tab()
    if tab == 'shap':      return build_shap_tab()
    if tab == 'retention': return build_retention_tab()
    if tab == 'lookup':    return build_lookup_tab()

@app.callback(Output('customer-detail', 'children'), Input('customer-dropdown', 'value'))
def show_customer(cid):
    if not cid:
        return html.P('Select a customer above to see their profile.',
                      style={'color': '#9CA3AF', 'fontSize': '13px'})

    row = df[df['customerID'] == cid].iloc[0]
    risk_color = {'High': '#FEE2E2', 'Medium': '#FEF3C7', 'Low': '#ECFDF5'}[str(row['RiskTier'])]
    risk_text_color = {'High': '#DC2626', 'Medium': '#D97706', 'Low': '#059669'}[str(row['RiskTier'])]

    fields = [
        ('Contract', row['Contract']),
        ('Internet', row['InternetService']),
        ('Tenure', f"{row['tenure']} months"),
        ('Monthly Charges', f"${row['MonthlyCharges']:.2f}"),
        ('Payment Method', row['PaymentMethod']),
        ('Services', f"{int(row['ServiceCount'])} add-ons"),
        ('Tech Support', row['TechSupport']),
        ('Senior Citizen', 'Yes' if row['SeniorCitizen'] else 'No'),
    ]

    return dbc.Row([
        dbc.Col(html.Div([
            html.Div([
                html.Div(cid, style={'fontWeight': '700', 'fontSize': '18px', 'color': '#111827'}),
                html.Div(f"{row['ChurnProbability']:.0%} churn probability",
                         style={'fontSize': '28px', 'fontWeight': '700', 'color': risk_text_color}),
                html.Div(f"Risk tier: {row['RiskTier']}",
                         style={'fontSize': '14px', 'color': risk_text_color, 'marginTop': '4px'}),
            ], style={'marginBottom': '20px'}),
            html.Div([
                html.Div([
                    html.Span(k, style={'color': '#6B7280', 'fontSize': '12px', 'width': '130px', 'display': 'inline-block'}),
                    html.Span(v, style={'fontWeight': '500', 'fontSize': '13px', 'color': '#111827'})
                ], style={'padding': '6px 0', 'borderBottom': '1px solid #F3F4F6'})
                for k, v in fields
            ])
        ], style={
            'background': risk_color, 'borderRadius': '12px',
            'padding': '20px', 'border': f'1px solid {risk_text_color}40'
        }), md=5),
        dbc.Col(html.Div([
            html.H6('Recommended Action', style={'fontWeight': '600', 'color': '#111827', 'marginBottom': '12px'}),
            html.Div(
                'Immediately eligible for contract upgrade offer — target with annual plan + tech support bundle at 20% discount.' if str(row['RiskTier']) == 'High'
                else 'Monitor closely. Consider proactive outreach with loyalty reward before next billing cycle.' if str(row['RiskTier']) == 'Medium'
                else 'Low churn risk. Continue standard engagement. Consider upsell opportunities.',
                style={'fontSize': '13px', 'color': '#374151', 'lineHeight': '1.6',
                       'background': '#FFFFFF', 'borderRadius': '8px', 'padding': '14px',
                       'border': '1px solid #E5E7EB'}
            )
        ], style={'padding': '20px 0'}), md=7),
    ])

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DASHBOARD RUNNING → http://127.0.0.1:8050")
    print("="*60)
    app.run(debug=False, port=8050)
