import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def display_percentage_bar(label, prob, class_names):
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, prob, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Tỉ lệ dự đoán (%)')
    ax.set_title('Tỉ lệ dự đoán các lớp')

    # Highlight the predicted class
    for i, v in enumerate(prob):
        if class_names[i] == label:
            ax.barh(i, v, align='center', color='orange')

def create_donut_figure(predicted_label, probs, class_names, title="Biểu đồ donut tỉ lệ (%)"):
    """
    Trả về Plotly Figure cho biểu đồ donut (pie with hole).
    probs: list (giá trị 0-1 hoặc 0-100)
    """
    probs = list(probs)
    if len(probs) == 0:
        return go.Figure()
    pct = [p * 100.0 for p in probs] if max(probs) <= 1.0 else [float(p) for p in probs]

    base_colors = px.colors.qualitative.Plotly
    colors = [base_colors[i % len(base_colors)] for i in range(len(class_names))]
    highlight_color = "#FFA500"
    colors_pie = [highlight_color if name == predicted_label else colors[i] for i, name in enumerate(class_names)]

    fig = go.Figure(
        go.Pie(
            labels=class_names,
            values=pct,
            hole=0.5,
            marker=dict(colors=colors_pie, line=dict(color='#222222', width=1)),
            textposition='outside',
            textfont=dict(size=16, color='#ffffff'),
            hovertemplate='%{label}: %{value:.2f}%<extra></extra>',
            textinfo='percent'
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20), xanchor='center'),
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='v', x=1.02, y=0.5, font=dict(size=24))
    )
    return fig

def create_bar_figure(predicted_label, probs, class_names, title="Biểu đồ cột tỉ lệ (%)"):
    """
    Trả về Plotly Figure cho biểu đồ cột (bar) — chuyển sang thanh ngang và hiển thị % ngoài thanh.
    """
    probs = list(probs)
    if len(probs) == 0:
        return go.Figure()
    # pct = [p * 100.0 for p in probs] if max(probs) <= 1.0 else [float(p) for p in probs]
    probs_pct = [p * 100 for p in probs]
    # max_p = max(probs_pct)

    base_colors = px.colors.qualitative.Plotly
    colors = [base_colors[i % len(base_colors)] for i in range(len(class_names))]
    highlight_color = "#FFA500"
    colors_bar = [highlight_color if name == predicted_label else colors[i] for i, name in enumerate(class_names)]

    # Tính giới hạn trục X và thêm một padding âm nhỏ để kéo điểm 0 ra khỏi mép trái
    max_x = max(100, max(probs_pct) * 1.1 + 5)

    fig = go.Figure(
        go.Bar(
            x=probs_pct,
            y=class_names,
            orientation='h',
            marker_color=colors_bar,
            text=[f"{v:.2f}%" for v in probs_pct],
            textposition='outside',
            textfont=dict(size=16, color='#ffffff'),
            hovertemplate='%{y}: %{x:.2f}%<extra></extra>'
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20),
            xanchor='center'),
        xaxis=dict(
            title='Tỉ lệ (%)',
            tickfont=dict(size=18),
            range=[0, max_x],
            automargin=True
        ),
        yaxis=dict(
            autorange="reversed", # giữ thứ tự giống bar dọc trước đó
            tickfont=dict(size=18),
            automargin=True
        ),  
        margin=dict(l=40, r=40, t=40, b=20)
    )
    return fig

def display_prediction_charts(predicted_label, probs, class_names, title="Phân bố tỉ lệ đánh giá"):
    """
    Hiển thị 2 biểu đồ cạnh nhau: donut và bar bằng các hàm tách riêng.
    Trả về tuple (fig_donut, fig_bar).
    """
    fig_pie = create_donut_figure(predicted_label, probs, class_names, title="Biểu đồ donut tỉ lệ (%)")
    fig_bar = create_bar_figure(predicted_label, probs, class_names, title="Biểu đồ cột tỉ lệ (%)")

    row1, row2 = st.rows([1, 1])
    with row1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with row2:
        st.plotly_chart(fig_pie, use_container_width=True)

    return fig_pie, fig_bar