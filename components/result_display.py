import streamlit as st
import matplotlib.pyplot as plt
import io
import base64

from util.chart_render import create_donut_figure, create_bar_figure

def class_probs_to_base64_image(probs, class_names=None, title="Biểu đồ xác suất phân loại"):
    """
    probs: list or 1D np.array of probabilities (giá trị 0-1 hoặc 0-100)
    class_names: list of labels (same length as probs). If None, use indices.
    returns: data URI string "data:image/png;base64,...." ready to embed in <img>
    """
    import numpy as np

    probs = list(probs)
    # chuẩn hóa về phần trăm
    if len(probs) == 0:
        return None
    if max(probs) <= 1.0:
        vals = [p * 100.0 for p in probs]
    else:
        vals = [float(p) for p in probs]

    if class_names is None:
        class_names = [str(i) for i in range(len(vals))]

    # figure size động theo số lớp
    height = max(2.5, 0.5 * len(vals))
    fig, ax = plt.subplots(figsize=(max(6, len(vals) * 0.6), height))

    # highlight giá trị lớn nhất
    max_idx = int(np.argmax(vals))
    colors = ['#FFA500' if i == max_idx else '#2a9d8f' for i in range(len(vals))]

    y_pos = np.arange(len(class_names))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor='#222222', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()  # show highest on top if original order has highest first
    ax.set_xlim(0, 100)
    ax.set_xlabel("Tỉ lệ (%)")
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    # Annotate values at end of bars
    for bar, v in zip(bars, vals):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va='center', fontsize=9)

    plt.tight_layout()

    # Save to PNG in memory and trả về data URI
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"

# def show_topk_chart(topk_list, title="Top 5 dự đoán", figsize=(6, 3.5), return_base64=False):
#     """
#     Vẽ biểu đồ ngang (horizontal bar) cho top-5 dự đoán và hiển thị trên Streamlit.
#     topk_list: list of (class_name, prob) with prob in [0,1], length <= 5
#     return_base64: nếu True trả về data URI của ảnh PNG (chuỗi), ngược lại trả về None
#     """
#     import numpy as np

#     # chuẩn hóa và lấy tối đa 5 phần tử, sắp xếp giảm dần
#     topk = sorted(topk_list, key=lambda x: x[1], reverse=True)[:5]
#     if len(topk) == 0:
#         st.info("Không có dự đoán để hiển thị.")
#         return None

#     labels = [t[0] for t in topk]
#     probs = [t[1] * 100.0 for t in topk]  # chuyển về %
#     y_pos = np.arange(len(labels))

#     fig, ax = plt.subplots(figsize=figsize)
#     # highlight top1 bằng màu khác
#     colors = ['#FFA500' if i == 0 else '#2a9d8f' for i in range(len(labels))]
#     bars = ax.barh(y_pos, probs, color=colors, edgecolor='#222222')
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(labels)
#     ax.invert_yaxis()  # top hàng đầu lên trên
#     ax.set_xlim(0, 100)
#     ax.set_xlabel("Tỉ lệ (%)")
#     ax.set_title(title)
#     ax.grid(axis='x', linestyle='--', alpha=0.3)

#     # Annotate mỗi thanh
#     for i, (bar, p) in enumerate(zip(bars, probs)):
#         ax.text(p + 1, bar.get_y() + bar.get_height() / 2, f"{p:.2f}%", va='center', fontsize=9)

#     plt.tight_layout()
#     st.pyplot(fig)

#     # nếu cần trả về base64 (ví dụ để embed vào email/HTML)
#     if return_base64:
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', bbox_inches='tight')
#         plt.close(fig)
#         buf.seek(0)
#         b64 = base64.b64encode(buf.getvalue()).decode('ascii')
#         return f"data:image/png;base64,{b64}"

#     plt.close(fig)
#     return None

def show_topk_chart(topk_list, title="Top 5 dự đoán", figsize=(6, 3.5), return_base64=False):
    """
    Vẽ biểu đồ ngang cho top-5 dự đoán — sử dụng create_bar_figure (Plotly) để giữ style nhất quán.
    topk_list: list of (class_name, prob) with prob in [0,1] hoặc [0,100], length <= 5
    """
    # chuẩn hóa và lấy tối đa 5 phần tử, sắp xếp giảm dần
    topk = sorted(topk_list, key=lambda x: x[1], reverse=True)[:5]
    if len(topk) == 0:
        st.info("Không có dự đoán để hiển thị.")
        return None

    labels = [t[0] for t in topk]
    probs_raw = [t[1] for t in topk]
    # chuẩn về tỷ lệ 0-1 nếu cần
    probs = [p if p <= 1.0 else p / 100.0 for p in probs_raw]

    # predicted_label: highlight phần tử đầu (top1)
    predicted_label = labels[0]

    # Tạo Plotly figure dùng hàm create_bar_figure (hàm đã có trong file)
    fig = create_bar_figure(predicted_label, probs, labels, title=title)

    # Hiển thị bằng Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Nếu cần trả về base64 của ảnh PNG (không cần trong hầu hết trường hợp)
    if return_base64:
        import io, base64
        buf = io.BytesIO()
        fig.write_image(buf, format="png", scale=2)
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f"data:image/png;base64,{b64}"

    return None

def show_topk_predictions(topk_list):
    st.subheader("🔍 Top-5 dự đoán")
    for cls, prob in topk_list:
        st.write(f"- **{cls}**: {prob*100:.2f}%")

def dispay_bar_chart(predicted_label, probs, class_names, title="Biểu đồ cột tỉ lệ (%)"):
    """
    Hiển thị biểu đồ cột (bar) bằng Plotly.
    Trả về Plotly Figure.
    """
    fig = create_bar_figure(predicted_label, probs, class_names, title=title)
    st.plotly_chart(fig, use_container_width=True)
    return fig

def display_donut_chart(predicted_label, probs, class_names, title="Biểu đồ donut tỉ lệ (%)"):
    """
    Hiển thị biểu đồ donut bằng Plotly.
    Trả về Plotly Figure.
    """
    fig = create_donut_figure(predicted_label, probs, class_names, title=title)
    st.plotly_chart(fig, use_container_width=True)
    return fig

def display_prediction_charts(predicted_label, probs, class_names, title="Phân bố tỉ lệ đánh giá"):
    """
    Hiển thị 2 biểu đồ cạnh nhau: donut và bar bằng các hàm tách riêng.
    Trả về tuple (fig_donut, fig_bar).
    """
    fig_pie = create_donut_figure(predicted_label, probs, class_names, title="Biểu đồ donut tỉ lệ (%)")
    fig_bar = create_bar_figure(predicted_label, probs, class_names, title="Biểu đồ cột tỉ lệ (%)")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.plotly_chart(fig_bar, use_container_width=True)

    return fig_pie, fig_bar