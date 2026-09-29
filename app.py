# -*- coding: utf-8 -*-
"""
Web App Streamlit – Tiểu luận Big Data trong TMĐT
Phân tích hành vi mua sắm, khai phá giỏ hàng & dự đoán mua lại – Instacart (mẫu 10% khách hàng)
Học viên: Đào Thị Hồng Vân (C25611268) – GVHD: TS. Nguyễn Thôn Dã
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------------
# CẤU HÌNH STREAMLIT
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Instacart Big Data Dashboard",
    page_icon="🛒",
    layout="wide"
)


# ------------------------------------------------------------------
# ĐƯỜNG DẪN DỮ LIỆU
# ------------------------------------------------------------------
# Các file dữ liệu nằm cùng thư mục với app.py trên GitHub.
# KHÔNG dùng /data vì repository hiện tại không có thư mục data.

DATA = Path(__file__).resolve().parent


# ------------------------------------------------------------------
# CẤU HÌNH
# ------------------------------------------------------------------

SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948"
]

BLUE_SEQ = [
    "#f4f8fe",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#1c5cab",
    "#104281",
    "#0d366b"
]

DOW_VI = [
    "Chủ Nhật",
    "Thứ 2",
    "Thứ 3",
    "Thứ 4",
    "Thứ 5",
    "Thứ 6",
    "Thứ 7"
]

FEAT_VI = {
    "up_orders": "Số lần KH đã mua SP",
    "up_order_rate": "Tỷ lệ đơn có SP",
    "up_orders_since_last": "Số đơn từ lần mua SP gần nhất",
    "up_rate_since_first": "Tỷ lệ mua SP từ lần đầu",
    "up_avg_cart_pos": "Vị trí TB trong giỏ (KH–SP)",
    "u_orders": "Số đơn của KH",
    "u_avg_gap": "Chu kỳ mua TB của KH",
    "u_reorder_ratio": "Tỷ lệ mua lại của KH",
    "u_avg_basket": "Giỏ hàng TB của KH",
    "u_distinct": "Số SP khác nhau KH đã mua",
    "p_orders": "Độ phổ biến SP",
    "p_reorder_rate": "Tỷ lệ mua lại của SP",
    "p_avg_cart_pos": "Vị trí TB trong giỏ của SP",
    "p_users": "Số KH đã mua SP",
    "t_days_since_prior": "Số ngày từ đơn trước"
}


# ------------------------------------------------------------------
# STYLE CHART
# ------------------------------------------------------------------

def style(fig, h=380):
    fig.update_layout(
        height=h,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.7)"
        ),
        hoverlabel=dict(font_size=13)
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#c3c2b7"
    )

    fig.update_yaxes(
        gridcolor="rgba(137,135,129,0.25)",
        zeroline=False
    )

    return fig


# ------------------------------------------------------------------
# DỮ LIỆU
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Đang nạp dữ liệu mẫu 10%…")
def load():

    orders = pd.read_parquet(
        DATA / "orders_sample.parquet"
    )

    lines = pd.read_parquet(
        DATA / "order_products_sample.parquet"
    )

    prods = pd.read_parquet(
        DATA / "products.parquet"
    )

    lines["eval_set"] = lines["eval_set"].astype("category")

    for c in [
        "product_name",
        "aisle",
        "department",
        "department_vi"
    ]:
        prods[c] = prods[c].astype("category")

    fact = (
        lines
        .merge(
            orders[
                [
                    "order_id",
                    "user_id",
                    "order_number",
                    "order_dow",
                    "order_hour_of_day",
                    "days_since_prior_order"
                ]
            ],
            on="order_id"
        )
        .merge(
            prods[
                [
                    "product_id",
                    "product_name",
                    "aisle",
                    "department_vi"
                ]
            ],
            on="product_id"
        )
    )

    return orders, lines, prods, fact


# ------------------------------------------------------------------
# DỮ LIỆU PHỤ
# ------------------------------------------------------------------

@st.cache_data
def load_small():

    rd = lambda f: pd.read_csv(DATA / f)

    js = lambda f: json.load(
        open(
            DATA / f,
            encoding="utf-8"
        )
    )

    return {
        "rules_p": rd("rules_product_full.csv"),
        "rules_a": rd("rules_aisle_full.csv"),
        "seg": rd("segment_profile_full.csv"),
        "metrics": rd("model_metrics_full.csv"),
        "imp": rd("feature_importance_gbt.csv"),
        "comp": rd("sample_vs_full.csv"),

        "kpi": js("kpi_full.json"),
        "lr": js("lr_model.json"),

        "pfeat": pd.read_parquet(
            DATA / "product_features_full.parquet"
        ),

        "useg": pd.read_parquet(
            DATA / "user_segments_sample.parquet"
        )
    }


# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

orders, lines, prods, fact = load()
S = load_small()


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------

with st.sidebar:

    st.markdown("### 🛒 Instacart Big Data")

    st.caption(
        "Tiểu luận môn **Nghiên cứu Dữ liệu lớn trong TMĐT**  \n"
        "Học viên: **Đào Thị Hồng Vân** (C25611268)  \n"
        "GVHD: **TS. Nguyễn Thôn Dã**"
    )

    st.divider()

    st.markdown("**Bộ lọc (áp dụng cho RQ1, RQ2)**")

    dept_opts = sorted(
        prods["department_vi"]
        .dropna()
        .unique()
        .tolist()
    )

    f_dept = st.multiselect(
        "Ngành hàng",
        dept_opts,
        placeholder="Tất cả ngành hàng"
    )

    f_dow = st.multiselect(
        "Ngày trong tuần",
        DOW_VI,
        placeholder="Tất cả các ngày"
    )

    st.divider()

    st.caption(
        f"Dữ liệu: mẫu 10% khách hàng "
        f"(lấy mẫu theo khách hàng, băm xxhash64) — "
        f"{orders.user_id.nunique():,} khách, "
        f"{len(orders):,} đơn, "
        f"{len(lines):,} dòng sản phẩm. "
        "Luật kết hợp, phân khúc và mô hình dự đoán "
        "được huấn luyện trên **100% dữ liệu** bằng Apache Spark."
    )


# ------------------------------------------------------------------
# FILTER
# ------------------------------------------------------------------

fx = fact

if f_dept:
    fx = fx[
        fx["department_vi"].isin(f_dept)
    ]

if f_dow:
    fx = fx[
        fx["order_dow"].isin(
            [DOW_VI.index(d) for d in f_dow]
        )
    ]

ox = (
    orders[
        orders["order_id"].isin(
            fx["order_id"].unique()
        )
    ]
    if (f_dept or f_dow)
    else orders
)


# ------------------------------------------------------------------
# TITLE
# ------------------------------------------------------------------

st.title(
    "🛒 Phân tích hành vi mua sắm & khai phá giỏ hàng – Instacart"
)

st.caption(
    "Dashboard tương tác minh họa kết quả pipeline Apache Spark "
    "trên 3,4 triệu đơn hàng TMĐT tạp hóa trực tuyến."
)


tabs = st.tabs(
    [
        "📊 Tổng quan",
        "⏰ RQ1 Thời gian",
        "🥑 RQ2 Sản phẩm",
        "🧺 RQ3 Giỏ hàng",
        "🎯 RQ4 Phân khúc",
        "🤖 RQ5 Dự đoán mua lại"
    ]
)


# ------------------------------------------------------------------
# TỔNG QUAN
# ------------------------------------------------------------------

with tabs[0]:

    k = S["kpi"]

    c = st.columns(5)

    bs = (
        lines
        .groupby("order_id")
        .size()
        .mean()
    )

    tiles = [
        (
            "Khách hàng (mẫu)",
            f"{orders.user_id.nunique():,}",
            f"{k['so_khach_hang']:,}"
        ),
        (
            "Đơn hàng (mẫu)",
            f"{len(orders):,}",
            f"{k['so_don_hang']:,}"
        ),
        (
            "Dòng sản phẩm (mẫu)",
            f"{len(lines):,}",
            f"{k['so_dong_san_pham']:,}"
        ),
        (
            "Giỏ hàng TB",
            f"{bs:.2f} SP",
            f"{k['gio_hang_tb']:.2f} SP"
        ),
        (
            "Tỷ lệ mua lại",
            f"{lines.reordered.mean()*100:.1f}%",
            f"{k['ty_le_mua_lai']*100:.1f}%"
        )
    ]

    for col, (lab, v, full) in zip(c, tiles):

        col.metric(
            lab,
            v
        )

        col.caption(
            f"Toàn bộ dữ liệu: **{full}**"
        )

    st.markdown(
        "#### Kiểm định tính đại diện của mẫu 10%"
    )

    comp = S["comp"].copy()

    def fmt(x):
        try:
            f = float(x)

            return (
                f"{f:,.0f}"
                if f.is_integer() and f > 1000
                else f"{f:,.2f}"
            )

        except ValueError:
            return x

    for col_ in [
        "Toàn bộ (100%)",
        "Mẫu 10%"
    ]:
        comp[col_] = comp[col_].map(fmt)

    st.dataframe(
        comp,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Mẫu được lấy **theo khách hàng** "
        "(giữ trọn lịch sử mua của mỗi khách) "
        "nên các chỉ số hành vi của mẫu gần như trùng khớp "
        "với toàn bộ dữ liệu, đủ tin cậy để minh họa trực quan "
        "và chạy thử mô hình trên web."
    )

    st.markdown(
        "#### Kiến trúc hệ thống"
    )

    st.markdown(
        """
| Tầng | Công nghệ | Vai trò |
|---|---|---|
| Thu thập | Kaggle (6 file CSV, ~700 MB) | Dữ liệu gốc 37,3 triệu dòng |
| Lưu trữ | Apache Parquet (nén Snappy) | Lưu trữ dạng cột, giảm dung lượng & tăng tốc truy vấn |
| Xử lý | Apache Spark 4 (PySpark, Spark SQL) | ETL, broadcast join, tổng hợp phân tán |
| Học máy | Spark MLlib (FP-Growth, K-Means, LR/RF/GBT) | Luật kết hợp, phân khúc, dự đoán mua lại |
| Trình bày | Streamlit + Plotly | Dashboard tương tác trên mẫu 10% |
"""
    )


# ------------------------------------------------------------------
# RQ1
# ------------------------------------------------------------------

with tabs[1]:

    st.subheader(
        "RQ1 – Khách hàng đặt hàng khi nào?"
    )

    hm = (
        ox
        .groupby(
            [
                "order_dow",
                "order_hour_of_day"
            ]
        )
        .size()
        .unstack(fill_value=0)
        .reindex(range(7), fill_value=0)
    )

    hm.index = DOW_VI

    fig = px.imshow(
        hm,
        color_continuous_scale=BLUE_SEQ,
        aspect="auto",
        labels={
            "x": "Giờ trong ngày",
            "y": "",
            "color": "Số đơn"
        },
        title="Ma trận mật độ đơn hàng: ngày × giờ"
    )

    fig.update_traces(
        hovertemplate="%{y}, %{x}h: %{z:,} đơn<extra></extra>"
    )

    st.plotly_chart(
        style(fig, 360),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    hr = (
        ox
        .groupby("order_hour_of_day")
        .size()
        .reset_index(name="n")
    )

    hr["pct"] = (
        hr.n / hr.n.sum() * 100
    )

    fig = px.area(
        hr,
        x="order_hour_of_day",
        y="pct",
        title="Tỷ trọng đơn theo giờ (%)",
        color_discrete_sequence=[SERIES[0]],
        labels={
            "order_hour_of_day": "Giờ",
            "pct": "% đơn"
        }
    )

    fig.update_traces(
        hovertemplate="%{x}h: %{y:.2f}%<extra></extra>"
    )

    c1.plotly_chart(
        style(fig, 330),
        use_container_width=True
    )

    gp = (
        ox
        .dropna(
            subset=["days_since_prior_order"]
        )
        .groupby("days_since_prior_order")
        .size()
        .reset_index(name="n")
    )

    gp["pct"] = (
        gp.n / gp.n.sum() * 100
    )

    gp["loại"] = np.where(
        gp.days_since_prior_order.isin([7, 30]),
        "Đỉnh chu kỳ",
        "Khác"
    )

    fig = px.bar(
        gp,
        x="days_since_prior_order",
        y="pct",
        color="loại",
        color_discrete_map={
            "Đỉnh chu kỳ": SERIES[1],
            "Khác": SERIES[0]
        },
        title="Chu kỳ quay lại mua (ngày kể từ đơn trước)",
        labels={
            "days_since_prior_order": "Ngày",
            "pct": "% đơn",
            "loại": ""
        }
    )

    fig.update_traces(
        hovertemplate="%{x} ngày: %{y:.2f}%<extra></extra>"
    )

    c2.plotly_chart(
        style(fig, 330),
        use_container_width=True
    )

    peak = hr.loc[
        hr.pct.idxmax()
    ]

    st.success(
        f"Đỉnh đặt hàng lúc **{int(peak.order_hour_of_day)}h** "
        f"({peak.pct:.2f}% đơn). "
        f"{gp.loc[gp.days_since_prior_order <= 7, 'pct'].sum():.1f}% "
        f"đơn quay lại trong vòng 7 ngày; "
        f"{gp.loc[gp.days_since_prior_order == 30, 'pct'].sum():.1f}% "
        f"đơn cách đơn trước ≥ 30 ngày."
    )


# ------------------------------------------------------------------
# RQ2
# ------------------------------------------------------------------

with tabs[2]:

    st.subheader(
        "RQ2 – Sản phẩm & ngành hàng nào dẫn dắt doanh số và lòng trung thành?"
    )

    n_top = st.slider(
        "Số sản phẩm hiển thị",
        10,
        40,
        20,
        step=5
    )

    top = (
        fx
        .groupby(
            "product_name",
            observed=True
        )
        .agg(
            n=("order_id", "size"),
            rr=("reordered", "mean")
        )
        .sort_values(
            "n",
            ascending=False
        )
        .head(n_top)
        .reset_index()
        .sort_values("n")
    )

    fig = px.bar(
        top,
        x="n",
        y="product_name",
        orientation="h",
        title=f"Top {n_top} sản phẩm theo lượt mua",
        color_discrete_sequence=[SERIES[0]],
        labels={
            "n": "Lượt mua",
            "product_name": ""
        },
        custom_data=[top.rr * 100]
    )

    fig.update_traces(
        hovertemplate=(
            "%{y}<br>"
            "%{x:,} lượt · "
            "mua lại %{customdata[0]:.1f}%"
            "<extra></extra>"
        )
    )

    st.plotly_chart(
        style(fig, 24 * n_top + 120),
        use_container_width=True
    )

    dp = (
        fact
        .groupby(
            "department_vi",
            observed=True
        )
        .agg(
            n=("order_id", "size"),
            rr=("reordered", "mean"),
            sp=("product_id", "nunique")
        )
        .reset_index()
    )

    dp["share"] = (
        dp.n / dp.n.sum() * 100
    )

    c1, c2 = st.columns([1.3, 1])

    dp["nhãn"] = np.where(
        dp["share"].rank(ascending=False) <= 8,
        dp["department_vi"].astype(str),
        ""
    )

    fig = px.scatter(
        dp,
        x="share",
        y=dp.rr * 100,
        size="sp",
        text="nhãn",
        log_x=True,
        size_max=40,
        hover_name="department_vi",
        color_discrete_sequence=[SERIES[0]],
        title="Ngành hàng: tỷ trọng lượt mua vs. tỷ lệ mua lại",
        labels={
            "share": "Tỷ trọng lượt mua (%, log)",
            "y": "Tỷ lệ mua lại (%)",
            "sp": "Số SP"
        }
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=11,
        hovertemplate=(
            "%{hovertext}<br>"
            "Tỷ trọng %{x:.2f}% · "
            "Mua lại %{y:.1f}%"
            "<extra></extra>"
        )
    )

    c1.plotly_chart(
        style(fig, 460),
        use_container_width=True
    )

    bsz = (
        fx
        .groupby("order_id")
        .size()
        .clip(upper=50)
        .value_counts()
        .sort_index()
        .reset_index()
    )

    bsz.columns = [
        "size",
        "n"
    ]

    fig = px.bar(
        bsz,
        x="size",
        y="n",
        title="Kích thước giỏ hàng (≥ 50 gộp vào cột 50)",
        color_discrete_sequence=[SERIES[0]],
        labels={
            "size": "Số SP/đơn",
            "n": "Số đơn"
        }
    )

    fig.update_traces(
        hovertemplate="%{x} SP: %{y:,} đơn<extra></extra>"
    )

    c2.plotly_chart(
        style(fig, 460),
        use_container_width=True
    )

    st.markdown(
        "##### 🔎 Tra cứu nhanh một sản phẩm"
    )

    pick = st.selectbox(
        "Chọn sản phẩm",
        top.sort_values(
            "n",
            ascending=False
        ).product_name.tolist()
        +
        sorted(
            set(prods.product_name.astype(str))
            -
            set(top.product_name.astype(str))
        )[:3000],
        index=0
    )

    sub = fact[
        fact.product_name == pick
    ]

    if len(sub):

        c = st.columns(4)

        c[0].metric(
            "Lượt mua (mẫu)",
            f"{len(sub):,}"
        )

        c[1].metric(
            "Số khách đã mua",
            f"{sub.user_id.nunique():,}"
        )

        c[2].metric(
            "Tỷ lệ mua lại",
            f"{sub.reordered.mean()*100:.1f}%"
        )

        c[3].metric(
            "Vị trí TB trong giỏ",
            f"{sub.add_to_cart_order.mean():.1f}"
        )


# ------------------------------------------------------------------
# RQ3
# ------------------------------------------------------------------

with tabs[3]:

    st.subheader(
        "RQ3 – Sản phẩm nào thường được mua cùng nhau?"
    )

    st.caption(
        "Luật kết hợp khai phá bằng **FP-Growth (Spark MLlib)** "
        "trên toàn bộ 3,2 triệu giỏ hàng prior."
    )

    level = st.radio(
        "Cấp phân tích",
        [
            "Sản phẩm",
            "Quầy hàng (aisle)"
        ],
        horizontal=True
    )

    R = (
        S["rules_p"]
        if level == "Sản phẩm"
        else S["rules_a"]
    )

    c1, c2, c3 = st.columns(3)

    mlift = c1.slider(
        "Lift tối thiểu",
        0.0,
        float(np.ceil(R.lift.max())),
        1.0,
        0.1
    )

    mconf = c2.slider(
        "Confidence tối thiểu (%)",
        0,
        int(R.confidence.max() * 100),
        0
    )

    kw = c3.text_input(
        "Lọc theo từ khóa (vd: Banana, yogurt)"
    )

    Rf = R[
        (R.lift >= mlift)
        &
        (R.confidence * 100 >= mconf)
    ]

    if kw:
        Rf = Rf[
            Rf.antecedent.str.contains(
                kw,
                case=False
            )
            |
            Rf.consequent.str.contains(
                kw,
                case=False
            )
        ]

    st.write(
        f"**{len(Rf):,}** luật thỏa điều kiện "
        f"(tổng {len(R):,})."
    )

    show = (
        Rf
        .sort_values(
            "lift",
            ascending=False
        )
        .copy()
    )

    show["support (%)"] = (
        show.support * 100
    ).round(3)

    show["confidence (%)"] = (
        show.confidence * 100
    ).round(2)

    show["lift"] = (
        show.lift
    ).round(3)

    st.dataframe(
        show[
            [
                "antecedent",
                "consequent",
                "support (%)",
                "confidence (%)",
                "lift",
                "freq_xy"
            ]
        ]
        .rename(
            columns={
                "antecedent": "Nếu mua (X)",
                "consequent": "Thì mua (Y)",
                "freq_xy": "Số giỏ chứa X∪Y"
            }
        ),
        hide_index=True,
        use_container_width=True,
        height=320
    )

    if len(Rf):

        fig = px.scatter(
            Rf,
            x=Rf.support * 100,
            y=Rf.confidence * 100,
            color="lift",
            color_continuous_scale=BLUE_SEQ[2:],
            hover_data={
                "antecedent": True,
                "consequent": True
            },
            title="Không gian support – confidence (màu = lift)",
            labels={
                "x": "Support (%)",
                "y": "Confidence (%)"
            }
        )

        fig.update_traces(
            marker=dict(
                size=10,
                line=dict(
                    width=1,
                    color="white"
                )
            )
        )

        st.plotly_chart(
            style(fig, 400),
            use_container_width=True
        )

    st.markdown(
        "##### 🛍️ Gợi ý mua kèm (tính trực tiếp trên mẫu 10%)"
    )

    popular = (
        fact.product_name
        .value_counts()
        .head(300)
        .index
        .astype(str)
        .tolist()
    )

    anchor = st.selectbox(
        "Khi khách thêm vào giỏ…",
        popular,
        index=0
    )

    prior = fact[
        fact.eval_set == "prior"
    ]

    n_orders = prior.order_id.nunique()

    oid = prior.loc[
        prior.product_name == anchor,
        "order_id"
    ].unique()

    co = (
        prior[
            prior.order_id.isin(oid)
            &
            (prior.product_name != anchor)
        ]
        .groupby(
            "product_name",
            observed=True
        )
        .size()
    )

    base = (
        prior
        .groupby(
            "product_name",
            observed=True
        )
        .size()
    )

    rec = pd.DataFrame(
        {
            "cùng giỏ": co
        }
    ).join(
        base.rename("tổng")
    )

    rec = rec[
        (rec["cùng giỏ"] >= max(
            30,
            0.005 * len(oid)
        ))
        &
        (rec["tổng"] >= 200)
    ]

    rec["confidence (%)"] = (
        rec["cùng giỏ"]
        / len(oid)
        * 100
    )

    rec["lift"] = (
        (rec["cùng giỏ"] / len(oid))
        /
        (rec["tổng"] / n_orders)
    )

    rec = (
        rec
        .sort_values(
            "lift",
            ascending=False
        )
        .head(10)
        .reset_index()
        .rename(
            columns={
                "product_name": "Gợi ý",
                "cùng giỏ": "Số giỏ cùng mua",
                "tổng": "Tổng số giỏ có SP"
            }
        )
    )

    st.caption(
        f"Tính trên {len(oid):,} giỏ hàng "
        f"(mẫu 10%) có chứa sản phẩm đã chọn; "
        f"chỉ xét sản phẩm xuất hiện trong ≥ 0,5% số giỏ đó."
    )

    st.dataframe(
        rec.round(2),
        hide_index=True,
        use_container_width=True
    )


# ------------------------------------------------------------------
# RQ4
# ------------------------------------------------------------------

with tabs[4]:

    st.subheader(
        "RQ4 – Khách hàng thuộc những phân khúc hành vi nào?"
    )

    seg = S["seg"]

    c1, c2 = st.columns([1, 1.2])

    fig = px.bar(
        seg.sort_values("n_users"),
        x="n_users",
        y="segment",
        orientation="h",
        color="segment",
        color_discrete_sequence=SERIES,
        title="Quy mô phân khúc (K-Means, 100% khách hàng)",
        labels={
            "n_users": "Số khách hàng",
            "segment": ""
        },
        custom_data=["share"]
    )

    fig.update_traces(
        hovertemplate=(
            "%{y}: %{x:,} KH "
            "(%{customdata[0]:.1f}%)"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        showlegend=False
    )

    c1.plotly_chart(
        style(fig, 360),
        use_container_width=True
    )

    t = seg[
        [
            "segment",
            "n_users",
            "share",
            "recency",
            "frequency",
            "monetary_items",
            "avg_gap_days",
            "avg_basket",
            "reorder_ratio"
        ]
    ].copy()

    t.columns = [
        "Phân khúc",
        "Số KH",
        "Tỷ trọng %",
        "Recency (ngày)",
        "Số đơn",
        "Tổng SP đã mua",
        "Chu kỳ TB (ngày)",
        "Giỏ TB",
        "Tỷ lệ mua lại"
    ]

    c2.dataframe(
        t.round(2),
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "##### 🎯 Chiến lược marketing gợi ý"
    )

    for _, r in seg.iterrows():

        st.markdown(
            f"- **{r['segment']}** "
            f"({r['share']:.1f}%): "
            f"{r['strategy']}"
        )

    st.markdown(
        "##### 🧪 Thử nghiệm phân cụm lại trên mẫu 10% (scikit-learn)"
    )

    k_sel = st.slider(
        "Số cụm k",
        2,
        8,
        4
    )

    so = orders.copy()

    ptr = orders[
        orders.eval_set == "prior"
    ]

    uf = (
        ptr
        .groupby("user_id")
        .agg(
            frequency=("order_id", "size"),
            avg_gap_days=(
                "days_since_prior_order",
                "mean"
            )
        )
    )

    uf = uf.join(
        orders[
            orders.eval_set != "prior"
        ]
        .set_index("user_id")[
            "days_since_prior_order"
        ]
        .rename("recency")
    )

    li = (
        fact[
            fact.eval_set == "prior"
        ]
        .groupby("user_id")
        .agg(
            items=("product_id", "size"),
            reorder_ratio=("reordered", "mean")
        )
    )

    uf = uf.join(li).dropna()

    uf["avg_basket"] = (
        uf["items"]
        / uf["frequency"]
    )

    X = np.column_stack(
        [
            uf.recency,
            np.log1p(uf.frequency),
            np.log1p(uf["items"]),
            uf.avg_gap_days,
            uf.avg_basket,
            uf.reorder_ratio
        ]
    )

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)

    lab = KMeans(
        n_clusters=k_sel,
        n_init=5,
        random_state=42
    ).fit_predict(Xs)

    uf["cụm"] = (
        "Cụm "
        + lab.astype(str)
    )

    fig = px.scatter(
        uf.sample(
            min(4000, len(uf)),
            random_state=1
        ),
        x="frequency",
        y="avg_basket",
        color="cụm",
        log_x=True,
        color_discrete_sequence=SERIES,
        opacity=0.7,
        category_orders={
            "cụm": [
                f"Cụm {i}"
                for i in range(k_sel)
            ]
        },
        title=(
            f"Phân bố khách hàng mẫu theo "
            f"số đơn × giỏ hàng TB (k = {k_sel})"
        ),
        labels={
            "frequency": "Số đơn (log)",
            "avg_basket": "Giỏ hàng TB"
        }
    )

    st.plotly_chart(
        style(fig, 420),
        use_container_width=True
    )

    st.dataframe(
        uf.groupby("cụm")[
            [
                "recency",
                "frequency",
                "items",
                "avg_gap_days",
                "avg_basket",
                "reorder_ratio"
            ]
        ]
        .mean()
        .round(2)
        .join(
            uf["cụm"]
            .value_counts()
            .rename("số KH")
        ),
        use_container_width=True
    )


# ------------------------------------------------------------------
# RQ5
# ------------------------------------------------------------------

with tabs[5]:

    st.subheader(
        "RQ5 – Dự đoán sản phẩm khách sẽ mua lại trong đơn kế tiếp"
    )

    m = S["metrics"]

    c1, c2 = st.columns([1.1, 1])

    mm = m.melt(
        id_vars="Mô hình",
        value_vars=[
            "AUC-ROC",
            "AUC-PR",
            "F1 tối ưu"
        ],
        var_name="Chỉ số",
        value_name="Giá trị"
    )

    fig = px.bar(
        mm,
        x="Chỉ số",
        y="Giá trị",
        color="Mô hình",
        barmode="group",
        color_discrete_sequence=SERIES,
        title=(
            "So sánh mô hình trên tập kiểm tra "
            "(100% dữ liệu, chia theo khách hàng)"
        )
    )

    fig.update_traces(
        hovertemplate=(
            "%{x}: %{y:.4f}"
            "<extra></extra>"
        )
    )

    c1.plotly_chart(
        style(fig, 380),
        use_container_width=True
    )

    imp = (
        S["imp"]
        .sort_values("importance")
    )

    fig = px.bar(
        imp,
        x="importance",
        y="feature_vi",
        orientation="h",
        color_discrete_sequence=[SERIES[0]],
        title="Mức độ quan trọng của đặc trưng (GBT)",
        labels={
            "importance": "Importance",
            "feature_vi": ""
        }
    )

    c2.plotly_chart(
        style(fig, 380),
        use_container_width=True
    )

    st.dataframe(
        m.round(4),
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        "##### 🔮 Demo: dự đoán giỏ hàng tiếp theo của một khách hàng "
        "(mô hình Logistic Regression)"
    )

    lr = S["lr"]

    train_users = (
        orders.loc[
            orders.eval_set == "train",
            "user_id"
        ]
        .sort_values()
        .unique()
    )

    uid = st.selectbox(
        "Chọn mã khách hàng (khách có đơn kiểm chứng)",
        train_users[:2000]
    )

    uo = orders[
        orders.user_id == uid
    ]

    t_order = (
        uo[
            uo.eval_set == "train"
        ]
        .iloc[0]
    )

    up = fact[
        (fact.user_id == uid)
        &
        (fact.eval_set == "prior")
    ]

    u_orders = int(
        uo[
            uo.eval_set == "prior"
        ]
        .order_number
        .max()
    )

    g = (
        up
        .groupby("product_id")
        .agg(
            up_orders=("order_id", "size"),
            up_first=("order_number", "min"),
            up_last=("order_number", "max"),
            up_avg_cart_pos=(
                "add_to_cart_order",
                "mean"
            )
        )
        .reset_index()
    )

    g["u_orders"] = u_orders

    g["u_avg_gap"] = (
        uo[
            uo.eval_set == "prior"
        ]
        .days_since_prior_order
        .mean()
    )

    g["u_reorder_ratio"] = (
        up.reordered.mean()
    )

    g["u_avg_basket"] = (
        len(up)
        / u_orders
    )

    g["u_distinct"] = (
        up.product_id.nunique()
    )

    g = g.merge(
        S["pfeat"],
        on="product_id",
        how="left"
    )

    g["up_order_rate"] = (
        g.up_orders
        / g.u_orders
    )

    g["up_orders_since_last"] = (
        g.u_orders
        - g.up_last
    )

    g["up_rate_since_first"] = (
        g.up_orders
        /
        (
            g.u_orders
            - g.up_first
            + 1
        )
    )

    g["t_days_since_prior"] = (
        t_order.days_since_prior_order
    )

    Xf = (
        g[lr["features"]]
        .fillna(0)
        .to_numpy(dtype=float)
    )

    z = (
        (
            Xf
            - np.array(lr["mean"])
        )
        /
        np.where(
            np.array(lr["std"]) == 0,
            1,
            np.array(lr["std"])
        )
    ) @ np.array(lr["coef"]) + lr["intercept"]

    g["xác suất"] = (
        1
        /
        (
            1
            + np.exp(-z)
        )
    )

    actual = set(
        fact[
            fact.order_id == t_order.order_id
        ].product_id
    )

    g = g.merge(
        prods[
            [
                "product_id",
                "product_name",
                "department_vi"
            ]
        ],
        on="product_id"
    )

    g["thực tế đã mua lại?"] = np.where(
        g.product_id.isin(actual),
        "✅ có",
        "—"
    )

    topk = st.slider(
        "Số sản phẩm gợi ý (Top-K)",
        5,
        20,
        10
    )

    res = (
        g
        .sort_values(
            "xác suất",
            ascending=False
        )
        .head(topk)
    )

    hits = (
        res["thực tế đã mua lại?"]
        == "✅ có"
    ).sum()

    n_re = len(
        actual
        &
        set(g.product_id)
    )

    c = st.columns(4)

    c[0].metric(
        "Số đơn trước đó",
        u_orders
    )

    c[1].metric(
        "SP ứng viên (đã từng mua)",
        len(g)
    )

    c[2].metric(
        f"Trúng trong Top-{topk}",
        f"{hits}/{topk}"
    )

    c[3].metric(
        "Recall@K (SP mua lại thực tế)",
        f"{hits / n_re * 100:.0f}%"
        if n_re
        else "—"
    )

    st.dataframe(
        res[
            [
                "product_name",
                "department_vi",
                "up_orders",
                "up_orders_since_last",
                "xác suất",
                "thực tế đã mua lại?"
            ]
        ]
        .rename(
            columns={
                "product_name": "Sản phẩm",
                "department_vi": "Ngành hàng",
                "up_orders": "Số lần đã mua",
                "up_orders_since_last": "Số đơn từ lần mua cuối"
            }
        )
        .round(3),
        hide_index=True,
        use_container_width=True
    )


# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------

st.divider()

st.caption(
    "© 2026 – Đào Thị Hồng Vân · "
    "Nguồn dữ liệu: Instacart Market Basket Analysis (Kaggle). "
    "Pipeline Big Data: Apache Spark 4 · PySpark · Spark MLlib · Streamlit."
)
