from datetime import datetime
import streamlit as st
from modules.emojis import country_emoji_map

def _pretty_usd(x):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    x = float(x)
    ax = abs(x)
    if ax >= 1e12:  return f"${x/1e12:.2f}T"
    if ax >= 1e9:   return f"${x/1e9:.2f}B"
    if ax >= 1e6:   return f"${x/1e6:.2f}M"
    if ax >= 1e3:   return f"${x/1e3:.2f}K"
    return f"${x:,.0f}"

def _table_pdf_bytes(df, logo_map, title="Treasury ranking list"):
    try:
        from fpdf import FPDF
    except Exception:
        st.error("Install fpdf2 first  pip install fpdf2")
        return b""
    import io, base64

    # ---------- PDF with footer ----------
    class _PDF(FPDF):
        def footer(self_inner):
            # thin rule
            self_inner.set_y(-17)
            self_inner.set_draw_color(73, 80, 87)
            self_inner.set_line_width(0.1)
            self_inner.line(self_inner.l_margin, self_inner.get_y(),
                            self_inner.w - self_inner.r_margin, self_inner.get_y())

            y = self_inner.get_y() + 2
            col_w = (self_inner.w - self_inner.l_margin - self_inner.r_margin) / 3.0
            x0 = self_inner.l_margin

            # left: timestamp
            now = datetime.now()
            ts = f"{now.strftime('%B')} {now.day}, {now.year}, {now.strftime('%H:%M:%S')}"
            self_inner.set_font("Helvetica", "", 8)
            self_inner.set_text_color(173, 181, 189)
            self_inner.set_xy(x0, y)
            self_inner.cell(col_w, 5, ts, align="L")

            # center: brand + socials (clickable)
            center_labels = [self_inner._brand_name, "LinkedIn", "X"]
            sep = " | "
            text = sep.join(center_labels)
            self_inner.set_xy(x0 + col_w, y)
            self_inner.cell(col_w, 5, text, align="C")

            # clickable areas for each label
            cx = x0 + col_w + (col_w - self_inner.get_string_width(text)) / 2.0
            for i, lbl in enumerate(center_labels):
                tw = self_inner.get_string_width(lbl)
                url = self_inner._social_urls.get(lbl)
                if url:
                    self_inner.link(cx, y, tw, 5, url)
                cx += tw
                if i < len(center_labels) - 1:
                    cx += self_inner.get_string_width(sep)

            # right: copyright
            self_inner.set_xy(x0 + 2 * col_w, y)
            self_inner.cell(col_w, 5, f"© {now.year} {self_inner._brand_name}", align="R")

    pdf = _PDF(orientation="L", unit="mm", format="A4")
    pdf._brand_name = "Crypto Treasury Tracker"
    pdf._social_urls = {
        "Crypto Treasury Tracker":  "https://crypto-treasury-tracker.streamlit.app/",
        "LinkedIn": st.secrets.get("LINKEDIN_URL", "https://www.linkedin.com/in/benjaminschellinger/"),
        "X":        st.secrets.get("X_URL",        "https://x.com/CTTbyBen"),
    }
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    # --- Dark theme background & defaults ---
    pdf.set_fill_color(22, 24, 28)                  # page bg
    pdf.rect(0, 0, pdf.w, pdf.h, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_draw_color(73, 80, 87)                  # subtle borders
    pdf.set_line_width(0.1)

    # --- Title (aligned logo) ---
    title_h = 8.0
    title_y = pdf.get_y()

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, title_h, title, ln=1)
    pdf.ln(2)

    # --- Columns (fit to page width) ---
    avail_w = pdf.w - pdf.l_margin - pdf.r_margin
    cols = [
        ("Rank",            0.04),
        ("Entity Name",     0.15),
        ("Ticker",          0.05),
        ("Entity Type",     0.12),
        ("Country",         0.09),
        ("Asset",           0.035),
        ("Holdings (Unit)", 0.09),
        ("% Supply",        0.10),
        ("USD Value",       0.07),
        ("Market Cap",      0.07),
        ("mNAV",            0.05),
        ("Premium",         0.07),
        ("TTMCR",           0.06),
    ]



    col_w = [round(avail_w * r, 2) for _, r in cols]
    col_x = [pdf.l_margin]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)

    # --- Header ---
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(43, 47, 54)                  # header bg
    y = pdf.get_y()
    for (name, _), x, w in zip(cols, col_x, col_w):
        pdf.set_xy(x, y)
        pdf.cell(w, 8, name, border=1, align="L", fill=True)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 9)

    # --- Helpers ---
    def _best_text_on(bg_rgb):
        r, g, b = [c/255.0 for c in bg_rgb]
        def _lin(c): return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4
        L = 0.2126*_lin(r) + 0.7152*_lin(g) + 0.0722*_lin(b)
        c_white = (1.0 + 0.05) / (L + 0.05)
        c_black = (L + 0.05) / 0.05
        return (255, 255, 255) if c_white >= c_black else (0, 0, 0)

    # distinct, finance-friendly palette (no grey)
    type_palette = {"Public Company": (123, 197, 237), # blue 
                        "Private Company": (232, 118, 226), # rose 
                        "DAO": (237, 247, 94), # amber 
                        "Non-Profit Organization": (34, 197, 94), # green 
                        "Government": (245, 184, 122), # slate 
                        "Other": (250, 250, 250), # white
                        }

    # Rounded capsule for tracks/pills
    def _draw_capsule(x, y, w, h, fill_rgb):
        r = h / 2.0
        pdf.set_fill_color(*fill_rgb)
        pdf.rect(x + r, y, max(0, w - 2*r), h, style="F")  # center
        pdf.ellipse(x, y, h, h, style="F")                  # left cap
        pdf.ellipse(x + w - h, y, h, h, style="F")          # right cap

    # Left-rounded fill (flat right) – scales correctly, no bubbles
    def _fill_left_capsule(x, y, w, h, fill_rgb):
        r = h / 2.0
        if w <= 0:
            return
        pdf.set_fill_color(*fill_rgb)
        if w <= r:
            pdf.rect(x, y + 0.8, max(w, 0.3), h - 1.6, style="F")
        else:
            pdf.ellipse(x, y, h, h, style="F")
            pdf.rect(x + r, y, w - r, h, style="F")

    # Round entity-type pill with auto black/white text
    def draw_type_pill(x, y, w, h, text):
        color = type_palette.get(text, (6, 182, 212))
        pill_h = 6.2
        pad_x = 3.6
        pdf.set_font("Helvetica", "B", 8)
        tw = pdf.get_string_width(text)
        pill_w = min(w - 4, tw + 2*pad_x)
        pill_x = x + (w - pill_w) / 2.0
        pill_y = y + (h - pill_h) / 2.0
        _draw_capsule(pill_x, pill_y, pill_w, pill_h, color)
        tr, tg, tb = _best_text_on(color)  # ONLY white or black depending on bg
        pdf.set_text_color(tr, tg, tb)
        pdf.set_xy(pill_x + (pill_w - tw)/2.0, pill_y + 0.9)
        pdf.cell(tw, pill_h - 1.8, text, align="C")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 9)

    # Rounded progress bar with correct proportional fill
    def draw_bar(x, y, w, h, pct):
        _draw_capsule(x, y, w, h, (52, 58, 64))  # rounded track
        pct = max(0.0, min(float(pct), 100.0))
        fill_w = (pct / 100.0) * w
        if pct >= 99.5:
            fill_w = w
        _fill_left_capsule(x, y, fill_w, h, (32, 201, 151))  # left-rounded, flat right
        label = f"{pct:.2f}%"
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(255, 255, 255)
        tw = pdf.get_string_width(label)
        pdf.set_xy(x + (w - tw)/2.0, y + (h - 3.8)/2.0)
        pdf.cell(tw, 3.8, label, align="C")
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 9)

    def text_fit(s, w, ellipsis=True):
        s = str(s)
        if pdf.get_string_width(s) <= w - 2:
            return s
        out = s
        suffix = "..." if ellipsis else ""
        while out and pdf.get_string_width(out + suffix) > w - 2:
            out = out[:-1]
        return out + suffix if out else suffix

    MAX_ROWS_PER_PAGE = 20
    rows_on_page = 0

    row_h = 8
    for rank, row in df.iterrows():
        # page break guard
        if rows_on_page >= MAX_ROWS_PER_PAGE or pdf.get_y() + row_h > (pdf.h - pdf.b_margin):
            pdf.add_page()
            # repaint dark bg
            pdf.set_fill_color(22, 24, 28)
            pdf.rect(0, 0, pdf.w, pdf.h, style="F")
            pdf.set_text_color(255, 255, 255)
            pdf.set_draw_color(73, 80, 87)
            pdf.set_line_width(0.1)

            # header
            pdf.set_font("Helvetica", "B", 9)
            header_y = pdf.get_y()
            pdf.set_fill_color(43, 47, 54)
            for (name, _), x, w in zip(cols, col_x, col_w):
                pdf.set_xy(x, header_y)
                pdf.cell(w, 8, name, border=1, align="L", fill=True)
            pdf.ln(8)
            pdf.set_font("Helvetica", "", 9)

            rows_on_page = 0  # reset counter


        y = pdf.get_y()

        # Rank
        pdf.set_xy(col_x[0], y)
        pdf.cell(col_w[0], row_h, str(rank), border=1, align="L")

        # Entity (with emoji already in df["Entity"])
        pdf.set_xy(col_x[1], y)
        entity_txt = text_fit(row.get("Entity Name", ""), col_w[1])
        pdf.cell(col_w[1], row_h, entity_txt, border=1, align="L")


        # Stock Ticker
        pdf.set_xy(col_x[2], y)
        ticker_txt = str(row.get("Ticker", "") or "-")
        pdf.cell(col_w[2], row_h, text_fit(ticker_txt, col_w[2]), border=1, align="L")

        # Entity Type (rounded pill)
        pdf.set_xy(col_x[3], y)
        pdf.cell(col_w[3], row_h, "", border=1)
        draw_type_pill(col_x[3], y, col_w[3], row_h, str(row["Entity Type"]))
       
        # Country
        pdf.set_xy(col_x[4], y)
        country = text_fit(row.get("Country", ""), col_w[4])
        pdf.cell(col_w[4], row_h, country, border=1, align="L")

        # Asset (logo)
        pdf.set_xy(col_x[5], y)
        pdf.cell(col_w[5], row_h, "", border=1)
        asset = str(row.get("Crypto Asset", ""))
        if asset in logo_map and logo_map[asset]:
            b64 = logo_map[asset].split(",")[-1]
            img = io.BytesIO(base64.b64decode(b64))
            img_h = 5.2
            x_img = col_x[5] + (col_w[5] - img_h) / 2.0  # center inside Asset column
            y_img = y + (row_h - img_h) / 2.0
            pdf.image(img, x=x_img, y=y_img, h=img_h)


        # Holdings
        pdf.set_xy(col_x[6], y)
        pdf.cell(col_w[6], row_h, f"{int(row['Holdings (Unit)']):,}".replace(",", " "), border=1, align="R")

        # % Supply
        pdf.set_xy(col_x[7], y)
        pdf.cell(col_w[7], row_h, "", border=1)
        draw_bar(col_x[7] + 2, y + 1.4, col_w[7] - 4, row_h - 2.8, float(row["% of Supply"]))

        # USD Value (pretty, dash if NA)
        pdf.set_xy(col_x[8], y)
        uv_txt = _pretty_usd(row.get("USD Value", None))
        pdf.cell(col_w[8], row_h, uv_txt, border=1, align="R")

        # Market Cap (USD pretty, dash if NA)
        pdf.set_xy(col_x[9], y)
        mc_txt = _pretty_usd(row.get("Market Cap", None))
        pdf.cell(col_w[9], row_h, mc_txt, border=1, align="R")

        # mNAV with color
        pdf.set_xy(col_x[10], y)
        _mn = row.get("mNAV", None)
        if _mn is None or (isinstance(_mn, float) and _mn != _mn):
            mn_txt = "-"
            pdf.set_text_color(255, 255, 255)  # white for dash
        else:
            mn_txt = f"{_mn:.2f}"
            if _mn > 1:
                pdf.set_text_color(67, 209, 160)   # NEUTRAL_POS green
            elif _mn < 1:
                pdf.set_text_color(249, 65, 68)    # NEUTRAL_NEG red
            else:
                pdf.set_text_color(255, 165, 0)    # orange for exactly 1
        pdf.cell(col_w[10], row_h, mn_txt, border=1, align="R")
        pdf.set_text_color(255, 255, 255)  # reset

        # Premium (2dp %, dash if NA)
        pdf.set_xy(col_x[11], y)
        _pr = row.get("Premium", None)
        pr_txt = "-" if (_pr is None or (isinstance(_pr, float) and _pr != _pr)) else f"{_pr:.2f}%"
        pdf.cell(col_w[11], row_h, pr_txt, border=1, align="R")

        # TTMCR (2dp %, dash if NA)
        pdf.set_xy(col_x[12], y)
        _tt = row.get("TTMCR", None)
        tt_txt = "-" if (_tt is None or (isinstance(_tt, float) and _tt != _tt)) else f"{_tt:.2f}%"
        pdf.cell(col_w[12], row_h, tt_txt, border=1, align="R")

        pdf.set_y(y + row_h)
        rows_on_page += 1

    out = pdf.output(dest="S")
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin1")
