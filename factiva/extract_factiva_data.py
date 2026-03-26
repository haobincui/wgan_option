import os
import glob
import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import pandas as pd

def _clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in s.split("\n")).strip()

def _parse_factiva_html(html_path: str, year_label: Optional[str] = None) -> List[Dict]:
    """
    解析单个 HTML，提取每篇文章的 HD/PD/ET/SN/LP/AN。
    year_label: 若提供，则在 SourceFile 中前缀为 'year/filename'
    """
    with open(html_path, "rb") as f:
        soup = BeautifulSoup(f, "lxml")

    results = []
    for art in soup.select("div.article.enArticle"):
        table = art.find("table")
        if not table:
            continue

        row_map = {}
        parent_div = art.find_parent("div", class_="article")
        row_map["ArticleID"] = parent_div.get("id") if parent_div else ""

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            left = tds[0]
            code_tag = left.find("b")
            code = code_tag.get_text(strip=True) if code_tag else left.get_text(strip=True)
            code = code.replace("\xa0", " ").strip()

            right = tds[1]

            if code in {"HD", "PD", "ET", "SN"}:
                value = right.get_text(" ", strip=True)
                row_map[code] = _clean_text(value)

            elif code == "LP":
                parts = []
                for node in right.find_all(["p", "pre"]):
                    txt = node.get_text("\n", strip=True)
                    if txt:
                        parts.append(txt)
                if not parts:
                    parts.append(right.get_text(" ", strip=True))
                row_map["LP"] = _clean_text("\n".join(parts))

            elif code == "AN":
                row_map["AN"] = _clean_text(right.get_text(" ", strip=True))

        base = os.path.basename(html_path)
        row_map["SourceFile"] = f"{year_label}/{base}" if year_label else base

        if any(row_map.get(k) for k in ("HD", "PD", "ET", "SN", "LP")):
            results.append(row_map)

    return results

def _collect_from_path(input_path: str) -> pd.DataFrame:
    """
    输入可以是：
      1) 单个 HTML 文件；
      2) 目录（自动抓取目录下的 *.html / *.htm）；
      3) 含通配符的模式（如 './input/2022/*.html' 或 './input/*/*.html'）。
    """
    # 先尝试把输入当作通配符模式展开（支持 ** 递归）
    files = sorted(glob.glob(input_path, recursive=True))

    # 如果没匹配到，再判断是否是目录；是目录则取目录下的 html/htm
    if not files and os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.html"))) + \
                sorted(glob.glob(os.path.join(input_path, "*.htm")))

    # 如果仍然为空，但路径确实存在（单文件）
    if not files and os.path.exists(input_path):
        files = [input_path]

    all_rows = []
    for fp in files:
        try:
            # 从路径中提取最后一个 4 位年份（19xx 或 20xx）
            # 例：./input/2022/foo.html  -> '2022'
            #     ./input/backup_2023/bar.html -> '2023'
            year_matches = re.findall(r'(?:19|20)\d{2}', fp)
            year_label = year_matches[-1] if year_matches else None

            all_rows.extend(_parse_factiva_html(fp, year_label=year_label))
        except Exception as e:
            print(f"⚠️ Parse failed for {fp}: {e}")

    if not all_rows:
        return pd.DataFrame(columns=["SourceFile","ArticleID","AN","HD","PD","ET","SN","LP"])

    df = pd.DataFrame(all_rows)
    cols = ["SourceFile", "ArticleID", "AN", "HD", "PD", "ET", "SN", "LP"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def extract_factiva_to_table(input_path: str, output_path: str) -> pd.DataFrame:
    """
    解析 Factiva/道琼斯 HTML（支持通配符），提取 HD/PD/ET/SN/LP，
    保存到 output_path（.csv 或 .xlsx），并返回 DataFrame。
    SourceFile 将自动带上路径里识别出的年份前缀（如 '2022/filename.html'）。
    """
    df = _collect_from_path(input_path)
    if df.empty:
        print("No records found.")
        return df

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path.lower())[1]

    if ext == ".xlsx":
        if len(df) > 1_048_576:
            alt = os.path.splitext(output_path)[0] + ".csv"
            df.to_csv(alt, index=False)
            print(f"⚠️ Rows exceed Excel limit. Saved CSV instead: {alt}")
        else:
            df.to_excel(output_path, index=False)
            print(f"✅ Saved Excel to: {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"✅ Saved CSV to: {output_path}")

    return df

import pandas as pd
from datetime import datetime

def add_datetime_from_pd_et(df, pd_col="PD", et_col="ET", out_col="datetime"):
    """
    将两列（日期 PD，如 '31 December 2022'；时间 ET，如 '00:00'）合并为 pandas 的 datetime。
    - 首选严格格式：'%d %B %Y %H:%M'
    - 失败则自动解析（dayfirst=True）
    - 若 ET 为空，默认 '00:00'
    """
    pd_series = df[pd_col].astype(str).str.strip()
    et_series = df[et_col].astype(str).str.strip().replace({"": "00:00", "nan": "00:00"})

    combo = pd_series + " " + et_series

    # 1) 严格格式（更快更稳）
    dt = pd.to_datetime(combo, format="%d %B %Y %H:%M", errors="coerce")

    # 2) 回退：自动解析（允许不同大小写或 AM/PM 等）
    mask = dt.isna()
    if mask.any():
        dt.loc[mask] = pd.to_datetime(combo[mask], errors="coerce", dayfirst=True)

    df[out_col] = dt
    return df


if __name__ == '__main__':
    # extract_factiva_to_table("./input/*/*.html", "./output/news_all_years.xlsx")

    df = extract_factiva_to_table("./input/*/*.html", "./output/news_all_years.xlsx")

    # 可选：合成 datetime 列并另存
    if not df.empty:
        df = add_datetime_from_pd_et(df, pd_col="PD", et_col="ET", out_col="Datetime")
        out2 = "./output/news_all_years_with_datetime.xlsx"
        # 简单处理 Excel 行数上限
        if len(df) > 1_048_576:
            df.to_csv(out2.replace(".xlsx", ".csv"), index=False)
        else:
            df.to_excel(out2, index=False)
