import os
import json
import pandas as pd
from transformers import AutoTokenizer

def _read_table(input_path: str) -> pd.DataFrame:
    ext = os.path.splitext(input_path.lower())[1]
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(input_path)
    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(input_path, sep=sep)
    raise ValueError(f"Unsupported input format: {ext}")

def _save_table(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path.lower())[1]
    if ext in [".xlsx", ".xls"]:
        # Excel 行上限 ~ 1,048,576
        if len(df) > 1_048_576:
            alt = os.path.splitext(output_path)[0] + ".csv"
            df.to_csv(alt, index=False)
            print(f"⚠️ Rows exceed Excel limit. Saved CSV instead: {alt}")
        else:
            df.to_excel(output_path, index=False)
            print(f"✅ Saved Excel to: {output_path}")
    elif ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        df.to_csv(output_path, index=False, sep=sep)
        print(f"✅ Saved to: {output_path}")
    else:
        raise ValueError(f"Unsupported output format: {ext}")

def _encode_texts(tokenizer, texts, add_special_tokens=True, padding=False, truncation=False, max_length=None):
    """
    返回 (input_ids_list, lengths_list)
    """
    enc = tokenizer(
        texts,
        add_special_tokens=add_special_tokens,
        padding=padding,          # 可用 "max_length" 来固定长度
        truncation=truncation,    # True 时需要配合 max_length
        max_length=max_length,
        return_attention_mask=False
    )
    input_ids = enc["input_ids"]
    lengths = [len(ids) for ids in input_ids]
    return input_ids, lengths

def tokenize_lp_hd_columns(
    input_path: str,
    tokenizer_path: str,
    output_path: str,
    lp_col: str = "LP",
    hd_col: str = "HD",
    # 如需固定长度，请改为 padding="max_length", truncation=True, max_length=XXX
    padding=False,
    truncation=False,
    max_length=None,
):
    """
    从本地加载 tokenizer，将 DataFrame 的 LP/HD 两列分别转为 token，输出到新列：
      - lp_token / lp_token_count
      - hd_token / hd_token_count

    参数：
      input_path:  输入表路径（.csv/.tsv/.xlsx）
      tokenizer_path: 本地 tokenizer 目录（可包含 tokenizer.json / vocab.txt / merges.txt 等）
      output_path: 输出表路径（.csv/.tsv/.xlsx）
      lp_col/hd_col: 源文本列名
      padding/truncation/max_length: 按需固定长度时使用
    """
    # 1) 读数据
    df = _read_table(input_path)

    # 2) 加载本地 tokenizer
    tok = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
        local_files_only=True
    )

    # 3) 处理 LP 列
    if lp_col in df.columns:
        lp_texts = df[lp_col].astype(str).fillna("").tolist()
        lp_ids, lp_lens = _encode_texts(tok, lp_texts, padding=padding, truncation=truncation, max_length=max_length)
        df["lp_token"] = [json.dumps(ids, ensure_ascii=False) for ids in lp_ids]
        # 若固定长度，lp_token_count 也可用 attention_mask 求真长度；此处直接用 len(ids)
        df["lp_token_count"] = lp_lens
    else:
        print(f"⚠️ Column '{lp_col}' not found. Filling empty results.")
        df["lp_token"] = "[]"
        df["lp_token_count"] = 0

    # 4) 处理 HD 列
    if hd_col in df.columns:
        hd_texts = df[hd_col].astype(str).fillna("").tolist()
        hd_ids, hd_lens = _encode_texts(tok, hd_texts, padding=padding, truncation=truncation, max_length=max_length)
        df["hd_token"] = [json.dumps(ids, ensure_ascii=False) for ids in hd_ids]
        df["hd_token_count"] = hd_lens
    else:
        print(f"⚠️ Column '{hd_col}' not found. Filling empty results.")
        df["hd_token"] = "[]"
        df["hd_token_count"] = 0

    # 5) 保存
    _save_table(df, output_path)
    return df

# ---------------- 示例调用 ----------------
if __name__ == "__main__":
    # 假设本地 tokenizer 在 ./tokenizer/
    # 输入表中包含列 "LP" 和 "HD"
    tokenize_lp_hd_columns(
        input_path="./output/news_all_years.xlsx",
        tokenizer_path="./model",
        output_path="./output/news_with_lp_hd_tokens.xlsx",
        lp_col="LP",
        hd_col="HD",
        # 如需统一长度，取消下面两行注释并设置长度
        # padding="max_length",
        # truncation=True, max_length=256,
    )
