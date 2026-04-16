"""App para análise de FIIs com foco em execução no Hugging Face Spaces.

Ferramentas utilizadas:
- pandas (tratamento de dados)
- numpy (cálculos numéricos)
- matplotlib (visualização)
- yfinance (coleta de dados)
- gradio (interface web)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_FIIS = "HGLG11, KNRI11, MXRF11, XPLG11, VISC11, XPML11, BRCO11"


@dataclass
class ScoreWeights:
    dy: float = 0.35
    pvp: float = 0.25
    liquidity: float = 0.20
    volatility: float = 0.20


def _normalize_series(values: pd.Series, reverse: bool = False) -> pd.Series:
    """Normaliza série para [0, 1]. Se reverse=True, inverte o sentido."""
    if values.empty:
        return values

    min_v = values.min()
    max_v = values.max()

    if pd.isna(min_v) or pd.isna(max_v) or math.isclose(float(min_v), float(max_v)):
        result = pd.Series(np.full(len(values), 0.5), index=values.index)
    else:
        result = (values - min_v) / (max_v - min_v)

    if reverse:
        result = 1 - result
    return result.clip(0, 1)


def _parse_tickers(raw_tickers: str) -> List[str]:
    tickers = [item.strip().upper() for item in raw_tickers.split(",") if item.strip()]
    normalized = []
    for ticker in tickers:
        normalized.append(ticker if ticker.endswith(".SA") else f"{ticker}.SA")
    return normalized


def _fii_metrics(ticker: str, period: str = "1y") -> dict:
    asset = yf.Ticker(ticker)
    history = asset.history(period=period, auto_adjust=False)

    if history.empty or len(history) < 20:
        raise ValueError("Histórico insuficiente para análise")

    closes = history["Close"].dropna()
    returns = closes.pct_change().dropna()

    info = asset.info if isinstance(asset.info, dict) else {}

    dy = info.get("dividendYield", np.nan)
    if isinstance(dy, (int, float)) and not pd.isna(dy):
        dy = float(dy) * 100

    pvp = info.get("priceToBook", np.nan)
    avg_volume_value = float((history["Volume"] * history["Close"]).tail(60).mean())

    volatility_annual = float(returns.std() * np.sqrt(252) * 100)

    return {
        "ticker": ticker.replace(".SA", ""),
        "preco_atual": float(closes.iloc[-1]),
        "dividend_yield_pct": float(dy) if not pd.isna(dy) else np.nan,
        "p_vp": float(pvp) if not pd.isna(pvp) else np.nan,
        "liquidez_media_60d_brl": avg_volume_value,
        "volatilidade_anual_pct": volatility_annual,
        "retorno_12m_pct": float((closes.iloc[-1] / closes.iloc[0] - 1) * 100),
        "history": closes,
    }


def rankear_fiis(raw_tickers: str, pesos: ScoreWeights | None = None):
    if not raw_tickers.strip():
        raise ValueError("Informe ao menos um ticker")

    pesos = pesos or ScoreWeights()
    tickers = _parse_tickers(raw_tickers)

    metrics = []
    errors = []

    for ticker in tickers:
        try:
            metrics.append(_fii_metrics(ticker))
        except Exception as exc:  # dados externos instáveis
            errors.append(f"{ticker.replace('.SA', '')}: {exc}")

    if not metrics:
        raise ValueError("Nenhum ticker pôde ser analisado. Verifique os códigos informados.")

    df = pd.DataFrame(metrics)

    # Regras de fallback para valores ausentes
    for col in ["dividend_yield_pct", "p_vp", "liquidez_media_60d_brl", "volatilidade_anual_pct"]:
        if df[col].isna().all():
            df[col] = 0
        else:
            df[col] = df[col].fillna(df[col].median())

    dy_score = _normalize_series(df["dividend_yield_pct"])

    # P/VP: ideal próximo de 1 com leve preferência abaixo de 1
    pvp_distance = (df["p_vp"] - 1.0).abs()
    pvp_base = 1 - _normalize_series(pvp_distance)
    penalty_overpriced = (df["p_vp"] > 1.15).astype(float) * 0.15
    pvp_score = (pvp_base - penalty_overpriced).clip(0, 1)

    liquidity_score = _normalize_series(df["liquidez_media_60d_brl"])
    volatility_score = _normalize_series(df["volatilidade_anual_pct"], reverse=True)

    df["score_final"] = (
        dy_score * pesos.dy
        + pvp_score * pesos.pvp
        + liquidity_score * pesos.liquidity
        + volatility_score * pesos.volatility
    )

    df = df.sort_values("score_final", ascending=False).reset_index(drop=True)

    best = df.iloc[0]
    mensagem = (
        f"✅ Recomendação quantitativa: **{best['ticker']}** (score {best['score_final']:.3f}).\n\n"
        f"Justificativa: melhor combinação entre DY, P/VP, liquidez e volatilidade "
        f"no conjunto informado."
    )

    if errors:
        mensagem += "\n\n⚠️ Tickers com falha na coleta: " + "; ".join(errors)

    return df, mensagem, metrics


def gerar_grafico_scores(df: pd.DataFrame):
    top = df.head(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top["ticker"], top["score_final"], color="#1f77b4")
    ax.set_title("Top 5 FIIs por Score")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Score (0-1)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig


def gerar_grafico_preco(metrics: Iterable[dict], best_ticker: str):
    serie = None
    for item in metrics:
        if item["ticker"] == best_ticker:
            serie = item["history"]
            break

    if serie is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Sem histórico para plot", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(serie.index, serie.values, color="#2ca02c")
    ax.set_title(f"Evolução de preço - {best_ticker} (12m)")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço (R$)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    return fig


def analisar(raw_tickers: str):
    df, mensagem, metrics = rankear_fiis(raw_tickers)

    df_saida = df[[
        "ticker",
        "preco_atual",
        "dividend_yield_pct",
        "p_vp",
        "liquidez_media_60d_brl",
        "volatilidade_anual_pct",
        "retorno_12m_pct",
        "score_final",
    ]].copy()

    df_saida.columns = [
        "Ticker",
        "Preço Atual (R$)",
        "Dividend Yield (%)",
        "P/VP",
        "Liquidez Média 60d (R$)",
        "Volatilidade Anual (%)",
        "Retorno 12m (%)",
        "Score Final",
    ]

    fig_scores = gerar_grafico_scores(df)
    best_ticker = df.iloc[0]["ticker"]
    fig_preco = gerar_grafico_preco(metrics, best_ticker)

    return mensagem, df_saida.round(4), fig_scores, fig_preco


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Analisador de FIIs") as demo:
        gr.Markdown(
            """
            # 📈 Analisador de FIIs (Python + Pandas + NumPy + Matplotlib + yfinance)
            Informe os tickers separados por vírgula para obter um ranking quantitativo.

            **Aviso:** este app é educacional e **não constitui recomendação financeira profissional**.
            """
        )

        input_tickers = gr.Textbox(
            label="Tickers de FIIs",
            value=DEFAULT_FIIS,
            placeholder="Ex: HGLG11, KNRI11, MXRF11",
        )
        btn = gr.Button("Analisar")

        output_msg = gr.Markdown(label="Recomendação")
        output_table = gr.Dataframe(label="Ranking")
        output_chart_1 = gr.Plot(label="Top 5 por score")
        output_chart_2 = gr.Plot(label="Evolução do melhor FII")

        btn.click(
            fn=analisar,
            inputs=[input_tickers],
            outputs=[output_msg, output_table, output_chart_1, output_chart_2],
        )

    return demo


app = build_app()


if __name__ == "__main__":
    app.launch()
