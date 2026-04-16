# Analisador de FIIs para Hugging Face Spaces

Aplicação em Python para analisar uma lista de FIIs e indicar, de forma **quantitativa**, qual ativo apresenta o melhor score no momento da execução.

## Ferramentas usadas

- **pandas**: organização e tratamento de dados
- **numpy**: cálculos numéricos
- **matplotlib**: gráficos de comparação e evolução de preços
- **yfinance**: coleta de dados de mercado
- **gradio**: interface web para rodar no Hugging Face Spaces

## Como funciona o score

O ranking combina quatro fatores:

1. Dividend Yield (maior melhor)
2. P/VP (mais próximo de 1, com penalização para sobrepreço)
3. Liquidez média em reais (maior melhor)
4. Volatilidade anualizada (menor melhor)

## Rodar localmente

```bash
pip install -r requirements.txt
python app.py
```

## Deploy no Hugging Face Spaces

1. Crie um Space do tipo **Gradio**.
2. Suba os arquivos deste repositório (`app.py`, `requirements.txt`, `README.md`).
3. O Space instalará as dependências e iniciará automaticamente.

## Observação importante

Este projeto é educacional e não representa recomendação de investimento personalizada.
