"""Gera o gráfico do histórico agregado preservado no portfólio."""

from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

RAIZ = Path(__file__).resolve().parents[1]


def main():
    dados = json.loads((RAIZ / "resultados/historico.json").read_text(encoding="utf-8"))
    candidatos = dados["candidatos"]
    data = lambda valor: datetime.fromisoformat(valor)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, barras) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.patch.set_facecolor("#101820")
    texto, grade = "#E8EFF5", "#34414C"
    for eixo in (ax, barras):
        eixo.set_facecolor("#101820")
        eixo.tick_params(colors=texto, labelsize=10)
        eixo.xaxis.label.set_color(texto)
        eixo.yaxis.label.set_color(texto)
        for borda in eixo.spines.values():
            borda.set_color(grade)
        eixo.grid(axis="y", color=grade, alpha=.6, linewidth=.6)
        eixo.set_axisbelow(True)
    locais = [c for c in candidatos if c["diagnostico_local"] is not None]
    publicos = sorted([c for c in candidatos if c["score_publico"] is not None], key=lambda c: c["data_publico"])
    ax.scatter([data(c["data_local"]) for c in locais], [c["diagnostico_local"] for c in locais], s=24, c="#F8D34D", alpha=.85, label="Diagnóstico local histórico", zorder=3)
    ax.scatter([data(c["data_publico"]) for c in publicos], [c["score_publico"] for c in publicos], marker="s", s=25, c="#46D79D", label="Score público", zorder=4)
    recorde, marcos, valores = -1, [], []
    for c in publicos:
        if c["score_publico"] > recorde:
            recorde = c["score_publico"]
            marcos.append(data(c["data_publico"]))
            valores.append(recorde)
    marcos.append(data(dados["data_corte"]))
    valores.append(recorde)
    ax.step(marcos, valores, where="post", color="#46D79D", linewidth=2, label="Melhor público acumulado", zorder=2)
    sem_score = [c for c in candidatos if c["diagnostico_local"] is None and c["score_publico"] is None]
    ax.scatter([data(c["data_criacao"]) for c in sem_score], [.445] * len(sem_score), marker="+", s=24, c="#82929E", alpha=.55, label="Sem score", zorder=2)
    ax.scatter([data("2026-08-29")], [.936], marker="*", s=360, color="#F075B5", edgecolor="#101820", linewidth=.8, zorder=6, label="E74 · primeiro máximo público")
    anotacoes = [
        ("2026-08-09", .595, "E48 · estatísticas de pixels\n0,595", (45, 50)),
        ("2026-08-11", .627, "E55 · supervisão soft\n0,627", (110, -60)),
        ("2026-08-29", .936, "E74 · ensemble DINOsaur\n0,936 público", (32, -72)),
    ]
    for dia, score, titulo, deslocamento in anotacoes:
        ax.annotate(titulo, (data(dia), score), xytext=deslocamento, textcoords="offset points", color=texto, fontsize=10, arrowprops={"arrowstyle": "-", "color": "#91A0AC", "lw": .8})
    ax.set_title("Evolução dos resultados", loc="left", color=texto, pad=16, fontsize=15)
    ax.set_ylabel("ROC-AUC macro")
    ax.set_xlabel("Data de avaliação · 2026", labelpad=12)
    ax.set_ylim(.42, 1.035)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    contagem = Counter(c["data_criacao"] for c in candidatos)
    inicio = min(data(c["data_criacao"]) for c in candidatos)
    fim = data(dados["data_corte"])
    dias = [inicio + timedelta(days=i) for i in range((fim - inicio).days + 1)]
    alturas = [contagem.get(d.strftime("%Y-%m-%d"), 0) for d in dias]
    barras.bar(dias, alturas, width=.82, color="#9AC8F5")
    for d, n in zip(dias, alturas):
        if n >= 20:
            barras.text(d, n + 1.3, str(n), color=texto, ha="center", fontsize=11)
    barras.set_title("Candidatos registrados por dia", loc="left", color=texto, pad=16, fontsize=15)
    barras.set_ylabel("Quantidade de IDs")
    barras.set_xlabel("Data de criação · 2026", labelpad=12)
    barras.set_ylim(0, max(alturas) * 1.23)
    barras.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    barras.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    barras.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    fig.suptitle("RSNA Knee · um laboratório de pesquisa aplicada", x=.065, y=.965, ha="left", color=texto, fontsize=23, weight="bold")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.06, .91), ncol=3, frameon=False, labelcolor=texto, fontsize=10, columnspacing=2)
    fig.text(.065, .17, "205 candidatos e variações registrados · melhor público: 0,936", color=texto, fontsize=13, weight="bold")
    fig.text(.065, .125, "Scores locais usam protocolos distintos; nem todos são OOF independente.\nIDs incluem tentativas sem score. Resultados confirmados até 23/09/2026; score privado final não documentado.", color="#AAB9C5", fontsize=9.5, linespacing=1.6)
    fig.text(.065, .05, "Curadoria: betitanx · fontes e créditos no repositório", color="#AAB9C5", fontsize=10)
    fig.text(.95, .13, "≈187 h", ha="right", color=texto, fontsize=25, weight="bold")
    fig.text(.95, .075, "execução estimada\npode haver sobreposição; trabalho manual excluído", ha="right", color="#AAB9C5", fontsize=9.5, linespacing=1.5)
    fig.subplots_adjust(left=.065, right=.95, top=.77, bottom=.27, wspace=.23)
    destino = RAIZ / "docs/imagens"
    destino.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino / "progresso.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(destino / "progresso.svg", facecolor=fig.get_facecolor())
    svg = destino / "progresso.svg"
    svg.write_text("\n".join(linha.rstrip() for linha in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    print("Gráficos gerados em docs/imagens/.")


if __name__ == "__main__":
    main()
