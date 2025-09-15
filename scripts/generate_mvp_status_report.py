#!/usr/bin/env python3
"""
Generate a client-facing Week 6 MVP status PDF with a curated, concise 5-page
layout: title, phase timeline, Week 5–6 deliverables + metrics, a tasteful
visuals page (≤3 charts with captions), and next steps.
"""

from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, Preformatted
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
import json


def quick_csv_rows(path: Path) -> int:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # subtract header if present
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return 0


def safe_img(path: Path, width: float, height: float):
    if path.exists():
        img = Image(str(path))
        img.drawWidth = width
        img.drawHeight = height
        img.hAlign = 'CENTER'
        return img
    return Paragraph(f"[Missing image: {path}]")


def main():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6, alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], leading=14))
    styles.add(ParagraphStyle(name="BodyJustify", parent=styles["Normal"], leading=14, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor('#475569'), alignment=TA_CENTER, spaceBefore=4, spaceAfter=6))
    styles.add(ParagraphStyle(name="CodeSmall", parent=styles["Code"], fontName='Courier', fontSize=8, leading=10))

    out_path = Path("LEO_MVP_Status_Week6.pdf")
    # tighter margins to reduce empty space
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=0.5*inch, rightMargin=0.5*inch, topMargin=0.55*inch, bottomMargin=0.55*inch)

    story = []

    # Page 1 — Title + Executive Summary
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("LEO Satellite Communication — MVP Status Report (Week 6)", styles["TitleCenter"]))
    story.append(Paragraph(datetime.now().strftime("%B %d, %Y"), styles["Body"]))
    story.append(Spacer(1, 0.18*inch))
    story.append(Paragraph(
        "This report summarizes progress through Week 6 of Phase 1. We completed the unified data\n"
        "pipeline (cleaning, smoothing, feature engineering, dataset A/B split), produced analysis\n"
        "visuals, and authored a Kaggle-ready training notebook for the dual-AI core (Predictive LSTM\n"
        "and LSTM Autoencoder). Training is underway with cloud compute.", styles["BodyJustify"]))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph(
        "Impact so far: higher-quality training data, clear KPIs/visual validation, and groundwork\n"
        "for an MVP that proactively steers traffic and flags critical anomalies.", styles["BodyJustify"]))
    story.append(Spacer(1, 0.2*inch))
    # immediately follow with phases table to keep page density
    story.append(Paragraph("Project Phases", styles["H2"]))

    # Page 2 — Phase timeline table
    # (header injected above)
    # Build wrapped cells using Paragraphs so long text flows over two lines
    def cell(text: str):
        return Paragraph(text, styles['Body'])

    phase_header = [cell("Phase"), cell("Title"), cell("Duration"), cell("Objective"), cell("Status")]
    phase_rows = [
        [cell("1"), cell("Foundational Analysis & MVP Development"), cell("8 Weeks"), cell("Simulator + dual-AI core MVP"), cell("In Progress (Week 6)")],
        [cell("2"), cell("Advanced Modeling & Simulated Stabilization"), cell("12 Weeks"), cell("Enhance models, playbook, RL research"), cell("Planned")],
        [cell("3"), cell("Real-World Integration & Edge Deployment"), cell("16 Weeks"), cell("Integrate hardware, edge inference, cloud arch"), cell("Planned")],
        [cell("4"), cell("Full-Scale Stabilization & Commercialization"), cell("Ongoing"), cell("Operate at scale, refine with live data"), cell("Planned")],
    ]
    phase_data = [phase_header] + phase_rows
    # Fit within frame width; allow double-line wrapping
    phase_table = Table(phase_data, colWidths=[0.55*inch, 2.25*inch, 0.9*inch, 2.25*inch, 0.95*inch])
    phase_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f4ed8')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.HexColor('#eef2ff')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    phase_table.hAlign = 'CENTER'
    story.append(phase_table)
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Status: Phase 1 on track. Data pipeline + visualization delivered; notebook created; datasets uploaded; training in progress.", styles["Body"]))
    # Allow natural flow to next page without forced break

    # Page 3 — Week 5–6 deliverables + metrics
    story.append(Paragraph("Week 1–6: Progress To Date", styles["H2"]))
    # KPI snapshot
    a_path = Path('data/processed/hybrid_v1_dataset_a.csv')
    b_path = Path('data/processed/hybrid_v1_dataset_b.csv')
    kpis = [
        ["KPI", "Value"],
        ["Phase 1 progress", "~75% (Week 6/8)"],
        ["Dataset A rows", f"{quick_csv_rows(a_path):,}" if a_path.exists() else "—"],
        ["Dataset B rows", f"{quick_csv_rows(b_path):,}" if b_path.exists() else "—"],
        ["Visualization outputs", "analysis_a/analysis_b charts generated" if Path('data/processed/analysis_b').exists() else "—"],
        ["Training notebook", "dual_ai_training.ipynb (Kaggle-ready)" if Path('notebooks/dual_ai_training.ipynb').exists() else "—"],
    ]
    kpi_table = Table(kpis, colWidths=[2.3*inch, 4.1*inch])
    kpi_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f766e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    kpi_table.hAlign = 'CENTER'
    story.append(kpi_table)
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("Week 5–6: Dual AI Model Training — Deliverables", styles["H2"]))
    story.append(Paragraph(
        "• Predictive LSTM: predicts current_optimal_path 60s ahead (Dataset A).<br/>"
        "• LSTM Autoencoder: trained on normal-only data (Dataset B) for anomaly detection.<br/>"
        "• Validation: classifier accuracy on held-out test; autoencoder reconstruction-error separation.",
        styles["Body"]))

    # Model metrics if available (from Kaggle run)
    metrics_rows = [["Artifact", "Value"]]
    for fn in [Path('predictive_metrics.json'), Path('anomaly_metrics.json')]:
        if fn.exists():
            try:
                data = json.loads(fn.read_text())
                for k, v in data.items():
                    metrics_rows.append([f"{fn.stem}:{k}", str(v)])
            except Exception:
                pass
    if len(metrics_rows) == 1:
        metrics_rows.extend([
            ["predictive_metrics:test_accuracy", "—"],
            ["anomaly_metrics:train_99pct_threshold", "—"],
        ])
    t = Table(metrics_rows, colWidths=[3.2*inch, 3.2*inch])
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0ea5e9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    story.append(t)
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Interpretation: Accuracy measures proactive path selection skill; reconstruction-error threshold distinguishes normal vs anomaly windows.", styles["Body"]))
    story.append(Spacer(1, 0.12*inch))
    # Milestone narrative
    story.append(Paragraph("Milestones Completed (W1–W6)", styles["H2"]))
    story.append(Paragraph(
        "• W1–W2: Hybrid simulator with predictable events + injected failures.\n"
        "• W3–W4: Unified data pipeline; smoothing, feature engineering; A/B datasets produced.\n"
        "• W5: Visualization & analytics; Kaggle datasets uploaded.\n"
        "• W6: Training notebook authored; model training running on Kaggle.", styles["Body"]))
    # No forced page break; keep dense until we start visuals

    # Page 4 — Curated visuals (≤3 charts)
    story.append(Paragraph("Key Visuals", styles["H2"]))
    # Candidate paths (prefer B as it's normal-only)
    candidates = [
        Path('data/processed/analysis_b/optimal_path_share.png'),
        Path('data/processed/analysis_b/quality_cost_timeseries.png'),
        Path('data/processed/analysis_b/correlation_matrix.png'),
        Path('data/processed/analysis_a/timeseries_latency.png'),
    ]
    charts = [p for p in candidates if p.exists()][:3]
    rows = []
    for i in range(0, len(charts), 2):
        row_imgs = []
        for p in charts[i:i+2]:
            # Use conservative sizes to avoid layout overflow; keep images simple without KeepTogether
            row_imgs.append(safe_img(p, 3.0*inch, 1.8*inch))
        if len(row_imgs) == 1:
            row_imgs.append(Paragraph("", styles['Caption']))
        rows.append(row_imgs)
    if rows:
        table = Table(rows, colWidths=[3.1*inch, 3.1*inch])
        table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP')]))
        story.append(table)
    story.append(Spacer(1, 0.1*inch))
    for p in charts[:2]:
        story.append(Paragraph(p.name.replace('_',' ').replace('.png','').title(), styles['Caption']))
    story.append(Paragraph("These visuals summarize path-share balance, cost dynamics over time, and inter-metric correlations that inform feature weighting.", styles["Body"]))
    story.append(PageBreak())

    # Page 4 & 5 — Code highlights (rendered, not images)
    def read_snippet(file_path: Path, start_token: str, end_token: str, max_lines: int = 120) -> str:
        try:
            text = file_path.read_text(encoding='utf-8', errors='ignore').splitlines()
            start_idx = 0
            end_idx = min(len(text), max_lines)
            for i, line in enumerate(text):
                if start_token in line:
                    start_idx = i
                    break
            for j in range(start_idx + 1, len(text)):
                if end_token in text[j]:
                    end_idx = j + 1
                    break
            snippet = "\n".join(text[start_idx:end_idx])
            return snippet[:4000]
        except Exception:
            return f"# Unable to load {file_path}"

    story.append(Paragraph("Code Highlights — Data Pipeline Core", styles["H2"]))
    dp_file = Path('src/utils/data_processor.py')
    dp_snippet = read_snippet(dp_file, 'def process', 'return {') if dp_file.exists() else '# data_processor.py not found'
    story.append(Preformatted(dp_snippet, styles['CodeSmall']))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("This function orchestrates loading, cleaning, smoothing, feature engineering, dataset split, and export to CSV.", styles['Body']))
    story.append(PageBreak())

    story.append(Paragraph("Code Highlights — CLI & Visualization", styles["H2"]))
    cli_file = Path('scripts/build_datasets.py')
    vis_file = Path('scripts/analyze_and_visualize.py')
    cli_snippet = read_snippet(cli_file, 'def main', 'outputs') if cli_file.exists() else '# build_datasets.py not found'
    vis_snippet = read_snippet(vis_file, 'def main', 'Wrote report') if vis_file.exists() else '# analyze_and_visualize.py not found'
    story.append(Paragraph("Dataset Builder (excerpt)", styles['Caption']))
    story.append(Preformatted(cli_snippet, styles['CodeSmall']))
    story.append(Spacer(1, 0.08*inch))
    story.append(Paragraph("Visualization Runner (excerpt)", styles['Caption']))
    story.append(Preformatted(vis_snippet, styles['CodeSmall']))

    # Page 5 — Risks, Mitigations & Next Steps
    story.append(Paragraph("Risks & Mitigations", styles["H2"]))
    story.append(Paragraph(
        "• Data drift: Mitigation — nightly re-training window and drift monitors.\n"
        "• Class imbalance (predictive labels): Mitigation — weighted loss, focal loss option.\n"
        "• False positives in anomaly detection: Mitigation — threshold calibration on validation and\n"
        "  guardrails (hysteresis) in remediation logic.", styles["BodyJustify"]))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Next Steps (Week 7–8): MVP Integration & Demo", styles["H2"]))
    story.append(Paragraph(
        "• Integrate both models into a Streamlit MVP dashboard with live simulation feed.\n"
        "• Implement remediation playbook: proactive switch + critical quarantine alerts.\n"
        "• Package artifacts (models, scaler), publish inference functions, and finalize demo script.", styles["BodyJustify"]))

    doc.build(story)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()


