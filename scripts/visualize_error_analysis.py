# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Error Analysis for RAG Submissions")
    # ปรับเปลี่ยน Path เริ่มต้นตามที่คุณระบุ
    parser.add_argument("--oracle", type=str, default="ai_studio_code.csv", help="Path to ground truth (Oracle) CSV")
    parser.add_argument("--pred", type=str, default="submissions/submission_test.csv", help="Path to predicted submission CSV")
    parser.add_argument("--outdir", type=str, default="./visualizations", help="Output directory for plots")
    args = parser.parse_args()

    oracle_path = Path(args.oracle)
    pred_path = Path(args.pred)
    outdir = Path(args.outdir)

    print(f"🔍 Starting Error Analysis Dashboard Generation...")
    print(f"Ground Truth (Oracle): {oracle_path}")
    print(f"Prediction (Model): {pred_path}")

    if not oracle_path.exists():
        print(f"❌ Error: ไม่พบไฟล์ Ground truth ที่ {oracle_path}")
        return
    if not pred_path.exists():
        print(f"❌ Error: ไม่พบไฟล์ Prediction ที่ {pred_path}")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    # Load Data
    df_oracle = pd.read_csv(oracle_path).rename(columns={"answer": "oracle_ans"})
    df_pred = pd.read_csv(pred_path).rename(columns={"answer": "pred_ans"})

    # Merge on ID
    try:
         df = pd.merge(df_oracle, df_pred, on='id', how='inner')
    except KeyError:
         print("❌ Error: ทั้งสองไฟล์ต้องมีคอลัมน์ 'id' และ 'answer'")
         return

    total = len(df)
    df['is_correct'] = df['oracle_ans'] == df['pred_ans']
    correct = df['is_correct'].sum()
    accuracy = correct / total

    # --- Setup Dashboard Figure ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(f"RAG Error Analysis Dashboard\nModel: {pred_path.name} | Overall Accuracy: {accuracy*100:.1f}%", 
                 fontsize=24, fontweight='bold', y=0.98)

    # =============== 1. Confusion Matrix (ความแม่นยำรายข้อ) ===============
    ax1 = plt.subplot(1, 3, 1)
    classes = list(range(1, 11))
    cm = pd.crosstab(df['oracle_ans'], df['pred_ans'])
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False, ax=ax1,
                annot_kws={'size': 12, 'weight': 'bold'}, linewidths=1, linecolor='#f0f0f0')
    ax1.set_title("1. Confusion Matrix (Actual vs Predicted)", fontsize=16, pad=15, fontweight='bold')
    ax1.set_ylabel('Actual Label (Oracle)', fontsize=14)
    ax1.set_xlabel('Predicted Label (Model)', fontsize=14)

    # =============== 2. Accuracy per Class (จุดอ่อนของโมเดล) ===============
    ax2 = plt.subplot(1, 3, 2)
    class_acc = []
    for c in classes:
        class_subset = df[df['oracle_ans'] == c]
        acc = class_subset['is_correct'].mean() * 100 if len(class_subset) > 0 else 0
        class_acc.append({"Class": c, "Accuracy (%)": acc, "Total": len(class_subset)})
    
    df_class = pd.DataFrame(class_acc)
    bars = sns.barplot(data=df_class, x="Class", y="Accuracy (%)", palette="magma", ax=ax2)
    ax2.set_title("2. Accuracy per Class", fontsize=16, pad=15, fontweight='bold')
    ax2.set_ylim(0, 115)
    
    # Add labels on top of bars
    for i, p in enumerate(ax2.patches):
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width()/2., height + 2,
                f"{height:.0f}%\n(n={df_class['Total'].iloc[i]})", 
                ha="center", va="bottom", fontsize=10, fontweight='bold')

    # =============== 3. Error Insight (วิเคราะห์คู่ที่ผิดบ่อย) ===============
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    ax3.set_title("3. Top Misclassified Insights", fontsize=16, pad=15, fontweight='bold')
    
    errors = df[~df['is_correct']].copy()
    if errors.empty:
        text_content = "🎉 PERFECT SCORE!\nNo misclassifications found."
    else:
        # วิเคราะห์คู่ที่สับสนบ่อยที่สุด
        errors['error_pair'] = errors.apply(lambda r: f"Model Answer {r['pred_ans']} แต่ความจริงคือ {r['oracle_ans']}", axis=1)
        top_errors = errors['error_pair'].value_counts().head(5)
        
        text_content = "Top 5 Frequent Mistakes:\n\n"
        for i, (pair, count) in enumerate(top_errors.items(), 1):
             text_content += f"{i}. {pair}\n    (พบ {count} ครั้ง)\n\n"
             
        # สรุปปัญหาที่พบ
        text_content += "------------------------------------------\n"
        text_content += f"Total Errors: {len(errors)} out of {total}\n"
        text_content += f"Most problematic class: {df_class.loc[df_class['Accuracy (%)'].idxmin(), 'Class']}\n"

    ax3.text(0.05, 0.95, text_content, fontsize=13, va='top', ha='left', 
             bbox=dict(boxstyle="round,pad=1", facecolor="#fff9f9", edgecolor="#ffcccc", linewidth=2),
             fontfamily='monospace')

    # --- Save and Show ---
    plt.tight_layout(pad=4.0)
    dashboard_path = outdir / "error_analysis_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ วิเคราะห์เสร็จสิ้น!")
    print(f"📊 Dashboard ถูกบันทึกไว้ที่: {dashboard_path}")
    print(f"📈 Overall Accuracy: {accuracy*100:.2f}%")
    
    plt.show()

if __name__ == "__main__":
    main()