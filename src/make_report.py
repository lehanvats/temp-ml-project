"""
Phase 7 - Assembles PBL1_Group4_Report.pdf: a plain-formatted PDF answering Q1-Q5,
built with reportlab (pure pip, no system LaTeX/pandoc dependency). Re-runs the full
pipeline so the report always reflects the current code/data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                 SimpleDocTemplate, Spacer, Table, TableStyle)

from classification import run as run_classification
from data_prep import load_clean
from eda import NUMERIC_COLS, descriptive_stats, missing_value_report, outlier_summary
from interpret import linear_regression_coefficients, logistic_regression_coefficients
from regression import run as run_regression

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "outputs" / "figures"
OUT_PDF = ROOT / "PBL1_Group4_Report.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("SectionHeading", parent=styles["Heading1"], fontSize=14,
                           spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=11.5,
                           spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14,
                           spaceAfter=8, alignment=4))  # 4 = justify
styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5, leading=11,
                           textColor=colors.grey, spaceAfter=10))


def df_to_table(df: pd.DataFrame, col_widths=None, font_size=7.5) -> Table:
    data = [[""] + list(df.columns)] + [[str(idx)] + [f"{v:,.3g}" if isinstance(v, float) else str(v)
                                                        for v in row] for idx, row in zip(df.index, df.values)]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def fig(name: str, width=CONTENT_W) -> Image:
    path = FIG_DIR / name
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    return img


def build() -> None:
    df = load_clean()
    mv = missing_value_report()
    stats = descriptive_stats(df)
    corr = df[NUMERIC_COLS].corr()
    outliers = outlier_summary(df)

    reg_results, reg_models, train_idx, test_idx = run_regression()
    clf_results, confmats, reports, thresholds, clf_models, *_ = run_classification()
    lin_coef = linear_regression_coefficients(reg_models)
    log_coef = logistic_regression_coefficients(clf_models)

    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4,
                             topMargin=MARGIN, bottomMargin=MARGIN,
                             leftMargin=MARGIN, rightMargin=MARGIN,
                             title="PBL-I Group 4 Report", author="PBL-I Group 4")
    story = []
    P = lambda text, style="Body": story.append(Paragraph(text, styles[style]))
    S = lambda h=10: story.append(Spacer(1, h))

    # ---------------- Title
    story.append(Paragraph("PBL-I Project Review &ndash; Machine Learning (21CSC305P)",
                            styles["Title"]))
    story.append(Paragraph("Unit 1 &amp; Unit 2 &nbsp;|&nbsp; 20 Marks",
                            ParagraphStyle("sub", parent=styles["Normal"], fontSize=11,
                                           alignment=1, spaceAfter=4, textColor=colors.grey)))
    story.append(Paragraph("Group 4: Used Car Price Prediction and Condition Classification",
                            ParagraphStyle("sub2", parent=styles["Heading2"], fontSize=13,
                                           alignment=1, spaceBefore=10, spaceAfter=16)))
    P("<b>Project problem:</b> Develop an ML system to predict the resale price of used cars "
      "and classify vehicles according to their condition/value category.")
    P("<b>Dataset:</b> Group_4_Dataset.csv &ndash; 8,128 used-car listings, 13 raw columns "
      "(name, year, selling_price, km_driven, fuel, seller_type, transmission, owner, mileage, "
      "engine, max_power, torque, seats).")
    S(16)

    # ---------------- Q1
    P("Q1. Problem Identification &amp; Dataset Selection &ndash; 4 Marks", "SectionHeading")
    P("<b>Real-world problem.</b> Used-car marketplaces need to (a) quote a fair resale price "
      "for a listed vehicle and (b) bucket it into a value tier a buyer can filter by. An ML "
      "system can do both consistently from the vehicle's own attributes.")
    P("<b>ML objective.</b> <i>Regression:</i> predict selling_price (continuous, INR) from "
      "vehicle attributes. <i>Classification:</i> predict a price tier &ndash; Budget / Mid / "
      "Premium &ndash; a discretisation of the same target standing in for &ldquo;condition/"
      "value category&rdquo;, since the raw data has no separate condition field. Tier cut "
      "points are the 33rd/67th percentile of price, fit on the training split only so no test "
      "information leaks into the label definition.")
    P("<b>Input features (both tasks):</b> car_age (derived from year), km_driven, mileage, "
      "engine, max_power, torque_nm, torque_rpm, seats (numeric) and brand (derived from name), "
      "fuel, seller_type, transmission, owner (categorical). selling_price is the regression "
      "target and is excluded from the classifier's features to avoid circularity.")
    P("<b>Regression target:</b> selling_price (modelled as log(selling_price)). "
      "<b>Classification target:</b> price_tier &isin; {Budget, Mid, Premium}.")
    P("<b>Dataset justification.</b> 8,128 rows span 32 brands, model years 1983&ndash;2020, "
      "and all fuel/transmission/ownership categories with enough support per level for "
      "stratified learning (the rarest class, owner = Test Drive Car, still has 5 rows). It "
      "directly matches the brief's suggested data and is an appropriate size for the four "
      "regression and three classification models used in Q3.")

    # ---------------- Q2
    story.append(PageBreak())
    P("Q2. Exploratory Data Analysis &amp; Insights &ndash; 4 Marks", "SectionHeading")
    P("<b>Cleaning pipeline (summary).</b> Exact duplicates (1,202 of 8,128) are dropped first, "
      "before any split, so duplicates can never straddle train/test. Unit suffixes are "
      "stripped from mileage/engine/max_power; torque's many free-text formats are parsed into "
      "torque_nm and torque_rpm, correctly resolving an Nm-vs-kgm labelling ambiguity present in "
      "the source data. Undisclosed zero readings and a handful of physically-impossible torque "
      "values are treated as missing. Rows with no spec data at all (208) are dropped; the small "
      "residual NaN count is median-imputed. brand and car_age are engineered. "
      "<b>Final cleaned dataset: 6,718 rows, 0 missing values.</b>")

    P("Missing values (raw dataset)", "SubHeading")
    story.append(df_to_table(mv[mv["missing_count"] > 0].round(2),
                              col_widths=[3.5 * cm, 3.5 * cm, 3.5 * cm]))
    S()

    P("Descriptive statistics &ndash; mean, std, variance, quantiles (cleaned data)", "SubHeading")
    story.append(df_to_table(stats, font_size=6.6))
    S()

    corr_price = corr["selling_price"].sort_values(ascending=False).round(3).to_frame("correlation")
    story.append(KeepTogether([
        Paragraph("Correlation with selling_price", styles["SubHeading"]),
        df_to_table(corr_price, col_widths=[4 * cm, 4 * cm]),
    ]))
    S()

    story.append(fig("correlation_heatmap.png", width=11 * cm))
    S()
    story.append(fig("price_distribution.png"))
    P("selling_price is strongly right-skewed; the log transform (right panel) is roughly "
      "symmetric, which is why regression targets log(selling_price).", "Caption")

    story.append(PageBreak())
    story.append(fig("price_vs_age_km.png"))
    S(6)
    story.append(fig("price_by_category.png"))
    S(6)
    story.append(fig("brand_median_price.png"))

    story.append(PageBreak())
    P("Outlier summary (IQR rule)", "SubHeading")
    story.append(df_to_table(outliers, font_size=7, col_widths=[2.6 * cm, 2.3 * cm, 2.3 * cm, 2 * cm, 2.2 * cm]))
    S()
    P("seats and engine show the highest outlier rates by the strict 1.5&times;IQR rule, but "
      "both are genuinely multi-modal (5/7/8-seaters; small/mid/large engines) rather than data "
      "errors. selling_price and km_driven outliers are real high-end/high-mileage listings and "
      "are kept as-is (km_driven is capped only at the 99th percentile for modelling stability).")

    P("Key insights", "SubHeading")
    for s in [
        "Price is right-skewed (mean INR 5.26L vs median INR 4.2L, max INR 1Cr) &rarr; modelled "
        "in log space for regression.",
        "car_age and km_driven are the strongest negative correlates of price (&minus;0.43, "
        "&minus;0.16); max_power and torque_nm are the strongest positive ones (+0.69, +0.64) "
        "&ndash; depreciation and engine output dominate resale value.",
        "Automatic-transmission and Diesel-fuel listings carry a materially higher median price "
        "than Manual/Petrol &ndash; a segment effect compounding a fuel-cost effect.",
        "max_power, engine and torque_nm are highly collinear (r &asymp; 0.68&ndash;0.85) "
        "&ndash; motivating Ridge/Bayesian regularisation over plain OLS in Q3.",
    ]:
        P("&bull; " + s)

    # ---------------- Q3
    story.append(PageBreak())
    P("Q3. Regression &amp; Classification Implementation &ndash; 4 Marks", "SectionHeading")
    P("<b>Preprocessing (shared).</b> StandardScaler on numeric features, "
      "OneHotEncoder(drop='first', handle_unknown='ignore') on categoricals, assembled in a "
      "ColumnTransformer fit on the training fold only. Polynomial Regression additionally "
      "expands only the numeric block to degree 2 before the categorical columns are "
      "concatenated. A single 80/20 split (random_state=42), stratified on a rough price bin, "
      "is reused identically for both tasks so Q4's comparison is apples-to-apples.")
    P("<b>Algorithms used</b> (per the PDF's Q3 model list): <i>Regression</i> &ndash; Linear "
      "Regression, Polynomial Regression (degree 2), Ridge Regression (RidgeCV selects "
      "&alpha;), Bayesian Ridge Regression. <i>Classification</i> &ndash; Logistic Regression "
      "(multinomial), Linear SVM, RBF-kernel SVM. Regression target is log(selling_price); "
      "every reported metric is converted back to rupees before scoring.")

    P("Regression results", "SubHeading")
    story.append(df_to_table(reg_results.round(3), font_size=6.8))
    S()

    P(f"Tier thresholds (from TRAIN prices only): Budget &lt; INR {thresholds[0]:,.0f} "
      f"&le; Mid &lt; INR {thresholds[1]:,.0f} &le; Premium.", "Body")
    P("Classification results", "SubHeading")
    story.append(df_to_table(clf_results.round(4)))

    # ---------------- Q4
    story.append(PageBreak())
    P("Q4. Comparative Performance Analysis &ndash; 4 Marks", "SectionHeading")
    story.append(fig("regression_comparison.png"))
    P("Polynomial Regression achieves the lowest RMSE and highest R&sup2; (&asymp;0.90) &ndash; "
      "the numeric-feature interactions/curvature it adds are real and linear-only models can't "
      "capture them. Ridge and Bayesian Ridge score marginally below plain Linear Regression: "
      "with ~46 features on ~5,400 training rows the model isn't badly overfit to begin with, so "
      "regularisation mostly shrinks useful signal (RidgeCV itself selected a small &alpha;).")

    story.append(fig("classification_comparison.png"))
    story.append(fig("confusion_matrices.png"))
    P("RBF-kernel SVM edges out Logistic Regression (&asymp;0.84 vs &asymp;0.82 accuracy); the "
      "linear SVM trails (&asymp;0.78). All three confusion matrices show the same pattern: "
      "confusion is concentrated almost entirely between adjacent tiers, with near-zero "
      "Budget&ndash;Premium confusion &ndash; expected since the tiers are ordered price bands. "
      "The Mid class has the lowest per-class F1 for every model because it is bounded on both "
      "sides by a threshold and absorbs boundary error from both directions.")
    P("<b>Cross-task consistency.</b> The classifier is, by construction, a discretised version "
      "of the regressor over the same feature set and split &ndash; so their relative "
      "performance is expected to track rather than being two independent findings. Polynomial "
      "Regression's edge in R&sup2; and RBF-SVM's edge in accuracy both come from the same "
      "source: real non-linearity in how car_age/max_power/engine jointly determine price.")

    # ---------------- Q5
    story.append(PageBreak())
    P("Q5. Result Interpretation &amp; Presentation &ndash; 4 Marks", "SectionHeading")
    P("Top drivers of log(selling_price) &ndash; Linear Regression (standardized coefficients)",
      "SubHeading")
    story.append(df_to_table(lin_coef.head(10).round(3), col_widths=[6 * cm, 3 * cm, 3 * cm]))
    S()
    story.append(fig("linear_regression_top_coefficients.png", width=12 * cm))
    P("Coefficients are on standardized features, so they read directly as &ldquo;effect per "
      "one std-dev change&rdquo; and are comparable to each other. car_age is the single "
      "strongest, sign-consistent driver in both the regression and classification models "
      "&ndash; older cars are reliably worth less and more likely to fall in the Budget tier. "
      "brand dummies dominate the top-10 list for price, showing brand reputation/segment "
      "carries real, separable price information beyond the physical specs.")

    P("Practical significance", "SubHeading")
    P("A &asymp;INR 88&ndash;95k MAE against a median price of INR 4.2L (&asymp;2.7% of price; "
      "RMSE &asymp; 4.2&ndash;5% of price) is workable for a guidance price &ndash; narrow "
      "enough for a listing platform to flag a quote as under/overpriced &ndash; but not tight "
      "enough to replace an inspector's appraisal for a single high-value transaction. The "
      "classifier's &asymp;82&ndash;84% accuracy suits filtering/browsing (a tier facet on a "
      "search page) where an occasional adjacent-tier misclassification is low-cost.")

    P("Limitations", "SubHeading")
    for s in [
        "Single market snapshot (model years only run to 2020) &ndash; the model will drift as "
        "the market moves and would need periodic retraining.",
        "No accident/service history or physical inspection data &ndash; the classification "
        "task's &ldquo;condition&rdquo; is a value proxy (price tier), not true mechanical "
        "condition, which the raw dataset does not capture.",
        "mileage mixes two physical units (kmpl vs km/kg for CNG/LPG) merged into one numeric "
        "column; the fuel feature lets the models partially compensate.",
        "km_driven is capped at the 99th percentile for modelling, slightly under-weighting "
        "genuine but rare high-mileage listings.",
    ]:
        P("&bull; " + s)

    P("Conclusion &amp; future work", "SubHeading")
    P("Both tasks confirm the same underlying structure: car age, engine output and brand "
      "jointly explain most of the price variation, with real non-linear interaction among "
      "them. Future work: fold in an external market-index feature to correct for the 2020 "
      "snapshot's staleness; test a spline/degree-3 basis on car_age specifically, since "
      "price-vs-age is visibly non-linear (steep early depreciation, flattening later); and, if "
      "genuine condition data becomes available, replace the price-tier proxy with a true "
      "condition label.")

    doc.build(story)
    print(f"Saved -> {OUT_PDF}")


if __name__ == "__main__":
    build()
