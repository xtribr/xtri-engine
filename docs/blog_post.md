# We Reverse-Engineered Brazil's Biggest Exam — And Matched INEP's Scores Perfectly

*How we validated an open-source IRT scoring engine against 360,000 real ENEM students and discovered that the textbook formula is wrong.*

---

## The Puzzle

Every November, about **4 million Brazilian students** sit down for ENEM — the National High School Exam. It's the biggest high-stakes test in Latin America. Your ENEM score decides whether you get into a public university, whether you qualify for scholarships, and in many cases whether you'll be the first person in your family with a college degree.

For the last 15 years, INEP (the government agency that runs ENEM) has scored the test using a sophisticated statistical model called **Three-Parameter Logistic Item Response Theory** (3PL IRT). Instead of just counting right answers, the model asks a more subtle question: *which* items did you get right?

Two students can have the same number of correct answers and get very different scores. That's not a bug — that's the whole point of the model. If Ana answered the hard items and missed the easy ones, the model is suspicious. If Beatriz answered the easy items consistently and got some hard ones too, the model rewards her consistency. It's an elegant idea, and for better or worse, it shapes the futures of millions of Brazilian teenagers every year.

But here's the thing: **nobody outside INEP had ever proven, at the individual-student level, that they could reproduce INEP's scores exactly.** Simulado platforms (commercial test-prep services that offer "ENEM-like" exams) all claim to produce "ENEM-compatible" scores. Private tutoring schools claim the same. Nobody has ever published a benchmark showing how close their implementation really is to INEP's real calculation.

We decided to find out.

---

## What We Built

Over the past couple of weeks, we built **XTRI Engine** — an open-source Python implementation of the entire ENEM scoring pipeline. It does three things:

1. **Calibration** — Takes the binary correct/incorrect response matrix of a group of students and estimates each item's difficulty, discrimination, and guessing parameters using Marginal Maximum Likelihood with Gauss-Hermite quadrature. This is the same algorithm INEP uses, just written from scratch.

2. **EAP ability estimation** — For each student, computes the Expected A Posteriori estimate of their latent ability (theta) using a standard normal prior over a 61-point numerical integration grid.

3. **Score transformation** — Converts theta to the reported 0–1000 ENEM scale.

The first two parts are textbook psychometrics — any graduate student in measurement can look up the formulas in Baker & Kim (2004). The third part is where things got interesting.

---

## The Textbook Formula Is Wrong

Every introductory IRT textbook tells you the same thing: to convert a standardized ability estimate into the ENEM scale, you multiply by 100 and add 500. So a student with theta = 1.0 gets 600 points, theta = 2.0 gets 700 points, and so on. Simple, clean, memorable. It's even on INEP's own explanatory pages. You'll find it in SAE Digital's blog, in every vestibular prep site, in every "understand your ENEM score" article ever written.

**It's wrong.**

Not hugely wrong. But meaningfully, quantifiably wrong — and wrong in a different direction for each of the four areas of the exam.

Here's what we found when we ran a regression of real ENEM scores against the computed theta for 5,000 students per area:

| Area | Real σ (not 100) | Real μ (not 500) |
|------|-----------------|------------------|
| **Languages** | 108.09 | 499.98 |
| **Humanities** | 112.31 | 501.49 |
| **Natural Sciences** | 113.10 | 501.14 |
| **Mathematics** | **129.65** | 500.02 |

Look at Mathematics. Its sigma is **129.65**, not 100. That's 30% larger than the textbook claims. This means a Mathematics student with a theta of 2.0 doesn't get 700 points — they get about 759 points. The difference between the textbook formula and reality is enormous for high performers, and it explains something that's always bugged Brazilian test-prep students: why does Mathematics regularly produce scores above 950, while Languages scores almost never break 830?

The answer is simple: **the reference population's standard deviation was different for each area**. The 2009 cohort that ENEM anchors its scale to had wider dispersion in Mathematics than in Languages. The real ENEM scale reflects that. The textbook simplification doesn't.

And the values are **stunningly stable across years**. Mathematics in 2022: 129.6473. 2023: 129.6486. 2024: 129.6484. Differences of 1 part in 10,000. INEP anchors the scale to the 2009 population and re-calibrates new items onto that anchor every year, and they do it with extraordinary precision.

---

## The Benchmark

OK, so we had a working engine and the real transformation parameters. The only question left was: **how close can we actually get to INEP's real scores?**

We ran the validation on **360,000 real ENEM students** — 30,000 per combination, across 3 years (2022, 2023, 2024) and all 4 areas (Languages, Humanities, Natural Sciences, Mathematics). Twelve combinations in total. For each student, we had:

- Their binary response vector (from the microdata published by INEP)
- Their test form (color/prova version)
- Their official INEP score (NU_NOTA_*)

We ran the XTRI Engine on the same students and compared score to score. Here's the summary:

| Year | Area | N | Correlation | MAE | Level Agreement |
|------|------|---|-------------|-----|-----------------|
| 2022 | LC | 30,000 | **1.000000** | 0.025 | 99.96% |
| 2022 | CH | 30,000 | **1.000000** | 0.025 | 99.96% |
| 2022 | CN | 30,000 | **1.000000** | 0.025 | 99.96% |
| 2022 | MT | 30,000 | **1.000000** | 0.025 | 99.96% |
| 2023 | LC | 30,000 | **1.000000** | 0.025 | 99.98% |
| 2023 | CH | 30,000 | **1.000000** | 0.025 | 99.95% |
| 2023 | CN | 30,000 | **1.000000** | 0.025 | 99.95% |
| 2023 | MT | 30,000 | **1.000000** | 0.026 | 99.96% |
| 2024 | LC | 30,000 | **1.000000** | 0.025 | 99.95% |
| 2024 | CH | 30,000 | **1.000000** | 0.025 | 99.95% |
| 2024 | CN | 30,000 | **1.000000** | 0.025 | 99.94% |
| 2024 | MT | 30,000 | **1.000000** | 0.025 | 99.99% |

**Pearson correlation of 1.000000 across all 360,000 students.** Mean absolute error of 0.025 points on a scale that goes from 0 to 1000. The 95th percentile of errors is 0.048 points.

To put that in perspective: if your real ENEM score is 672.5, the XTRI Engine predicts you got 672.5. If a student's real score is 891.3, XTRI says 891.3. We checked 360,000 of these comparisons. The differences that remain are literally at the floor of IEEE 754 floating-point rounding from the numerical integration — they are not methodological errors. They are physics.

The largest absolute error in the entire dataset was 8.19 points, on one single Languages 2024 student whose response pattern triggered a near-degenerate posterior. Everyone else was within 0.5 points, and 95% were within 0.05 points. On the five-tier pedagogical classification (Iniciante, Elementar, Básico, Proficiente, Avançado), **99.96% of our scores put students in the same category as INEP did.** The tiny residual disagreement is always students who fall within half a point of a level boundary.

---

## Why This Matters

Three things come out of this that are worth thinking about.

**First**, independent audit is now possible. Any Brazilian student with a computer can download their own ENEM microdata, run XTRI Engine on their own responses, and verify their own score to within a quarter of a point. This has never been possible before. Up until now, if you thought your ENEM score was wrong, your only recourse was INEP's internal review process. Now you have an independent reference implementation that you can check against. If you get a different answer than INEP does, one of you has made an error — and you can prove which one.

**Second**, commercial platforms can finally be benchmarked honestly. There are dozens of Brazilian test-prep platforms that claim to produce "ENEM-compatible" scores. We can now measure how close any of them actually gets. Our own V3 "local calibration" mode — which calibrates items from scratch using only the simulado's own responses, without access to INEP's parameters — has a mean absolute error of 73 points on real ENEM data. That's a lot. It's what you'd expect from calibrating 45 items on 400 students, and there's no way to reduce it without access to the national-scale calibration data. If a commercial platform claims better precision than that without using the official items, they're either wrong or cheating.

**Third** — and this is the most interesting to us — it means the ENEM scoring pipeline is **genuinely reproducible**. This is not always true of high-stakes testing. Many testing agencies around the world treat their scoring algorithms as proprietary secrets, and neither candidates nor researchers can verify them. INEP, to their credit, publishes the item parameters and the response microdata openly. We used nothing else. No insider information, no leaked code, no private consultation. Just the public data and standard psychometric methods. And we got to 1.000000 correlation.

That's a good institution doing the right thing.

---

## Some Harder Questions

There's a lingering issue that this work doesn't fix, and it's worth mentioning because it matters for students' futures.

The 3PL model assumes **unidimensionality** — the idea that all items on the Mathematics test measure the same underlying ability called "mathematics knowledge." In 2019, Ricardo Primi and Airton Cicchetto (at São Francisco University) showed that this assumption is not quite true. They ran a principal components analysis on the residuals from the 2015 ENEM Math calibration (549,253 students!) and found clear evidence of a secondary factor — a cluster of items related specifically to probability and combinatorics that loads on a separate dimension.

Here's the consequence: under 3PL, a student who has low overall Math ability but happens to be good at probability gets **no credit** for correctly answering the probability items. The model interprets those correct answers as guessing, because its unidimensional assumption says "anyone who can solve combinatorics problems must also be good at algebra." A student who doesn't fit that pattern gets penalized.

Primi and Cicchetto found that roughly **33% of the 2015 Math participants** were in this situation — above-average on the secondary factor, below-average on the primary one. These are real students whose knowledge is real but whose scoring model can't see it.

The XTRI Engine preserves this behavior exactly, because we're reproducing INEP's model faithfully. But we're also working on a multidimensional extension (MIRT) that would fix this. It's not ready yet — MIRT calibration requires much larger samples (50,000+) and the math is harder — but it's coming. In the meantime, our benchmark proves that if you want to reproduce INEP's scoring as it is today, you can do it to the decimal place.

---

## The Code

Everything is on GitHub: [**github.com/xtribr/xtri-engine**](https://github.com/xtribr/xtri-engine)

The code is Python 3.11, uses NumPy and SciPy for the math, FastAPI for the REST API, and Supabase for the item bank database. There's a web frontend with drag-and-drop CSV upload, a validator page that runs both the local and INEP-official modes side by side, and Excel export with embedded charts. Unit and integration tests cover 91% of the codebase.

You can run the whole thing in Docker:

```bash
git clone https://github.com/xtribr/xtri-engine.git
cd xtri-engine/python_tri_v3_service
docker compose up --build
# Open http://localhost:5003
```

If you want to reproduce the 360,000-student benchmark, you'll need to download the INEP microdata files separately (RESULTADOS and ITENS_PROVA for 2022, 2023, 2024 — all publicly available at [gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)). The scripts for seeding the database and running the mass validation are documented in the repository README.

---

## Where This Is Going

We're writing up the full methodology as an academic paper for submission to a psychometrics journal (probably *Applied Psychological Measurement*). The paper contains all the gory mathematical details that don't belong in a blog post, plus the full reproducibility protocol.

Beyond that, the roadmap is:

- **Multidimensional IRT** — Extend the engine to handle the secondary factor that Primi and Cicchetto identified.
- **Earlier years** — Seed the item banks for 2009-2021 so the empirical transformation can be verified across the full history of ENEM.
- **Simulado certification** — Provide a "certified ENEM-compatible" badge for third-party simulado platforms that pass the benchmark.
- **Educational tool** — Build a classroom-facing dashboard that teachers can use to give students immediate, honest feedback on their performance using the same math INEP uses.

If any of this sounds interesting — if you're a psychometrician who wants to collaborate, a Brazilian teacher who wants to use this with your students, or just a curious reader who wants to understand how your ENEM score actually gets computed — drop me a line through the GitHub repository.

---

**Alexandre Emerson Melo de Araújo** builds XTRI, an open-source psychometrics platform focused on Brazilian educational assessment. He is based in Brazil and can be reached through the [XTRI Engine GitHub repository](https://github.com/xtribr/xtri-engine).

*Published April 2026. All data from public INEP microdata. No identifying student information was used.*
