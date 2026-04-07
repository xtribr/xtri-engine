# Findings: The Brittleness Pattern

## Patches Made So Far

Categorized by what each patch was actually fixing:

### Category A — Parser format gaps
| Patch | File that triggered it | What was missing |
|-------|------------------------|------------------|
| XLSX support | `dados-mat.xlsx` | openpyxl integration |
| TAB delimiter | `cn.txt` | `_detect_delimiter` only checked `,` and `;` |
| Headerless detection | `cn.txt` | parser assumed first row is always header |
| ID column inference (blank header) | `mat.csv` (semicolon CSV) | parser only looked for literal "ID" |
| Position-based item alignment | `mat.csv` via V4 endpoint | endpoint required exact length match |

**Pattern**: Each new format reveals a new assumption baked into the parser. The parser was built for one format and grew via patches.

### Category B — Constants wrong
| Patch | Trigger | What was wrong |
|-------|---------|----------------|
| `TRI_MAXIMA_OFICIAL` LC: 835 → 800 | LC.txt user complaint | Constant came from old Excel, never validated against microdata |
| `TRI_MAXIMA_OFICIAL` CH: 855 → 835 | Same fix | Same issue |
| Test `test_obter_45_acertos` updated | Constant change broke test | Test asserted constant >= table value, no longer true |

**Pattern**: Constants were hand-tuned from a single Excel file. They were never empirically verified against the actual ENEM scoring distribution. When the user noticed (LC student getting 835 = impossible), we fixed only that case.

### Category C — Anchoring quirks
| Patch | Trigger | What was wrong |
|-------|---------|----------------|
| Adaptive smoothing for small groups | cn.txt complaint about 2 students with same acertos getting wildly different notas | Rank Intragrupo distributed students from tri_min to tri_max regardless of group size |

**Pattern**: The anchoring algorithm worked for large samples but had pathological behavior for groups of 2-5. We added smoothing only when noticed.

### Category D — Frontend assumptions
| Patch | Trigger | What was wrong |
|-------|---------|----------------|
| Dynamic CO_PROVA dropdown | mat.csv via V4 with wrong hardcoded code | Hardcoded list of CO_PROVAs was guessed, not from DB |
| LC tp_lingua handling | LC test with 50-char gabarito | Frontend assumed gabarito always 45 chars |

**Pattern**: Frontend hardcoded values that should have been queried from the backend.

## The Root Cause

**There is no single "ENEM scoring spec" that the codebase tests against.** Instead, behavior was inferred from:
1. The user's example files (one at a time)
2. An Excel reference table from unknown vintage
3. Hand-tuned constants

Result: each new file is a "natural experiment" that exposes another untested case.

## What a Robust Foundation Looks Like

### Principle 1: Parser must handle all reasonable formats by default
The parser should accept any combination of:
- Delimiters: `,` `;` `\t` (comma, semicolon, tab)
- Headers: present, missing, or partial
- ID column: present (named or first column), or synthesize
- File types: CSV/TSV/TXT (text), XLSX (Excel)
- Encoding: UTF-8, UTF-8-BOM, Latin-1 (with auto-fallback)
- Item count: 30 to 50 binary columns (some areas may have 44-45 due to abandoned items)

### Principle 2: All constants derived from microdata, not Excel
Replace `TRI_MAXIMA_OFICIAL`, `TRI_P999_REAL`, etc. with values explicitly computed from:
- ENEM 2022 RESULTADOS_*.csv
- ENEM 2023 MICRODADOS_ENEM_2023.csv
- ENEM 2024 RESULTADOS_2024.csv

For each (year, area), compute and version-control:
- `mediana_nacional`
- `p99_real`
- `p99_9_real`  
- `p99_99_real`
- `max_observado`

### Principle 3: Test suite must cover format variations
Currently 119 tests but most test internal logic. Need parametric tests like:

```python
@pytest.mark.parametrize("file,area,expected_n,expected_max", [
    ("fixtures/mat.csv", "MT", 403, 990),
    ("fixtures/cn.txt", "CN", 49, 875),
    ("fixtures/lc.txt", "LC", 54, 800),
    ("fixtures/ch.txt", "CH", 54, 835),
    ("fixtures/dados-mat.xlsx", "CN", 403, 875),
])
def test_parser_handles_format_combo(file, area, expected_n, expected_max):
    ...
```

If a new format breaks, add it to the test list, not to the parser.

### Principle 4: Document what's accepted
A `FORMATS.md` file at the repo root listing every format the parser accepts, with example snippets.

## Files in Test Corpus

| File | Format | Encoding | Header | ID col | Area | N | Items |
|------|--------|----------|--------|--------|------|---|-------|
| mat.csv | comma CSV | UTF-8 | yes | "ID" | MT | 403 | 45 |
| dados-mat.xlsx | XLSX | — | yes | "ID" | CN | 403 | 45 |
| RESULTADO_2022.csv | semicolon | Latin-1 | yes | NU_INSCRICAO | all 4 | 3.5M | 45/area |
| RESULTADOS_2024.csv | semicolon | Latin-1 | yes | NU_SEQUENCIAL | all 4 | 4.3M | 45/area |
| cn.txt | TAB | UTF-8 | no | none | CN | 49 | 45 |
| LC.txt | TAB | UTF-8 | no | none | LC | 54 | 45 |
| CH.txt | TAB | UTF-8 | no | none | CH | 54 | 45 |
| mat.txt | TAB | UTF-8 | no | none | MT | 48 | 45 |

This is the corpus the parser MUST handle without modification.

## Empirical Constants Needed (replace TRI_MAXIMA_OFICIAL hardcoded values)

Computed from microdata sample 100K-200K students per (year, area):

```
              | LC    | CH    | CN    | MT
--------------|-------|-------|-------|-------
2022 mediana  | 524   | 510   | 480   | 510
2022 max      | 764.6 | 829.1 | 868.7 | 983.9
2022 P99.9    | 707.8 | 745.6 | 753.4 | 916.3
2023 mediana  | ~530  | ~512  | ~485  | ~515
2023 max      | 758.6 | 823.0 | 868.4 | 958.6
2023 P99.9    | 719.3 | 755.7 | 779.6 | 913.9
2024 mediana  | 531.3 | 516.1 | 488.3 | 498.8
2024 max      | 795.8 | 819.7 | 867.2 | 961.9
2024 P99.9    | 709.2 | 768.8 | 750.6 | 918.2
```

## Conclusion

The user is right. We've been patching. The right fix is to:

1. Define the parser contract (what it MUST accept) via tests
2. Compute constants from microdata once, version them
3. Stop hand-tuning anything

After this refactor, processing a new file should require ZERO code changes — just upload it.
