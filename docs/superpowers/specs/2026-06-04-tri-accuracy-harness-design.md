# Arnês de acurácia unificado + loop de tuning dirigido — Design

- **Data:** 2026-06-04
- **Serviço:** XTRI Engine (TRI V3) — `python_tri_v3_service/`
- **Objetivo declarado pelo usuário:** melhorar a **acurácia vs. ENEM real**, definida como **arnês unificado** (recuperação de parâmetro + nota ponta-a-ponta com hold-out, com atribuição de erro). Abordagem escolhida: **③ k-fold + já tunar o modelo**.

---

## 1. Problema e motivação

O motor científico (MML/EM 2PL/3PL → EAP → ancoragem) é maduro, mas **a acurácia não está sendo medida de forma confiável hoje**:

1. **Guardrail desligado.** `TestRealINEPBenchmark` (`tests/test_accuracy_benchmarks.py:149-220`) — descrito no próprio código como "o guardrail mais importante contra deriva silenciosa de acurácia" — é **pulado silenciosamente** por um bug de caminho: `app/microdata/loader.py:35` define `MICRODATA_DIR = "/Volumes/KINGSTON/apps/apps/MIRT/microdata"` (um `apps/` a mais). O CSV real está em `/Volumes/KINGSTON/apps/MIRT/microdata/MICRODADOS_ENEM_2024.csv`. Logo `default_csv_path(2024)` aponta para arquivo inexistente → `pytest.skip`.
2. **Avaliação viciada.** Mesmo se rodasse, o benchmark calibra e avalia no **mesmo dado** (sem hold-out) → mede memorização, não generalização.
3. **Recuperação de parâmetro ausente.** Existe `params_oficiais.json` (INEP, CN/MT, 2017+2024) parado, sem nenhum teste comparando nossos `(a,b,c)` calibrados aos oficiais.

**Princípio (Karpathy #4):** o critério de sucesso precisa primeiro *existir e ser confiável*. Consertar a medição vem antes de qualquer mudança no modelo. Por isso o trabalho é dividido em **Fase 1 (termômetro confiável)** e **Fase 2 (tuning dirigido pela medição)**.

---

## 2. Verdade-base disponível e cobertura honesta

| Medida | Cobertura | Fonte |
|---|---|---|
| **Nota ponta-a-ponta** (`nota_v3` vs `NU_NOTA`) | **4 áreas** (LC, CH, CN, MT), 2024 | `microdata/MICRODADOS_ENEM_2024.csv` |
| **Recuperação de parâmetro** (nosso `a,b,c` vs oficial) | **CN, MT** (cor AZUL), **2024 e 2017** | `scripts_local/data/params_oficiais.json` |
| **Atribuição** (calibração vs ancoragem) | **CN, MT** (onde há params oficiais) | combinação das duas |

**Limitações declaradas:**
- LC/CH recebem **apenas** erro total ponta-a-ponta (sem params oficiais → sem decomposição calibração/ancoragem).
- Recuperação/atribuição restritas à **cor AZUL** (`co_prova` oficiais: 2024 CN=1333, MT=1323; 2017 CN=391, MT=403), porque os params oficiais são só dessa versão. O erro ponta-a-ponta total pode usar todas as provas (cada uma calibrada no seu grupo).
- 2017 valida diretamente o caso de uso real (Simulado Fredão CN/MT): não há `NU_NOTA` oficial para esses alunos (não são participantes do ENEM), então 2017 entra **só na recuperação de parâmetro** (calibra nas respostas do Fredão, compara aos params oficiais 2017).

### Estrutura de `params_oficiais.json`
```
{ "_meta": { "cor": "AZUL aplicação regular", "co_provas": {"2017_CN":391,"2017_MT":403,"2024_CN":1333,"2024_MT":1323} },
  "2017": { "CN": [[91, a, b, c, "D"], ...45 itens], "MT": [[136, a, b, c, "A"], ...] },
  "2024": { "CN": [...], "MT": [...] } }
```
Cada item = `[posição_global, a, b, c, gabarito]`, na ordem canônica da prova AZUL.

---

## 3. Sutileza psicométrica central: linking de escala

Os `(a,b)` oficiais vivem na **métrica histórica fixa do INEP**; nossa calibração usa prior **N(0,1) fresco por grupo**. As escalas **não são diretamente comparáveis** (comparar `b` cru produz deslocamento + escala espúrios).

- **Recuperação de parâmetro EXIGE linking** dos nossos params para a métrica INEP antes de medir `Δa/Δb/Δc`. Os 45 itens são comuns (mesma prova AZUL), então o linking é direto.
- **Decisão:** reportar a recuperação sob **dois métodos de linking** — **Stocking-Lord** (baseado na função de resposta, padrão-ouro) **e mean-sigma** (alinha média/desvio dos `b`). Reportar ambos serve de checagem de robustez do próprio linking.
- **Nota ponta-a-ponta NÃO precisa de linking** — `nota_v3` e `NU_NOTA` já são notas finais ancoradas na mesma escala ENEM.

---

## 4. Núcleo da atribuição de erro

Para cada aluno de **teste** (hold-out), área **CN/MT**, prova **AZUL**:

```
A) nota_full   = ancora( EAP(resp, NOSSOS params calibrados no treino) )
B) nota_anchor = ancora( EAP(resp, params OFICIAIS INEP) )

erro_total(A)      = | nota_full   − NU_NOTA |
erro_ancoragem(B)  = | nota_anchor − NU_NOTA |
parcela_calibração ≈ erro_total − erro_ancoragem
```

O caminho **B** é essencialmente o **V4** já existente (pontuar com params oficiais). Tudo reusa peças prontas:
- `calibrar_itens_fast(resp_matrix, ...)` → nossos `(a,b,c)` (`calibration.py:50`).
- `eap_theta_batch(resp_matrix, item_params)` → θ com params **fixos**, nossos OU oficiais (`calibration.py:70`).
- ancoragem em `app/irt/transform.py`.

---

## 5. Fase 1 — termômetro confiável

1. **Corrigir o bug bloqueador**: `app/microdata/loader.py:35` (e docstring na linha 7) `apps/apps` → `apps`.
2. **Skip barulhento**: quando o microdata/params faltam, o arnês **falha com mensagem clara** (CLI) ou emite `xfail`/erro explícito — nunca "verde por ausência de dado". (O guard fino do pytest pode `skip`, mas loga *por quê* e quantos casos foram pulados.)
3. **k-fold sobre alunos** dentro de cada grupo `(área, co_prova[, tp_lingua para LC])`: calibra params nos alunos de treino, pontua os de teste com params fixos (`eap_theta_batch`) → mede generalização.
4. **IC bootstrap** (reamostragem dos alunos de teste) em MAE / RMSE / correlação por área.
5. **Linking de escala** (Stocking-Lord + mean-sigma) antes da recuperação de parâmetro (CN/MT, 2017+2024).
6. **Relatório** (JSON estruturado + tabela legível estilo INEP, reaproveitando a estética de `scripts_local/tabela_rodada1_inep_style.py`): por área — erro total, erro de ancoragem, parcela de calibração, recuperação `Δa/Δb/Δc` (sob ambos linkings), ICs, e **onde o erro se concentra**.
7. **Baseline congelado**: os números atuais viram a linha de base contra a qual a Fase 2 é julgada.

---

## 6. Fase 2 — loop de tuning dirigido pela atribuição

A decomposição decide a alavanca (sem mexer no chute):

- **Se calibração domina** → `c_strategy=bayes`; prior/shrinkage suave em `a` (lognormal, ex. mode ~1.0); revisar limiar de saturação `b≥4.8`; tolerância de convergência do EM.
- **Se ancoragem domina** → interpolação por spline monotônica (corrige descontinuidade de inclinação em `pct=0.5` no `transform.py`); correção do colapso de teto em 0/45 acertos; entrada `LC=45`.

**Portão de aceitação:** cada candidato re-roda no k-fold; só é aceito se o **IC bootstrap da diferença pareada de MAE** (por aluno: `|erro_candidato| − |erro_baseline|`) **exclui 0** para o lado da melhora. (Diferença pareada, não comparação de dois ICs independentes — é mais sensível e correto.) Ao aceitar, os *floors* do pytest (`_INEP_BENCHMARK`) são **apertados** para travar o ganho (anti-regressão).

---

## 7. Estrutura de código

Arquivos pequenos e focados (rule: many small files, alta coesão):

```
app/evaluation/
  __init__.py
  ground_truth.py   # carrega params oficiais + microdata; filtra AZUL p/ CN/MT; alinha posição↔item
  linking.py        # Stocking-Lord + mean-sigma (nossos params → métrica INEP)
  kfold.py          # split de alunos por grupo; calibra treino → pontua teste
  attribution.py    # run com params oficiais vs nossos → decomposição
  metrics.py        # MAE/RMSE/r + IC bootstrap + stats de recuperação
  report.py         # render JSON + tabela estilo INEP
  __main__.py       # CLI: `python -m app.evaluation --year 2024 [--kfolds 5] [--out report.json]`
tests/
  test_accuracy_harness.py   # floors finos sobre a saída do arnês + correção do TestRealINEPBenchmark
```

Reuso, sem reescrever: `calibration.calibrar_itens_fast` / `eap_theta_batch` / `calibrate_area_group`, `transform` (ancoragem), `tabela_referencia`, caminho V4 (params oficiais), `microdata.loader` (carga + filtro primeira-aplicação).

---

## 8. Execução

- **Relatório completo (pesado)**: `python -m app.evaluation --year 2024` — roda local sobre o microdata real, emite JSON + tabela.
- **Guard de CI (rápido)**: `tests/test_accuracy_harness.py` mantém só floors (MAE/r por área) + valida a correção do caminho; `skip` com motivo logado se o dado não estiver presente.

Isso separa **medição exploratória** (manual/local, lenta) de **regressão** (CI, rápida).

---

## 9. Tratamento de erros e validação

- Validar alinhamento item↔oficial: comprimento 45, gabarito casando entre microdata e `params_oficiais.json` (detecta cor/posição trocada).
- Restringir recuperação/atribuição aos `co_prova` AZUL; logar quantos alunos entraram por grupo.
- Falhar alto e claro quando dado-base falta (lição do bug de caminho).
- Respeitar a **regra inviolável**: calibração escopada a um `(simulado/prova × área)` — o k-fold nunca mistura provas/áreas num modelo.

---

## 10. Estratégia de testes

- **Unitários** (rápidos, sintéticos): `linking` (recupera transformação A,B conhecida em dado sintético), `kfold` (split correto, sem vazamento treino↔teste), `metrics` (MAE/IC bootstrap em casos fechados).
- **Integração** (skip-if-missing, barulhento): floors ponta-a-ponta sobre microdata 2024 + recuperação CN/MT.
- Meta de cobertura do novo módulo: ≥ 80% (alinha com o padrão do repo, 92%).

---

## 11. Critérios de sucesso verificáveis

- **Fase 1 pronta** quando: o arnês roda no dado real 2024 **sem skip silencioso**; emite relatório com as 3 medidas (ponta-a-ponta 4 áreas em 2024; recuperação CN/MT sob 2 linkings em 2024+2017; atribuição CN/MT em **2024**, único ano com `NU_NOTA`) + ICs bootstrap; e grava um **baseline**.
- **Fase 2 pronta** quando: a fonte dominante de erro é reduzida com **melhoria confirmada pelo IC bootstrap da diferença pareada de MAE** (exclui 0), e os floors do pytest são apertados para o novo nível.

---

## 12. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Alinhamento posição↔item oficial errado (cor/ordem) | Validar gabarito casa; abortar com erro se não casar |
| Linking instável por item outlier | Reportar Stocking-Lord **e** mean-sigma; divergência grande = sinal de problema |
| k-fold lento em 200k alunos | Calibração já é batch/vetorizada; usar amostra por grupo se necessário, logando o corte |
| `nota_v3` vs `NU_NOTA` mistura erro de ancoragem (tabelas ≠ transformação exata INEP) | É exatamente o que a atribuição separa; MAE residual de ancoragem é esperado |
| Tuning "consertar A, quebrar B" | Portão por IC + floors apertados pós-aceite travam regressão |

---

## 13. Fora de escopo (YAGNI)

- LC/CH recuperação de parâmetro (sem params oficiais disponíveis).
- Reescrever os motores; o arnês **mede**, a Fase 2 ajusta cirurgicamente o que a medição apontar.
- Pipeline em nuvem/CI pesado — execução local é suficiente para single-user.
