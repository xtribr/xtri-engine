# Changelog

## python_tri_v4 (30/10/2026)

Correções no motor científico e no `POST /api/v3/processar-json`. O modelo
3PL, a ancoragem condicional e as tabelas de referência não mudaram.

- **Chute inicial de b corrigido** (`irt/unidim_em.py`, `_init_item_params`).
  O sinal estava invertido: item fácil começava com b positivo. Agora
  b = ln((1 − c)/(p − c) − 1); com 30% de acerto e c = 0,20, b inicial = +1,95.
- **EM com até 500 iterações e diagnóstico de convergência.**
  `calibrar_itens_mml_em` ganhou `return_info=False`; com `True` devolve
  `(params, {"iteracoes", "convergiu"})`. Avisa no log quando termina sem
  convergir. `pipeline_cientifico` grava `em_iteracoes` e `em_convergiu` no
  `diagnostico`.
- **Bloco `execucao` por área** (pipeline e rota): `motor_versao`,
  `ano_referencia`, `c_strategy`, `em_iteracoes`, `em_convergiu`,
  `processado_em` (fuso America/Fortaleza). A versão vem de
  `XTRI_MOTOR_VERSAO` (build arg `GIT_COMMIT` no Dockerfile e no compose) ou
  do git local.
- **Alunos ausentes por dia de prova.** Novo campo opcional
  `presenca: {"dia1", "dia2"}` e `null` aceito nas respostas como questão em
  branco. Ausente no dia fica fora da calibração e dos resumos, sai com nota
  `null` e nível `AUSENTE`; zero acertos com marcações continua presente.
- Efeito medido (600 alunos sintéticos, 4 áreas): tempo de 27 a 32 s para
  18 a 20 s; o EM converge em 24 a 59 iterações (MT passou do antigo teto de 50);
  notas mudam em média 0,1 a 0,3 ponto (máximo 2,6).
