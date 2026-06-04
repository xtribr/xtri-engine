#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

fail <- function(message, output_dir = NULL, model_spec = NULL, status = 1) {
  if (!is.null(output_dir) && !is.null(model_spec)) {
    output_path <- file.path(output_dir, "worker_output.json")
    payload <- list(
      status = "error",
      artifact_dir = output_dir,
      n_students = 0,
      n_items = 0,
      model_spec = model_spec,
      warnings = list(),
      error = message
    )
    write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  }
  cat(message, "\n", file = stderr())
  quit(status = status)
}

normalize_score_columns <- function(scores_df, factor_ids) {
  n_factors <- length(factor_ids)
  if (n_factors == 0) {
    stop("Nenhum fator foi definido para o score MIRT.")
  }

  if (ncol(scores_df) == n_factors) {
    names(scores_df) <- factor_ids
    return(scores_df)
  }

  if (ncol(scores_df) == (2 * n_factors)) {
    names(scores_df) <- c(factor_ids, paste0("SE_", factor_ids))
    return(scores_df)
  }

  stop(sprintf(
    "Formato inesperado de fscores: %s colunas para %s fatores.",
    ncol(scores_df),
    n_factors
  ))
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  fail("Uso: mirt_fit.R <worker_input.json>")
}

input_path <- args[[1]]
if (!file.exists(input_path)) {
  fail(sprintf("worker_input.json não encontrado: %s", input_path))
}

payload <- fromJSON(input_path, simplifyVector = FALSE)
output_dir <- payload$output_dir
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("mirt", quietly = TRUE)) {
  fail("Pacote R `mirt` não está instalado no ambiente.", output_dir, payload$model_spec)
}

response_df <- utils::read.csv(
  payload$response_matrix_path,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
student_id_col <- if ("student_id" %in% names(response_df)) "student_id" else names(response_df)[1]
item_cols <- setdiff(names(response_df), student_id_col)
if (length(item_cols) == 0) {
  fail("A matriz de resposta enviada ao worker não possui colunas de item.", output_dir, payload$model_spec)
}

response_matrix <- as.matrix(response_df[, item_cols, drop = FALSE])
storage.mode(response_matrix) <- "numeric"

qmatrix_spec <- fromJSON(payload$qmatrix_path, simplifyVector = FALSE)
factor_ids <- vapply(qmatrix_spec$factors, function(f) f$factor_id, character(1))
if (length(factor_ids) == 0) {
  fail("Q-matrix sem fatores declarados.", output_dir, payload$model_spec)
}

qtable <- matrix(
  0,
  nrow = length(qmatrix_spec$rows),
  ncol = length(factor_ids),
  dimnames = list(
    vapply(qmatrix_spec$rows, function(row) row$item_id, character(1)),
    factor_ids
  )
)

for (row in qmatrix_spec$rows) {
  row_id <- row$item_id
  cargas <- row$cargas
  if (length(cargas) == 0) {
    next
  }
  for (factor_id in names(cargas)) {
    qtable[row_id, factor_id] <- as.integer(cargas[[factor_id]])
  }
}

missing_items <- setdiff(item_cols, rownames(qtable))
if (length(missing_items) > 0) {
  fail(
    sprintf(
      "Q-matrix não cobre todos os itens da matriz de resposta: %s",
      paste(missing_items, collapse = ", ")
    ),
    output_dir,
    payload$model_spec
  )
}

qtable <- qtable[item_cols, , drop = FALSE]
general_factor_id <- payload$model_spec$scoring_factor_id
model_family <- payload$model_spec$family

fit <- tryCatch({
  if (identical(model_family, "bifactor")) {
    specific_factor_ids <- setdiff(colnames(qtable), general_factor_id)
    if (length(specific_factor_ids) == 0) {
      stop("Modelo bifactor sem fatores específicos declarados.")
    }

    specific <- integer(length(item_cols))
    for (idx in seq_along(item_cols)) {
      active_specifics <- which(qtable[idx, specific_factor_ids, drop = TRUE] == 1)
      if (length(active_specifics) != 1) {
        stop(sprintf(
          "Item %s precisa carregar exatamente um fator específico para o bifactor.",
          item_cols[[idx]]
        ))
      }
      specific[[idx]] <- active_specifics[[1]]
    }

    mirt::bfactor(
      response_matrix,
      specific = specific,
      itemtype = "2PL",
      verbose = FALSE,
      technical = list(NCYCLES = 500)
    )
  } else if (identical(model_family, "compensatorio")) {
    syntax_lines <- c()
    for (factor_id in colnames(qtable)) {
      item_idx <- which(qtable[, factor_id] == 1)
      if (length(item_idx) == 0) {
        next
      }
      syntax_lines <- c(
        syntax_lines,
        sprintf("%s = %s", factor_id, paste(item_idx, collapse = ",")) # nolint
      )
    }

    if (length(syntax_lines) == 0) {
      stop("Não há cargas ativas suficientes para definir o modelo compensatório.")
    }

    model_def <- mirt::mirt.model(paste(syntax_lines, collapse = "\n"))
    mirt::mirt(
      response_matrix,
      model = model_def,
      itemtype = "2PL",
      verbose = FALSE,
      technical = list(NCYCLES = 500)
    )
  } else {
    stop(sprintf("Família MIRT não suportada pelo worker: %s", model_family))
  }
}, error = function(err) {
  fail(
    sprintf("Falha no ajuste MIRT: %s", conditionMessage(err)),
    output_dir,
    payload$model_spec
  )
})

model_rds_path <- file.path(output_dir, "model.rds")
item_params_path <- file.path(output_dir, "item_params.json")
factor_scores_path <- file.path(output_dir, "factor_scores.csv")
fit_summary_path <- file.path(output_dir, "fit_summary.json")
diagnostics_path <- file.path(output_dir, "diagnostics.json")
qmatrix_snapshot_path <- file.path(output_dir, "qmatrix_snapshot.json")
worker_output_path <- file.path(output_dir, "worker_output.json")

saveRDS(fit, model_rds_path)
write_json(qmatrix_spec, qmatrix_snapshot_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

coefs <- mirt::coef(fit, IRTpars = TRUE, simplify = TRUE)
item_params <- as.data.frame(coefs$items)
item_params$item_id <- rownames(item_params)
rownames(item_params) <- NULL
write_json(
  item_params,
  item_params_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  dataframe = "rows"
)

scores <- as.data.frame(mirt::fscores(fit, method = "EAP", full.scores.SE = TRUE))
scores <- normalize_score_columns(scores, factor_ids)
scores$student_id <- response_df[[student_id_col]]
scores <- scores[, c("student_id", setdiff(names(scores), "student_id")), drop = FALSE]
utils::write.csv(scores, factor_scores_path, row.names = FALSE)

fit_summary <- list(
  model_id = payload$model_spec$model_id,
  family = model_family,
  qmatrix_version = payload$model_spec$qmatrix_version,
  logLik = as.numeric(mirt::extract.mirt(fit, "LL")),
  AIC = as.numeric(mirt::extract.mirt(fit, "AIC")),
  BIC = as.numeric(mirt::extract.mirt(fit, "BIC"))
)
write_json(fit_summary, fit_summary_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

diagnostics <- list(
  n_students = nrow(response_matrix),
  n_items = ncol(response_matrix),
  factor_ids = colnames(qtable),
  scoring_factor_id = general_factor_id,
  response_matrix_path = payload$response_matrix_path,
  qmatrix_path = payload$qmatrix_path
)
write_json(diagnostics, diagnostics_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

worker_output <- list(
  status = "done",
  artifact_dir = output_dir,
  n_students = nrow(response_matrix),
  n_items = ncol(response_matrix),
  model_spec = payload$model_spec,
  artifact_files = list(
    model_rds_path = model_rds_path,
    item_params_path = item_params_path,
    factor_scores_path = factor_scores_path,
    fit_summary_path = fit_summary_path,
    diagnostics_path = diagnostics_path,
    qmatrix_snapshot_path = qmatrix_snapshot_path
  ),
  warnings = list(),
  error = NULL
)
write_json(worker_output, worker_output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
