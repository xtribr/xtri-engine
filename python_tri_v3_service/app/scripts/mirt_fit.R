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
  check.names = FALSE,
  colClasses = "character"
)
student_id_col <- if ("student_id" %in% names(response_df)) "student_id" else names(response_df)[1]
item_cols <- setdiff(names(response_df), student_id_col)
if (length(item_cols) == 0) {
  fail("A matriz de resposta enviada ao worker não possui colunas de item.", output_dir, payload$model_spec)
}

response_matrix <- as.matrix(response_df[, item_cols, drop = FALSE])
response_matrix[response_matrix == ""] <- NA_character_
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
score_factor_ids <- factor_ids
guess_value <- if (nrow(response_matrix) < 500) 0.20 else 0.0
guess_values <- rep(guess_value, ncol(response_matrix))
estimation_method <- "EM"
convergence_tolerance <- 1e-3

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

    score_factor_ids <- c(general_factor_id, specific_factor_ids)
    mirt::bfactor(
      response_matrix,
      model = specific,
      itemtype = "2PL",
      guess = guess_values,
      verbose = FALSE,
      TOL = convergence_tolerance,
      technical = list(NCYCLES = 1000)
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
    estimation_method <- if (ncol(qtable) > 3) "MHRM" else "EM"
    set.seed(20260807)
    mirt::mirt(
      response_matrix,
      model = model_def,
      itemtype = "2PL",
      guess = guess_values,
      method = estimation_method,
      verbose = FALSE,
      TOL = convergence_tolerance,
      technical = list(NCYCLES = 1000)
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

converged <- isTRUE(mirt::extract.mirt(fit, "converged"))
iterations <- as.integer(mirt::extract.mirt(fit, "iterations"))
second_order_test <- mirt::extract.mirt(fit, "secondordertest")
condition_number <- suppressWarnings(
  as.numeric(mirt::extract.mirt(fit, "condnum"))
)
parameters <- as.numeric(mirt::extract.mirt(fit, "parvec"))
if (!converged) {
  fail(
    sprintf("O ajuste MIRT não convergiu após %s iterações.", iterations),
    output_dir,
    payload$model_spec
  )
}
if (any(!is.finite(parameters))) {
  fail("O ajuste MIRT produziu parâmetros não finitos.", output_dir, payload$model_spec)
}

model_rds_path <- file.path(output_dir, "model.rds")
item_params_path <- file.path(output_dir, "item_params.json")
factor_scores_path <- file.path(output_dir, "factor_scores.csv")
fit_summary_path <- file.path(output_dir, "fit_summary.json")
diagnostics_path <- file.path(output_dir, "diagnostics.json")
qmatrix_snapshot_path <- file.path(output_dir, "qmatrix_snapshot.json")
worker_output_path <- file.path(output_dir, "worker_output.json")

saveRDS(fit, model_rds_path)
qmatrix_copied <- file.copy(
  payload$qmatrix_path,
  qmatrix_snapshot_path,
  overwrite = TRUE
)
if (!qmatrix_copied) {
  fail("Não foi possível persistir o snapshot exato da Q-matrix.")
}

coefs <- mirt::coef(fit, IRTpars = FALSE, simplify = TRUE)
item_params <- as.data.frame(coefs$items)
slope_columns <- paste0("a", seq_along(score_factor_ids))
for (idx in seq_along(slope_columns)) {
  if (slope_columns[[idx]] %in% names(item_params)) {
    names(item_params)[names(item_params) == slope_columns[[idx]]] <-
      paste0("a_", score_factor_ids[[idx]])
  }
}
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

score_method <- if (length(score_factor_ids) > 3) "MAP" else "EAP"
retain_scores <- tolower(Sys.getenv("XTRI_MIRT_RETAIN_RESPONSES", "")) %in%
  c("1", "true", "yes")
if (retain_scores) {
  scores <- as.data.frame(
    mirt::fscores(fit, method = score_method, full.scores.SE = TRUE)
  )
  scores <- normalize_score_columns(scores, score_factor_ids)
  scores$student_id <- response_df[[student_id_col]]
  scores <- scores[, c("student_id", setdiff(names(scores), "student_id")), drop = FALSE]
  utils::write.csv(scores, factor_scores_path, row.names = FALSE)
} else {
  score_columns <- c(
    "student_id",
    score_factor_ids,
    paste0("SE_", score_factor_ids)
  )
  empty_scores <- as.data.frame(
    matrix(nrow = 0, ncol = length(score_columns)),
    stringsAsFactors = FALSE
  )
  names(empty_scores) <- score_columns
  utils::write.csv(empty_scores, factor_scores_path, row.names = FALSE)
}

fit_summary <- list(
  model_id = payload$model_spec$model_id,
  family = model_family,
  qmatrix_version = payload$model_spec$qmatrix_version,
  converged = converged,
  iterations = iterations,
  second_order_test = second_order_test,
  condition_number = condition_number,
  estimation_method = estimation_method,
  convergence_tolerance = convergence_tolerance,
  scoring_method = score_method,
  parameterization = "multidim_slope_intercept",
  c_fixo = guess_value,
  c_policy = if (guess_value > 0) "fixo_0.20_amostra_menor_500" else "2PL_sem_chute",
  R_version = R.version.string,
  mirt_version = as.character(utils::packageVersion("mirt")),
  jsonlite_version = as.character(utils::packageVersion("jsonlite")),
  logLik = as.numeric(mirt::extract.mirt(fit, "logLik")),
  AIC = as.numeric(mirt::extract.mirt(fit, "AIC")),
  BIC = as.numeric(mirt::extract.mirt(fit, "BIC"))
)
write_json(fit_summary, fit_summary_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

diagnostics <- list(
  n_students = nrow(response_matrix),
  n_items = ncol(response_matrix),
  factor_ids = colnames(qtable),
  scoring_factor_id = general_factor_id,
  scoring_method = score_method,
  parameterization = "multidim_slope_intercept",
  estimation_method = estimation_method,
  convergence_tolerance = convergence_tolerance,
  converged = converged,
  iterations = iterations,
  second_order_test = second_order_test,
  condition_number = condition_number,
  c_fixo = guess_value,
  R_version = R.version.string,
  mirt_version = as.character(utils::packageVersion("mirt")),
  jsonlite_version = as.character(utils::packageVersion("jsonlite")),
  response_matrix_path = payload$response_matrix_path,
  qmatrix_path = payload$qmatrix_path
)
write_json(diagnostics, diagnostics_path, auto_unbox = TRUE, pretty = TRUE, null = "null")

worker_warnings <- list()
if (identical(score_method, "MAP")) {
  worker_warnings <- append(
    worker_warnings,
    "MAP usado no scoring por haver mais de 3 fatores latentes"
  )
}
if (!retain_scores) {
  worker_warnings <- append(
    worker_warnings,
    "Scores da coorte não foram persistidos pela política de privacidade"
  )
}
worker_warnings <- append(
  worker_warnings,
  "model.rds contém padrões de resposta desidentificados; proteja o volume de runtime"
)
if (identical(second_order_test, FALSE)) {
  worker_warnings <- append(
    worker_warnings,
    "O ajuste não passou no teste de segunda ordem; revise os diagnósticos"
  )
}

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
  warnings = worker_warnings,
  error = NULL
)
write_json(worker_output, worker_output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
