#!/usr/bin/env Rscript

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
if (length(args) != 3) {
  cat("Uso: mirt_score.R <model.rds> <response_matrix.csv> <output.csv>\n", file = stderr())
  quit(status = 1)
}

if (!requireNamespace("mirt", quietly = TRUE)) {
  cat("Pacote R `mirt` não está instalado no ambiente.\n", file = stderr())
  quit(status = 1)
}

model_rds_path <- args[[1]]
response_matrix_path <- args[[2]]
output_path <- args[[3]]

if (!file.exists(model_rds_path)) {
  cat(sprintf("Arquivo model.rds não encontrado: %s\n", model_rds_path), file = stderr())
  quit(status = 1)
}
if (!file.exists(response_matrix_path)) {
  cat(sprintf("Matriz de respostas não encontrada: %s\n", response_matrix_path), file = stderr())
  quit(status = 1)
}

fit <- readRDS(model_rds_path)
response_df <- utils::read.csv(
  response_matrix_path,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

student_id_col <- if ("student_id" %in% names(response_df)) "student_id" else names(response_df)[1]
item_cols <- setdiff(names(response_df), student_id_col)
if (length(item_cols) == 0) {
  cat("A matriz de respostas não possui colunas de item.\n", file = stderr())
  quit(status = 1)
}

response_matrix <- as.matrix(response_df[, item_cols, drop = FALSE])
storage.mode(response_matrix) <- "numeric"

coef_list <- mirt::coef(fit, simplify = TRUE)
factor_ids <- setdiff(names(coef_list), c("items", "means", "cov"))

scores <- as.data.frame(
  mirt::fscores(
    fit,
    response.pattern = response_matrix,
    method = "EAP",
    full.scores.SE = TRUE
  )
)
scores <- normalize_score_columns(scores, factor_ids)
scores$student_id <- response_df[[student_id_col]]
scores <- scores[, c("student_id", setdiff(names(scores), "student_id")), drop = FALSE]

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(scores, output_path, row.names = FALSE)
