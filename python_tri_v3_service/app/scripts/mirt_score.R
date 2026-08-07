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
if (!(length(args) %in% c(3, 4))) {
  cat(
    "Uso: mirt_score.R <model.rds> <response_matrix.csv> <output.csv> [qmatrix.json]\n",
    file = stderr()
  )
  quit(status = 1)
}

if (!requireNamespace("mirt", quietly = TRUE)) {
  cat("Pacote R `mirt` não está instalado no ambiente.\n", file = stderr())
  quit(status = 1)
}

model_rds_path <- args[[1]]
response_matrix_path <- args[[2]]
output_path <- args[[3]]
qmatrix_path <- if (length(args) == 4) args[[4]] else NULL

if (!file.exists(model_rds_path)) {
  cat(sprintf("Arquivo model.rds não encontrado: %s\n", model_rds_path), file = stderr())
  quit(status = 1)
}
if (!file.exists(response_matrix_path)) {
  cat(sprintf("Matriz de respostas não encontrada: %s\n", response_matrix_path), file = stderr())
  quit(status = 1)
}

fit <- readRDS(model_rds_path)
fit_summary <- NULL
fit_summary_path <- file.path(dirname(model_rds_path), "fit_summary.json")
if (file.exists(fit_summary_path)) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    cat("Pacote R `jsonlite` não está instalado.\n", file = stderr())
    quit(status = 1)
  }
  fit_summary <- jsonlite::fromJSON(fit_summary_path, simplifyVector = TRUE)
  expected_mirt_version <- fit_summary$mirt_version
  current_mirt_version <- as.character(utils::packageVersion("mirt"))
  if (
    !is.null(expected_mirt_version) &&
    !identical(as.character(expected_mirt_version), current_mirt_version)
  ) {
    cat(
      sprintf(
        "Versão MIRT incompatível: artefato=%s runtime=%s\n",
        expected_mirt_version,
        current_mirt_version
      ),
      file = stderr()
    )
    quit(status = 1)
  }
}
response_df <- utils::read.csv(
  response_matrix_path,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  colClasses = "character"
)

student_id_col <- if ("student_id" %in% names(response_df)) "student_id" else names(response_df)[1]
item_cols <- setdiff(names(response_df), student_id_col)
if (length(item_cols) == 0) {
  cat("A matriz de respostas não possui colunas de item.\n", file = stderr())
  quit(status = 1)
}

response_matrix <- as.matrix(response_df[, item_cols, drop = FALSE])
response_matrix[response_matrix == ""] <- NA_character_
storage.mode(response_matrix) <- "numeric"

qmatrix_spec <- NULL
declared_factor_ids <- NULL
if (!is.null(qmatrix_path)) {
  if (!file.exists(qmatrix_path)) {
    cat(sprintf("Q-matrix não encontrada: %s\n", qmatrix_path), file = stderr())
    quit(status = 1)
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    cat("Pacote R `jsonlite` não está instalado.\n", file = stderr())
    quit(status = 1)
  }
  qmatrix_spec <- jsonlite::fromJSON(qmatrix_path, simplifyVector = FALSE)
  declared_factor_ids <- vapply(
    qmatrix_spec$factors,
    function(f) f$factor_id,
    character(1)
  )
}

n_factors <- if (!is.null(declared_factor_ids)) {
  length(declared_factor_ids)
} else {
  as.integer(mirt::extract.mirt(fit, "nfact"))
}
score_method <- if (n_factors > 3) "MAP" else "EAP"
internal_factor_ids <- as.character(mirt::extract.mirt(fit, "factorNames"))

if (!is.null(qmatrix_spec)) {
  model_family <- if (!is.null(fit_summary$family)) {
    as.character(fit_summary$family)
  } else {
    NA_character_
  }

  if (identical(model_family, "bifactor")) {
    general_factor_ids <- vapply(
      Filter(function(f) identical(f$kind, "geral"), qmatrix_spec$factors),
      function(f) f$factor_id,
      character(1)
    )
    specific_factor_ids <- vapply(
      Filter(function(f) identical(f$kind, "especifico"), qmatrix_spec$factors),
      function(f) f$factor_id,
      character(1)
    )
    if (
      length(general_factor_ids) != 1 ||
      length(internal_factor_ids) != (1 + length(specific_factor_ids))
    ) {
      stop("Não foi possível mapear os fatores internos para a Q-matrix.")
    }
    # `bfactor()` usa G/S1/S2 internamente na ordem dos índices específicos;
    # sempre os remapeamos para a ordem semântica congelada na Q-matrix.
    factor_ids <- c(general_factor_ids, specific_factor_ids)
  } else if (identical(model_family, "compensatorio")) {
    if (!setequal(internal_factor_ids, declared_factor_ids)) {
      stop("Fatores internos do modelo compensatório divergem da Q-matrix.")
    }
    # Os nomes internos já são semânticos. Mantemos sua ordem para não trocar
    # scores caso ela difira da ordem de declaração da Q-matrix.
    factor_ids <- internal_factor_ids
  } else {
    stop("Família MIRT ausente ou inválida no fit_summary.json.")
  }
} else {
  # Compatibilidade com artefatos antigos: usa os nomes que o próprio modelo
  # retorna. A API nova sempre fornece o snapshot da Q-matrix.
  factor_ids <- internal_factor_ids
}

scores <- as.data.frame(
  mirt::fscores(
    fit,
    response.pattern = response_matrix,
    method = score_method,
    full.scores.SE = TRUE
  )
)
scores <- normalize_score_columns(scores, factor_ids)
scores$student_id <- response_df[[student_id_col]]
scores <- scores[, c("student_id", setdiff(names(scores), "student_id")), drop = FALSE]

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(scores, output_path, row.names = FALSE)
