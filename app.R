# Shiny app to run Embedded Covariate Selection
# Save this file as app.R and run with: shiny::runApp('path/to/app.R')

library(shiny)
library(readxl)
library(dplyr)
library(stringi)
library(data.table)
library(glmnet)
library(mgcv)
library(RRF)
library(xgboost)
library(lightgbm)

# --- Utility: read uploaded file (csv or excel) ---
read_file_auto <- function(path) {
  ext <- tools::file_ext(path)
  if (tolower(ext) %in% c("xls", "xlsx")) {
    readxl::read_excel(path)
  } else if (tolower(ext) %in% c("csv", "txt")) {
    read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  } else {
    stop("Unsupported file type: ", ext)
  }
}

# --- Paste the embedded_covariate_selection function here ---
# (This is the same function you provided, slightly adapted for Shiny.)

embedded_covariate_selection <- function(covdata, pa, weights = NULL, force = NULL, 
                                         algorithms = c("glm", "gam", "rf", "xgb", "lgb"), 
                                         ncov = ceiling(log2(length(which(pa == 1)))), 
                                         maxncov = 12, nthreads = parallel::detectCores()/2) {
  ranks_1 <- data.frame()
  if (!is.numeric(weights)) weights <- rep(1, length(pa))
  
  # GLM (elastic-net)
  if ("glm" %in% algorithms) {
    form <- as.formula(paste0("as.factor(pa) ~ ", paste(paste0("poly(", names(covdata), ",2)"), collapse = " + "), "-1"))
    x <- model.matrix(form, covdata)
    mdl.glm <- suppressWarnings(glmnet::cv.glmnet(x, as.factor(pa), alpha = 0.5, weights = weights, family = "binomial", type.measure = "deviance", parallel = TRUE))
    glm.beta <- as.data.frame(as.matrix(coef(mdl.glm, s = mdl.glm$lambda.1se)))
    glm.beta <- data.frame(covariate = row.names(glm.beta), coef = as.numeric(abs(glm.beta[, 1])))[which(glm.beta != 0), ][-1, ]
    if (nrow(glm.beta) < 1) {
      glm.beta <- as.data.frame(as.matrix(coef(mdl.glm, s = mdl.glm$lambda.min)))
      glm.beta <- data.frame(covariate = row.names(glm.beta), coef = as.numeric(abs(glm.beta[, 1])))[which(glm.beta != 0), ][-1, ]
    }
    if (nrow(glm.beta) >= 1) {
      glm.beta <- data.frame(glm.beta[order(glm.beta$coef, decreasing = TRUE), ], model = "glm")
      glm.beta$covariate <- stri_sub(glm.beta$covariate, 6, -6)
      glm.beta <- data.frame(data.table::setDT(glm.beta)[, .SD[which.max(coef)], by = covariate])
      glm.beta$rank <- 1:nrow(glm.beta)
      ranks_1 <- rbind(ranks_1, glm.beta[, c("covariate", "rank", "model")])
    }
  }
  
  # GAM
  if ("gam" %in% algorithms) {
    form <- as.formula(paste0("pa ~ ", paste(paste0("s(", names(covdata), ",bs='cr')"), collapse = " + ")))
    mdl.gam <- try(mgcv::bam(form, data = cbind(covdata, pa = pa), weights = weights, family = "binomial", method = "fREML", select = TRUE, discrete = TRUE, control = list(nthreads = nthreads)), silent = TRUE)
    if (!inherits(mdl.gam, "try-error")) {
      gam.beta <- data.frame(covariate = names(mdl.gam$model)[!names(mdl.gam$model) %in% c("(weights)", "pa")], summary(mdl.gam)$s.table, row.names = NULL)
      gam.beta <- gam.beta[gam.beta$p.value < 0.9, ]
      if (nrow(gam.beta) >= 1) {
        gam.beta <- data.frame(gam.beta[order(abs(gam.beta$Chi.sq), decreasing = TRUE), ], rank = 1:nrow(gam.beta), model = "gam")
        ranks_1 <- rbind(ranks_1, gam.beta[, c("covariate", "rank", "model")])
      }
    }
  }
  
  # RF
  if ("rf" %in% algorithms) {
    rf <- try(RRF::RRF(covdata, as.factor(pa), flagReg = 0), silent = TRUE)
    if (!inherits(rf, "try-error")) {
      impRF <- rf$importance[, "MeanDecreaseGini"]
      imp <- impRF/(max(impRF))
      gamma <- 0.5
      coefReg <- (1 - gamma) + gamma * imp
      mdl.rf <- RRF::RRF(covdata, as.factor(pa), classwt = c(`0` = min(weights), `1` = max(weights)), coefReg = coefReg, flagReg = 1)
      rf.beta <- data.frame(covariate = row.names(mdl.rf$importance), mdl.rf$importance, row.names = NULL)
      rf.beta <- rf.beta[which(rf.beta$MeanDecreaseGini > 0), ]
      if (nrow(rf.beta) >= 1) {
        rf.beta <- data.frame(rf.beta[order(rf.beta$MeanDecreaseGini, decreasing = TRUE), ], rank = 1:nrow(rf.beta), model = "rf")
        ranks_1 <- rbind(ranks_1, rf.beta[, c("covariate", "rank", "model")])
      }
    }
  }
  
  # XGBoost
  if ("xgb" %in% algorithms) {
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(covdata), label = pa, weight = weights)
    param <- list(objective = "binary:logistic", eval_metric = "logloss")
    mdl.xgb <- try(xgboost::xgb.train(params = param, data = dtrain, nrounds = 100, nthread = nthreads), silent = TRUE)
    if (!inherits(mdl.xgb, "try-error")) {
      imp <- xgboost::xgb.importance(model = mdl.xgb)
      xgb.beta <- data.frame(covariate = imp$Feature, rank = rank(-imp$Gain), model = "xgb")
      ranks_1 <- rbind(ranks_1, xgb.beta)
    }
  }
  
  # LightGBM
  if ("lgb" %in% algorithms) {
    dtrain <- try(lightgbm::lgb.Dataset(data = as.matrix(covdata), label = pa, weight = weights), silent = TRUE)
    if (!inherits(dtrain, "try-error")) {
      param <- list(objective = "binary", metric = "binary_logloss", num_threads = nthreads)
      mdl.lgb <- try(lightgbm::lgb.train(params = param, data = dtrain, nrounds = 100), silent = TRUE)
      if (!inherits(mdl.lgb, "try-error")) {
        imp <- lightgbm::lgb.importance(mdl.lgb)
        lgb.beta <- data.frame(covariate = imp$Feature, rank = rank(-imp$Gain), model = "lgb")
        ranks_1 <- rbind(ranks_1, lgb.beta)
      }
    }
  }
  
  if (nrow(ranks_1) < 1) {
    message("No covariate selected after the embedding procedure ...")
    return(NULL)
  } else {
    intersect.tmp <- ranks_1[ranks_1$covariate %in% names(which(table(ranks_1$covariate) == length(unique(ranks_1$model)))), ]
    intersect.tmp <- aggregate(intersect.tmp[, c("rank")], list(intersect.tmp$covariate), sum)
    colnames(intersect.tmp) <- c("covariate", "rank")
    intersect.sel <- data.frame(intersect.tmp[order(intersect.tmp$rank, decreasing = FALSE), ], rank.f = 1:nrow(intersect.tmp))
    union.tmp <- ranks_1[ranks_1$covariate %in% names(which(table(ranks_1$covariate) < length(unique(ranks_1$model)))), ]
    if (nrow(union.tmp) > 0) {
      union.tmp <- aggregate(union.tmp[, c("rank")], list(union.tmp$covariate), sum)
      colnames(union.tmp) <- c("covariate", "rank")
      union.sel.tmp <- data.frame(union.tmp[order(union.tmp$rank, decreasing = FALSE), ], rank.f = (max(intersect.sel$rank.f + 1)):(max(intersect.sel$rank.f) + nrow(union.tmp)))
      ranks_2 <- rbind(intersect.sel, union.sel.tmp)
    } else {
      ranks_2 <- intersect.sel
    }
    if (ncov > maxncov) ncov <- maxncov
    if (ncov > nrow(ranks_2)) ncov <- nrow(ranks_2)
    ranks_2 <- ranks_2[1:ncov, ]
    if (is.character(force)) {
      tf <- force[which(!(force %in% ranks_2$covariate))]
      if (length(tf > 1)) {
        toforce <- data.frame(covariate = tf, rank = "forced", rank.f = "forced")
        ranks_2[c(nrow(ranks_2) - nrow(toforce) + 1):c(nrow(ranks_2)), ] <- toforce
      }
    }
    ranks_2 <- ranks_2[, c("covariate", "rank.f")]
    covdata <- covdata[sub(".*\\.", "", unlist(ranks_2["covariate"]))]
    return(list(covdata = covdata, ranks_1 = ranks_1, ranks_2 = ranks_2))
  }
}

# --- Shiny UI ---
ui <- fluidPage(
  titlePanel("Covariate Selection ") ,
  sidebarLayout(
    sidebarPanel(
      helpText("Upload ONE covariate file (CSV/XLSX) containing all predictors, including RI, AR, DMR if applicable."),
      fileInput("flow", "Upload Discharge CSV/XLSX", accept = c('.csv', '.xls', '.xlsx')),
      fileInput("cov", "Upload Covariates CSV/XLSX", accept = c('.csv', '.xls', '.xlsx')),
      hr(),
      uiOutput("station_ui"),
      numericInput("maxncov", "Max covariates to select", value = 8, min = 1, max = 20),
      checkboxGroupInput("algos", "Algorithms to use", choices = c("glm","gam","rf","xgb","lgb"), selected = c("glm","gam","rf","xgb")),
      actionButton("run", "Run Selection"),
      hr(),
      downloadButton("downloadRanks", "Download Selected Covariates")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Correlation", plotOutput("corrplot")),
        tabPanel("AUC–PR", plotOutput("prcurve"), verbatimTextOutput("auc_value")),
        tabPanel("Ranks (All)", dataTableOutput("ranks_all")),
        tabPanel("Selected Covariates", dataTableOutput("ranks_selected")),
        tabPanel("Messages", verbatimTextOutput("messages"))
      )
    )
  )
)

# --- Shiny Server ---
server <- function(input, output, session) {
  values <- reactiveValues(flow = NULL, cov = NULL, stations = NULL, msg = NULL, results = NULL, treatment = NULL)
  
  observeEvent(input$flow, {
    req(input$flow)
    values$flow <- read_file_auto(input$flow$datapath)
    values$stations <- colnames(values$flow)[-1]
  })
  observeEvent(input$cov, { req(input$cov); values$cov <- read_file_auto(input$cov$datapath) })
  
  output$station_ui <- renderUI({
    if (is.null(values$stations)) return(helpText("Upload discharge file to populate stations"))
    selectInput("station", "Choose station", choices = values$stations)
  })
  
  observeEvent(input$run, {
    req(values$flow, values$cov, input$station)
    station <- input$station
    response <- values$flow[[station]]
    values$treatment <- ifelse(response > median(response, na.rm = TRUE), 1, 0)
    
    predictors <- values$cov
    
    values$results <- tryCatch({
      embedded_covariate_selection(
        covdata = as.data.frame(predictors),
        pa = values$treatment,
        algorithms = input$algos,
        ncov = ceiling(log2(sum(values$treatment == 1))),
        maxncov = input$maxncov,
        nthreads = max(1, parallel::detectCores()%/%2)
      )
    }, error = function(e) {
      values$msg <- e$message
      NULL
    })
  })
  
  # --- Correlation plot ---
  output$corrplot <- renderPlot({
    req(values$results)
    sel <- values$results$covdata
    cmat <- cor(sel, use = "pairwise.complete.obs")
    image(1:ncol(cmat), 1:ncol(cmat), cmat, axes = FALSE, col = heat.colors(20))
    axis(1, 1:ncol(cmat), colnames(cmat), las = 2)
    axis(2, 1:ncol(cmat), colnames(cmat), las = 2)
  })
  
  # --- AUC–PR curve ---
  output$prcurve <- renderPlot({
    req(values$results)
    X <- as.matrix(values$results$covdata)
    mdl <- glm(values$treatment ~ ., data = as.data.frame(X), family = binomial)
    prob <- predict(mdl, type = "response")
    pred <- ROCR::prediction(prob, values$treatment)
    perf <- ROCR::performance(pred, "prec", "rec")
    plot(perf, main = "Precision–Recall Curve")
  })
  
  output$auc_value <- renderPrint({
    req(values$results)
    X <- as.matrix(values$results$covdata)
    mdl <- glm(values$treatment ~ ., data = as.data.frame(X), family = binomial)
    prob <- predict(mdl, type = "response")
    pred <- ROCR::prediction(prob, values$treatment)
    aucpr <- ROCR::performance(pred, "aucpr")@y.values[[1]]
    cat("AUC–PR:", round(aucpr, 3))
  })
  
  output$ranks_all <- renderDataTable({ req(values$results); DT::datatable(values$results$ranks_1) })
  output$ranks_selected <- renderDataTable({ req(values$results); DT::datatable(values$results$ranks_2) })
  
  output$messages <- renderPrint({ values$msg })
  
  output$downloadRanks <- downloadHandler(
    filename = function() paste0("selected_covariates_", Sys.Date(), ".csv"),
    content = function(file) write.csv(values$results$ranks_2, file, row.names = FALSE)
  )
}

shinyApp(ui, server)
