library(tidyverse)
library(tidytext)
library(distr)
library(philentropy)
n = dplyr::n
sd = stats::sd

source("Analysis_Utils.R")

# Load Human ISA data ----

## Frequencies ----

bigrams_freqs <- read.csv("bigrams/3979_bigrams_with_frequencies.csv", header = TRUE)

unique_bigrams_freqs <- bigrams_freqs %>%
  filter(MakesSense != "Easy (majority very easy; no somewhat/very hard)") %>%  # Easy is a subset of makes sense
  mutate_at(c("Adjective", "Noun", "Bigram", "MakesSense", "AdjectiveClass", "Frequency"), factor) %>%
  mutate(Frequency = factor(Frequency, levels = c("Zero",
                                                  "Near-Zero (1-3)",
                                                  "Below 10th percentile",
                                                  "10th-25th percentile",
                                                  "25th-50th percentile",
                                                  "50th-75th percentile",
                                                  "75th-90th percentile",
                                                  "90th-95th percentile",
                                                  "95th-99th percentile",
                                                  "99th percentile"
  ))) %>%
  mutate(CoarseFrequency = fct_collapse(Frequency, 
                                        Zero = c("Zero"), 
                                        "Below 25th percentile" = c("Near-Zero (1-3)", "Below 10th percentile", "10th-25th percentile"),
                                        "50th-75th percentile" = c("50th-75th percentile"),
                                        "75th-90th percentile" = c("75th-90th percentile"),
                                        "90th-99th percentile" = c("90th-95th percentile", "95th-99th percentile", "99th percentile")
  ))

## ISA ----

isa_variance_12_combined = read.csv("isa_variance_12_combined.csv") 
isa_variance_12_combined %>%
  mutate(across(c(Adjective, Noun, Bigram, AdjectiveClass, Frequency, CoarseFrequency, Experiment), as.factor)) ->
  isa_variance_12_combined

## Context ----

an_context_combined_plus_target = read.csv("isa_context_combined_target.csv")
an_context_combined_plus_target %>%
  mutate(across(c(Bigram, UserId, ContextBias, Rating, Adjective, Noun), as.factor)) %>%
  group_by(Bigram, ContextBias) %>%
  summarise(Mean = mean(NumRating), SD = sd(NumRating), SE = SD / sqrt(n()), Variance = var(NumRating),
            Adjective = unique(Adjective), Noun = unique(Noun), .groups = "drop") %>%
  merge(unique_bigrams_freqs %>% dplyr::select("Bigram", "Count", "Frequency", "CoarseFrequency"), 
        by=c("Bigram"),
        all.x=TRUE) %>%
  mutate(Frequency = droplevels(Frequency), CoarseFrequency = droplevels(CoarseFrequency)) ->
  context_variance_combined

# Preprocessing ----

preprocess_labelled_responses <- function(responses) {
  responses %>%
    mutate(across(c("Adjective", "Noun", "Bigram"), as.factor)) %>%
    mutate(PredictedResponse = factor(PredictedResponse, 
                                      levels=c("Definitely not", "Probably not", 
                                               "Unsure", 
                                               "Probably yes", "Definitely yes")),
           NumPredictedResponse = as.integer(PredictedResponse)) -> tidy_responses
  return(tidy_responses)
}

# Some bigrams occur both in the original bigrams as fillers 
# (i.e. only one adjective-noun combination)
# and also in the full cross of the analogy bigrams
# Ratings will be identical for both cases so we can just keep the first
remove_duplicate_bigrams <- function(responses) {
  responses %>%
    distinct(Bigram, .keep_all=TRUE) -> dedup_responses
  return(dedup_responses)
}

# Sampling from log-probabilities ----

sample_preprocess_labelled_responses = function(responses) {
  responses %>%
    mutate(across(c(Adjective, Noun, Bigram), as.factor)) %>%
    mutate(PredictedResponse = factor(PredictedResponse, 
                                      levels=c("Definitely not", "Probably not", 
                                               "Unsure", 
                                               "Probably yes", "Definitely yes")),
           NumPredictedResponse = as.integer(PredictedResponse)) ->
    tidy_responses
  
  tidy_responses %>%
    mutate(
      DefinitelyYesProb = exp(-Definitely.yesSurprisal), 
      ProbablyYesProb = exp(-Probably.yesSurprisal), 
      UnsureProb = exp(-UnsureSurprisal),
      ProbablyNotProb = exp(-Probably.notSurprisal), 
      DefinitelyNotProb = exp(-Definitely.notSurprisal), 
      ProbSum = DefinitelyYesProb + ProbablyYesProb + UnsureProb + ProbablyNotProb + DefinitelyNotProb,
      BalancedDefinitelyYesProb = DefinitelyYesProb / ProbSum,
      BalancedProbablyYesProb = ProbablyYesProb / ProbSum,
      BalancedUnsureProb = UnsureProb / ProbSum,
      BalancedProbablyNotProb = ProbablyNotProb / ProbSum,
      BalancedDefinitelyNotProb = DefinitelyNotProb / ProbSum,
    ) %>%
    select(!ProbSum) ->
    tidy_responses
  
  return(tidy_responses)
}

sample_preprocess_numeric_responses = function(responses) {
  responses %>%
    mutate(across(c(Adjective, Noun, Bigram), as.factor)) %>%
    mutate(PredictedResponse = factor(PredictedResponse),
           NumPredictedResponse = as.integer(PredictedResponse)) ->
    tidy_responses
  
  tidy_responses %>%
    mutate(
      DefinitelyYesProb = exp(-X1Surprisal), 
      ProbablyYesProb = exp(-X2Surprisal), 
      UnsureProb = exp(-X3Surprisal),
      ProbablyNotProb = exp(-X4Surprisal), 
      DefinitelyNotProb = exp(-X5Surprisal), 
      ProbSum = DefinitelyYesProb + ProbablyYesProb + UnsureProb + ProbablyNotProb + DefinitelyNotProb,
      BalancedDefinitelyYesProb = DefinitelyYesProb / ProbSum,
      BalancedProbablyYesProb = ProbablyYesProb / ProbSum,
      BalancedUnsureProb = UnsureProb / ProbSum,
      BalancedProbablyNotProb = ProbablyNotProb / ProbSum,
      BalancedDefinitelyNotProb = DefinitelyNotProb / ProbSum,
    ) %>%
    select(!ProbSum) ->
    tidy_responses
  
  return(tidy_responses)
}

sample_responses = function(preprocessed_responses, sample_n, participant_label="Llama2Chat") {
  set.seed(42)
  preprocessed_responses %>%
    rowwise() %>%
    mutate(NumRating = list(r(DiscreteDistribution(
      supp = c(1, 2, 3, 4, 5), 
      prob = c(BalancedDefinitelyNotProb,
               BalancedProbablyNotProb,
               BalancedUnsureProb,
               BalancedProbablyYesProb,
               BalancedDefinitelyYesProb))
    )(sample_n))) %>%
    unnest(cols = c(NumRating)) %>%
    select(Adjective, Noun, Bigram, Question, NumRating) %>%
    mutate(Rating = case_when(
      NumRating == 5 ~ "Definitely yes",
      NumRating == 4 ~ "Probably yes",
      NumRating == 3 ~ "Unsure",
      NumRating == 2 ~ "Probably not",
      NumRating == 1 ~ "Definitely not"
    )) %>%
    mutate(ParticipantId = participant_label) %>%
    mutate(Rating = as.factor(Rating)) ->
    sampled_responses
  
  return(sampled_responses)
}

# 1 SD from human mean ----

## Accuracy ----

accuracy_1sd <- function(responses, isa_variance_12_combined) {
  responses %>%
    select(Bigram, NumIsaRating) %>%
    merge(isa_variance_12_combined %>% select(Bigram, AdjectiveClass, CoarseFrequency, Mean, SD) %>%
            rename(HumanMean = Mean, HumanSD = SD),
          by="Bigram") %>%
    mutate(CoarseFrequency = case_when(
      CoarseFrequency == "Zero" ~ "Zero frequency",
      CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency",
      .default = "Low frequency"
    )) %>%
    # Question: do we want to round the mean +/- SD to the nearest integer before calculating?
    mutate(WithinSD = !is.na(NumIsaRating) & 
             NumIsaRating <= round(HumanMean + HumanSD) & 
             NumIsaRating >= round(HumanMean - HumanSD)) ->
    within_sds
  
  within_sds %>%
    summarize(Accuracy = mean(WithinSD)) %>%
    mutate(AdjectiveClass = "Overall") ->
    total_accuracy
  
  within_sds %>%
    group_by(AdjectiveClass) %>%
    summarize(Accuracy = mean(WithinSD)) ->
    class_accuracy
  
  within_sds %>%
    group_by(CoarseFrequency) %>%
    summarize(Accuracy = mean(WithinSD)) %>%
    rename(AdjectiveClass = CoarseFrequency) ->
    freq_accuracy
  
  within_sds %>%
    group_by(AdjectiveClass, CoarseFrequency) %>%
    summarize(Accuracy = mean(WithinSD), .groups = "drop") %>%
    unite(AdjectiveClass, AdjectiveClass, CoarseFrequency, sep = ' ') ->
    class_freq_accuracy
  
  bind_rows(total_accuracy, class_accuracy, freq_accuracy, class_freq_accuracy) %>%
    select(AdjectiveClass, Accuracy) %>%
    tibble() -> result
  return(result)
}

human_accuracy_1sd <- function(isa_data_12_combined, isa_variance_12_combined) {
  if(!("Mean" %in% names(isa_data_12_combined))) {
    isa_data_12_combined %>%
      merge(isa_variance_12_combined %>% select(Bigram, Mean, SD),
            by="Bigram") -> isa_data_12_combined
  }
  isa_data_12_combined %>%
    mutate(CoarseFrequency = case_when(
      CoarseFrequency == "Zero" ~ "Zero frequency",
      CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency",
      .default = "Low frequency"
    )) %>%
    mutate(WithinSD = !is.na(NumRating) & 
             NumRating <= round(Mean + SD) & 
             NumRating >= round(Mean - SD)) ->
    human_within_sds
  
  if ("ParticipantID" %in% names(isa_data_12_combined)) {
    pid_column = "ParticipantId"
  } else {
    pid_column = "UserId"
  }
  
  human_within_sds %>%
    group_by_at(c(pid_column)) %>%
    summarize(Accuracy = mean(WithinSD)) %>%
    mutate(AdjectiveClass = "Overall") ->
    human_total_accuracy
  
  human_within_sds %>%
    group_by_at(c(pid_column, "AdjectiveClass")) %>%
    summarize(Accuracy = mean(WithinSD), .groups = "drop") ->
    human_class_accuracy
  
  human_within_sds %>%
    group_by_at(c(pid_column, "CoarseFrequency")) %>%
    summarize(Accuracy = mean(WithinSD), .groups = "drop") %>%
    rename(AdjectiveClass = CoarseFrequency) ->
    human_freq_accuracy
  
  human_within_sds %>%
    group_by_at(c(pid_column, "AdjectiveClass", "CoarseFrequency")) %>%
    summarize(Accuracy = mean(WithinSD), .groups = "drop") %>%
    unite(AdjectiveClass, AdjectiveClass, CoarseFrequency, sep = ' ') ->
    human_class_freq_accuracy
  
  bind_rows(human_total_accuracy, human_class_accuracy, human_freq_accuracy, human_class_freq_accuracy) %>%
    group_by(AdjectiveClass) %>%
    summarize(Accuracy = mean(Accuracy)) ->
    human_mean_acc
  
  return(human_mean_acc)
}

accuracy_1sd_context <- function(responses, context_variance_combined) {
  if (!('NumIsaRating' %in% names(responses))) {
    if ('NumPredictedResponse' %in% names(responses)) {
      responses %>%
        rename(NumIsaRating = NumPredictedResponse) ->
        responses
    } else {
      responses %>%
        rename(NumIsaRating = NumRating) ->
        responses
    }
  }
  
  if ('Privative' %in% levels(responses$ContextBias)) {
    responses %>%
      mutate(ContextBias = fct_recode(ContextBias, "privative" = "Privative", 
                                      "subsective" = "Subsective")) ->
      responses
  }
  
  responses %>%
    mutate(AdjectiveClass = "Privative") -> 
    responses
  
  context_variance_combined %>%
    mutate(AdjectiveClass = "Privative") ->
    context_variance_combined
  
  accuracy_1sd(responses %>% filter(ContextBias == 'privative'),
               context_variance_combined %>% filter(ContextBias == 'privative')) ->
    privative_acc
  
  accuracy_1sd(responses %>% filter(ContextBias == 'subsective'),
               context_variance_combined %>% filter(ContextBias == 'subsective')) ->
    subsective_acc
  
  accuracy_1sd(responses %>%  # accuracy_1sd merges over bigrams
                 unite(Bigram, Bigram, ContextBias),
               context_variance_combined %>%
                 unite(Bigram, Bigram, ContextBias)
               ) ->
    overall_acc
  
  bind_rows(overall_acc %>% mutate(ContextBias = 'both'),
            privative_acc %>% mutate(ContextBias = 'privative'),
            subsective_acc %>% mutate(ContextBias = 'subsective')
            ) %>%
    select(AdjectiveClass, ContextBias, Accuracy) ->
    context_acc
  
  return(context_acc)
}

## Plot within 1 SD ----

plot_1sd <- function(isa_variance, lm_responses,
                     adjectives, nouns, lm_name,
                     context = FALSE, poster = FALSE, 
                     title=TRUE, thresholds = FALSE,
                     human_color="#2C365E",
                     lm_color="#D81B60") {
  isa_variance %>%
    filter(Adjective %in% adjectives) %>%
    filter(Noun %in% nouns) %>%
    rename(NumIsaRating = Mean) -> filtered_variance

  lm_responses %>%
    filter(Adjective %in% adjectives) %>%
    filter(Noun %in% nouns) %>%
    rename(NumIsaRating = NumPredictedResponse) -> 
    filtered_lm_responses
  
  if (context == TRUE) {
    filtered_variance %>%
      mutate(ContextBias = fct_recode(ContextBias,
        "Privative Context" = "privative", "Subsective Context" = "subsective"
      )) -> filtered_variance
    filtered_lm_responses %>%
      mutate(ContextBias = fct_recode(ContextBias,
        "Privative Context" = "privative", "Subsective Context" = "subsective"
      )) -> filtered_lm_responses
  }
  
  if (poster == TRUE) {
    human_dot_size = 7
    lm_dot_size = 8
    error_bar_width = 0.4
    error_linewidth = 2
  } else {
    human_dot_size = 3
    lm_dot_size = 4
    error_bar_width = 0.2
    error_linewidth = 1
  }
  
  scale_values <- c('Human'=human_color)
  setNames(c(scale_values, lm_color), c(names(scale_values), lm_name)) -> scale_values
  
  filtered_variance %>%
    ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                                within=Adjective,fun=median), 
               y = NumIsaRating)) -> plot
  
  if (thresholds == TRUE) {
    plot +
      geom_hline(yintercept=2, color="#F8766D", linewidth=error_linewidth) +
      geom_hline(yintercept=4, color="#00BFC4", linewidth=error_linewidth) -> plot
  }
  
  plot +
    geom_point(aes(color='Human'), size=human_dot_size) +
    geom_errorbar(aes(ymin = pmax(1, round(NumIsaRating - SD)), ymax = pmin(5, round(NumIsaRating + SD))), 
                  width = error_bar_width, linewidth=error_linewidth, color=human_color) +
    geom_jitter(aes(color=lm_name), 
                data = filtered_lm_responses, 
                shape=18, height=0.1, width=0.1, size=lm_dot_size) +
    scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
    guides(x = guide_axis(angle = 90)) +
    xlab("Noun") +
    ylab("Rating") +
    scale_color_manual(name='Data Source',
                       breaks=c('Human', 
                                lm_name),
                       values=scale_values) +
    scale_y_continuous(breaks=1:5, limits = c(0.5,5.5)) +
    theme_minimal() ->
    plot
  
  if (title == TRUE) {
    if (length(adjectives) > 1) {
      if (context == TRUE) {
        plot + 
          ggtitle("Ratings for \'In this setting, is an AN still an N?\'") ->
          plot
      } else {
        plot + 
          ggtitle("Ratings for \'Is an AN still an N?\'") ->
          plot
      }
    } else {
      if (context == TRUE) {
        plot + 
          ggtitle(sprintf("Ratings for \'In this setting, is a %s N still an N?\'", adjectives[[1]])) ->
          plot
      } else {
        plot + 
          ggtitle(sprintf("Ratings for \'Is a %s N still an N?\'", adjectives[[1]])) ->
          plot
      }
    }
  } else {
    # Assume this is for paper, put legend on the bottom
    plot + 
      theme(legend.position = "bottom") +
      guides(fill = guide_legend(title.position="top", title.hjust = 0.5)) ->
      plot
  }
  
  if (context == TRUE) {
    if (length(adjectives) > 1) {
      plot +
        facet_wrap(~ Adjective * ContextBias, scale="free_x", ncol = 6) ->
        plot
    } else {
      plot +
        facet_wrap(~ ContextBias) ->
        plot
    }
  } else if (length(adjectives) > 1) {
    plot +
      facet_wrap(~ Adjective, scale="free_x") ->
      plot
  }
  
  plot + 
    theme(legend.position="bottom",
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank()
    ) ->
    plot
  
  if (poster == TRUE) {
    plot + 
      theme(text=element_text(size=36, color="#2C365E")) ->
      plot
  }
  return(plot)
}

# Straight accuracy for context ----

context_direct_accuracy <- function(responses) {
  if (!('NumRating' %in% names(responses))) {
    # LM responses
    responses %>%
      rename(NumRating = NumPredictedResponse) ->
      responses
  }
  
  if ('privative' %in% levels(responses$ContextBias)) {
    responses %>%
      mutate(ContextBias = fct_recode(ContextBias, "Privative" = "privative", 
                                      "Subsective" = "subsective")) ->
      responses
  }
  
  responses %>%
    filter(ContextBias %in% c("Privative", "Subsective")) %>%  
    mutate(CoarseFrequency = case_when(
      CoarseFrequency == "Zero" ~ "Zero frequency",
      CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency",
      .default = "Low frequency"
    )) %>%
    mutate(ContextEffective = (ContextBias == "Privative" & NumRating <= 2) | 
                                 (ContextBias == "Subsective" & NumRating >= 4)) ->
    eval_responses
  
  eval_responses %>%
    summarize(Accuracy = mean(ContextEffective)) %>%
    mutate(Class = "Overall") ->
    total_accuracy
  
  eval_responses %>%
    group_by(ContextBias) %>%
    summarize(Accuracy = mean(ContextEffective)) %>%
    rename(Class = ContextBias) ->
    bias_accuracy
  
  eval_responses %>%
    group_by(CoarseFrequency) %>%
    summarize(Accuracy = mean(ContextEffective)) %>%
    rename(Class = CoarseFrequency) ->
    freq_accuracy
  
  bind_rows(total_accuracy, bias_accuracy, freq_accuracy) %>%
    select(Class, Accuracy) %>%
    tibble() -> result
  return(result)
}

# Baselines ----

label.random_baseline = "Random baseline"
label.majority_baseline = "\"Majority\" baseline"

## Random baseline ----
# sample from uniform distribution of 1-5 for each bigram, 
# calculate accuracy, do this 1000 times, average

random_sample <- function(isa_variance_12_combined) {
  set.seed(42)
  bigram_count = nrow(isa_variance_12_combined)
  isa_variance_12_combined %>%
    select(Bigram, Mean, SD) %>%
    mutate(NumIsaRating = sample.int(5, size=bigram_count, replace=TRUE)) ->
    samples
  return(samples)
}

random_sample_1sd <- function(isa_variance_12_combined) {
  samples <- random_sample(isa_variance_12_combined)
  samples %>%
    accuracy_1sd(isa_variance_12_combined) ->
    accuracy_df
  return(accuracy_df)
}

calculate_random_baseline <- function(isa_variance_12_combined) {
  set.seed(42)
  replicate(100, random_sample_1sd(isa_variance_12_combined), simplify = FALSE) %>%
    do.call("rbind", .) %>%
    group_by(AdjectiveClass) %>%
    summarize(Accuracy = mean(Accuracy)) -> 
    random_baseline_acc
  
  return(random_baseline_acc)
}

random_sample_1sd_context <- function(context_variance_combined) {
  privative_samples <- random_sample(context_variance_combined %>% filter(ContextBias == "privative")) %>%
    mutate(ContextBias = "privative")
  subsective_samples <- random_sample(context_variance_combined %>% filter(ContextBias == "subsective")) %>%
    mutate(ContextBias = "subsective")
  bind_rows(privative_samples, subsective_samples) -> samples
  samples %>%
    accuracy_1sd_context(context_variance_combined) ->
    accuracy_df
  return(accuracy_df)
}

calculate_random_baseline_context <- function(context_variance_combined) {
  set.seed(42)
  replicate(100, random_sample_1sd_context(context_variance_combined), simplify = FALSE) %>%
    do.call("rbind", .) %>%
    group_by(AdjectiveClass, ContextBias) %>%
    summarize(Accuracy = mean(Accuracy)) -> 
    random_baseline_acc
  
  return(random_baseline_acc)
}

## Baseline of just guessing 3 for privative and 5 for subsective ----

calculate_majority_baseline <- function(isa_variance_12_combined) {
  isa_variance_12_capped %>%
    select(Bigram, AdjectiveClass, HumanMean, HumanSD) %>%
    mutate(NumIsaRating = ifelse(AdjectiveClass == "Subsective", 5, 3)) %>%
    accuracy_1sd(isa_variance_12_combined) ->
    majority_baseline_acc
  
  return(majority_baseline_acc)
}
# Majority-per-class baseline: Overall accuracy 0.751, privative: 0.587, subsective: 0.92
# With rounding to nearest integer: overall 0.882, privative 0.774, subsective 0.993


# KL Divergence & JS Divergence ----

## Build distribution ----

label.human = "Human"

build_human_lm_ratings <- function(isa_data, lm_data, lm_label="LLM") {
  if (!('NumRating' %in% names(lm_data))) {
    # LM responses
    lm_data %>%
      rename(NumRating = NumPredictedResponse,
             Rating = PredictedResponse) ->
      lm_data
  }
  
  human_bigrams = isa_data %>% distinct(Bigram) %>% pull(Bigram)
  lm_bigrams = lm_data %>% distinct(Bigram) %>% pull(Bigram)
  
  human_lm_ratings = bind_rows(isa_data %>% 
                                 mutate(HumanOrLM = label.human), 
                               lm_data %>%
                                 mutate(HumanOrLM = lm_label)
  ) %>%
    select("Bigram", "Adjective", "Noun", "CoarseFrequency", "AdjectiveClass", "Rating", "NumRating", "HumanOrLM") %>%
    mutate(CoarseFrequency = fct_relevel(CoarseFrequency, "Zero")) %>%
    mutate(across(c(HumanOrLM, Rating), as.factor)) %>%
    mutate(IsaRating = fct_relevel(Rating, "Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"),
           AdjectiveClass = fct_relevel(AdjectiveClass, "Subsective")) %>%
    select(!c("Rating"))
  
  human_lm_ratings %>%
    filter(Bigram %in% human_bigrams & Bigram %in% lm_bigrams) ->
    human_lm_ratings
  return(human_lm_ratings)
}

rating_distribution <- function(ratings, lm_label = NULL) {
  if (!('Rating' %in% names(ratings))) {
    ratings %>%
      mutate(Rating = IsaRating) -> ratings
  }
  
  if (is.null(lm_label)) {
    return(ratings %>%
      group_by(Bigram, Rating, .drop=FALSE) %>%
      summarize(Count = dplyr::n(), .groups = "drop_last") %>%
      # Fix adjective classes
      merge(ratings %>% select(Bigram, AdjectiveClass, CoarseFrequency) %>% distinct(), .by = "Bigram")
    )
  } else {
    return(ratings %>%
             filter(HumanOrLM %in% c(label.human, lm_label)) %>%
             mutate(HumanOrLM = fct_recode(fct_drop(HumanOrLM), LLM=lm_label)) %>%
             group_by(HumanOrLM, Bigram, Rating, .drop=FALSE) %>%
             summarize(Count = dplyr::n(), .groups = "drop_last") %>%
             # Fix adjective classes
             merge(ratings %>% select(Bigram, AdjectiveClass, CoarseFrequency) %>% distinct(), .by = "Bigram")
    )
  }
}


## Calculate divergences ----

df_kl_divergence <- function(x) {
  x %>% 
    filter(HumanOrLM == label.human) %>% 
    select(!HumanOrLM) %>% 
    as.numeric() -> human_probs
  x %>% 
    filter(HumanOrLM == "LLM") %>% 
    select(!HumanOrLM) %>%
    as.numeric() -> llm_probs
  # Make sure human probabilities are P (ground truth) and LLMs are Q 
  probs = rbind(human_probs, llm_probs)
  return(KL(probs, unit="log2"))
}

total_variation_distance <- function(x) {
  x %>% 
    filter(HumanOrLM == label.human) %>% 
    select(!HumanOrLM) %>% 
    as.numeric() -> human_probs
  x %>% 
    filter(HumanOrLM == "LLM") %>% 
    select(!HumanOrLM) %>%
    as.numeric() -> llm_probs
  tv_distance = 0.5 * sum(abs(human_probs - llm_probs))
  return(tv_distance)
}

counts_to_probs <- function(rating_count_dist) {
  if (!('Rating' %in% names(rating_count_dist))) {
    rating_count_dist %>%
      mutate(Rating = IsaRating) -> rating_count_dist
  }
  
  rating_count_dist %>% 
    select(c(Bigram, HumanOrLM, Rating, Count)) %>%
    group_by(Bigram, HumanOrLM) %>%
    mutate(Prob = Count / sum(Count)) %>%
    select(!Count) %>%
    ungroup() %>%
    pivot_wider(names_from=Rating, values_from=Prob) -> rating_probs
  return(rating_probs)
}

calculate_divergences <- function(human_lm_ratings, lm_label, js_only=FALSE) {
  human_lm_dists = rating_distribution(human_lm_ratings, lm_label)
  
  human_lm_dists %>%
    counts_to_probs() %>%
    group_by(Bigram) %>%
    arrange(HumanOrLM) -> prepared_dists # Sort Human before LLM
  return(calculate_divergences_from_probs(prepared_dists, js_only = js_only))
}

calculate_divergences_from_probs <- function(stacked_rating_dist, js_only = TRUE) {
  stacked_rating_dist %>%
    ungroup() %>%
    select(c(Bigram, HumanOrLM, 
             "Definitely yes", "Probably yes", "Unsure",
             "Probably not", "Definitely not")) %>%
    group_by(Bigram) %>%
    arrange(HumanOrLM) -> prepared_dists # Sort Human before LLM
  if(js_only==TRUE) {
    prepared_dists %>%
      group_modify(~ data.frame(JSDivergence=philentropy::distance(x = as.matrix(.x %>% select(!HumanOrLM)), 
                                                                   method = "jensen-shannon", 
                                                                   unit = "log2",
                                                                   mute.message = TRUE)
      )) -> dists
  } else {
    prepared_dists %>%
      group_modify(~ data.frame(KLDivergence=df_kl_divergence(.x),
                                # Jensen-Shannon divergence is symmetric so row order doesn't matter
                                JSDivergence=philentropy::distance(x=as.matrix(.x %>% select(!HumanOrLM)), 
                                                                   method = "jensen-shannon", 
                                                                   unit = "log2",
                                                                   mute.message = TRUE),
                                TVDistance=total_variation_distance(.x)
      )) -> dists
  }
  dists %>%
    add_frequency() ->
    human_lm_divergences
  return(human_lm_divergences)
}

ks_test_by_bigram <- function(human_ratings, lm_ratings, group_variable = "Bigram", adjust=FALSE) {
  bind_rows(human_ratings, lm_ratings) %>%
    mutate(HumanOrLM = factor(HumanOrLM)) -> human_lm_ratings
  
  human_bigrams = human_ratings %>% distinct(Bigram) %>% pull(Bigram)
  lm_bigrams = lm_ratings %>% distinct(Bigram) %>% pull(Bigram)
  
  human_lm_ratings %>%
    filter(Bigram %in% human_bigrams & Bigram %in% lm_bigrams) ->
    human_lm_ratings
  
  human_lm_ratings %>%
    group_by_at(group_variable) %>%
    group_modify(~ data.frame(KS_pvalue=ks.test(x = .x %>% filter(HumanOrLM == "Human") %>% pull(NumRating),
                                                y = .x %>% filter(HumanOrLM != "Human") %>% pull(NumRating))$p.value
    )) -> human_lm_kss
  
  if (adjust==TRUE) {
    human_lm_kss$KS_pvalue = p.adjust(human_lm_kss$KS_pvalue, method = "holm")
  }
 
  return(human_lm_kss)
}

build_human_dist <- function(human_ratings) {
  human_ratings %>%
    rating_distribution() %>%
    mutate(HumanOrLM = "Human") %>%
    counts_to_probs() %>%
    add_frequency() -> human_dist
  return(human_dist)
}

build_lm_dist <- function(lm_responses) {
  lm_responses %>%
    sample_preprocess_labelled_responses() %>%
    mutate("Definitely yes" = BalancedDefinitelyYesProb,
           "Probably yes" = BalancedProbablyYesProb,
           "Unsure" = BalancedUnsureProb,
           "Probably not" = BalancedProbablyNotProb,
           "Definitely not" = BalancedDefinitelyNotProb) %>%
    mutate(HumanOrLM = "LM") %>%
    select(c(Bigram, HumanOrLM, 
             "Definitely yes", "Probably yes", "Unsure",
             "Probably not", "Definitely not")) %>%
    add_frequency() -> lm_dist
  return(lm_dist)
}

calculate_distribution_js <- function(human_dist, lm_dist) {
  bind_rows(human_dist, lm_dist) %>%
    mutate(HumanOrLM = factor(HumanOrLM)) -> human_lm_dist
  
  human_bigrams = human_dist %>% distinct(Bigram) %>% pull(Bigram)
  lm_bigrams = lm_dist %>% distinct(Bigram) %>% pull(Bigram)
  
  human_lm_dist %>%
    filter(Bigram %in% human_bigrams & Bigram %in% lm_bigrams) ->
    human_lm_dist
  return(calculate_divergences_from_probs(human_lm_dist, js_only=TRUE))
}

# Uniform distribution baseline
calculate_uniform_js <- function(human_ratings) {
  human_ratings %>%
    rating_distribution() %>%
    mutate(HumanOrLM = "Human") %>%
    counts_to_probs() -> human_dist
  human_dist %>% select(Bigram) %>%
    mutate("Definitely yes" = 0.2,
           "Probably yes" = 0.2,
           "Unsure" = 0.2,
           "Probably not" = 0.2,
           "Definitely not" = 0.2) %>%
    mutate(HumanOrLM = "LM") -> lm_dist
  bind_rows(human_dist, lm_dist) %>%
    mutate(HumanOrLM = factor(HumanOrLM)) -> human_lm_dist
  return(calculate_divergences_from_probs(human_lm_dist, js_only=TRUE))
}

# Majority distribution baseline 
calculate_majority_js <- function(human_ratings) {
  human_ratings %>%
    rating_distribution() %>%
    mutate(HumanOrLM = "Human") %>%
    counts_to_probs() -> human_dist
  human_dist %>% select(Bigram) %>%
    add_frequency() %>%
    mutate("Definitely yes" = case_when(AdjectiveClass == "Subsective" ~ 1, .default = 0),
           "Probably yes" = 0,
           "Unsure" = case_when(AdjectiveClass == "Privative" ~ 1, .default = 0),
           "Probably not" = 0,
           "Definitely not" = 0) %>%
    select(c("Bigram", "Definitely yes", "Probably yes", "Unsure", "Probably not", "Definitely not")) %>%
    mutate(HumanOrLM = "LM") -> lm_dist
  bind_rows(human_dist, lm_dist) %>%
    mutate(HumanOrLM = factor(HumanOrLM)) -> human_lm_dist
  return(calculate_divergences_from_probs(human_lm_dist, js_only=TRUE))
}

calculate_single_human_resampled_js <- function(human_ratings, sample_n=12) {
  human_ratings %>%
    rating_distribution() %>%
    mutate(HumanOrLM = "Human") %>%
    counts_to_probs() -> human_dist
  human_dist %>%
    rename_with(make.names) %>%
    rowwise() %>%
    mutate(NumRating = list(r(DiscreteDistribution(
      supp = c(1, 2, 3, 4, 5), 
      prob = c(Definitely.not,
               Probably.not,
               Unsure,
               Probably.yes,
               Definitely.yes))
    )(sample_n))) %>%
    unnest(cols = c(NumRating)) %>%
    select(Bigram, NumRating) %>%
    mutate(Rating = case_when(
      NumRating == 5 ~ "Definitely yes",
      NumRating == 4 ~ "Probably yes",
      NumRating == 3 ~ "Unsure",
      NumRating == 2 ~ "Probably not",
      NumRating == 1 ~ "Definitely not"
    )) %>%
    mutate(Rating = factor(Rating)) %>%
    add_frequency() -> sampled_ratings
  sampled_ratings %>%
    rating_distribution() %>%
    mutate(HumanOrLM = "LM") %>%
    counts_to_probs() -> lm_dist
  bind_rows(human_dist, lm_dist) %>%
    mutate(HumanOrLM = factor(HumanOrLM)) -> human_lm_dist
  return(calculate_divergences_from_probs(human_lm_dist, js_only=TRUE))
}

calculate_human_resampled_js <- function(human_ratings) {
  replicate(50, calculate_single_human_resampled_js(human_ratings), simplify = FALSE) %>%
    do.call("rbind", .) %>%
    group_by(Bigram) %>%
    summarize(JSDivergence = mean(JSDivergence), AdjectiveClass=first(AdjectiveClass), CoarseFrequency=first(CoarseFrequency)) -> divs
  return(divs)
}

# LOO human distribution baseline
calculate_human_loo_js <- function(human_ratings) {
  all_divs = list()
  unique_pids <- unique(human_ratings$UserId)
  for (i in seq_along(unique_pids)) {
    pid <- unique_pids[[i]]
    human_ratings %>%
      filter(UserId == pid) %>%
      select(Bigram) %>% 
      pull() -> bigrams_rated_by_pid 
    human_ratings %>%
      filter(UserId != pid & Bigram %in% bigrams_rated_by_pid) %>%
      rating_distribution() %>%
      mutate(HumanOrLM = "LM") %>%
      counts_to_probs() -> loo_dist
    bind_rows(human_dist %>% 
                filter(Bigram %in% bigrams_rated_by_pid), 
              loo_dist) %>%
      mutate(HumanOrLM = factor(HumanOrLM)) -> human_lm_dist
    calculate_divergences_from_probs(human_lm_dist, js_only=TRUE) -> js_divs
    js_divs %>%
      mutate(LOOId = pid) -> js_divs
    all_divs[[i]] <- js_divs
  }
  
  stacked_divs <- do.call(dplyr::bind_rows, all_divs)
  stacked_divs %>% 
    group_by(Bigram) %>%
    summarize(JSDivergence=mean(JSDivergence), 
              AdjectiveClass=first(AdjectiveClass), 
              CoarseFrequency=first(CoarseFrequency)) -> mean_divs
  return(mean_divs)
}
  
pivot_dist_longer <- function(rating_dist) {
  rating_dist %>%
    pivot_longer(cols=c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"),
                 names_to = "Rating", values_to = "Probability") %>%
    mutate(Rating = factor(Rating, levels=c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"))) %>%
    mutate(NumRating = as.integer(Rating)) -> long_dist
  return(long_dist)
}

summarize_js_divergences <- function(stacked_divergences) {
  stacked_divergences %>%
    group_by(Method, Parameters) %>%
    summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence), .groups = "drop") %>%
    mutate(AdjectiveClass = "Overall") %>%
    bind_rows(stacked_divergences %>%
                group_by(Method, Parameters, AdjectiveClass) %>%
                summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence), .groups = "drop")) %>%
    mutate(AdjectiveClass = factor(AdjectiveClass, levels=c("Privative", "Subsective", "Overall"))) %>%
    mutate(AdjectiveClass = fct_recode(AdjectiveClass, "Privative adjective" = "Privative", "Subsective adjective" = "Subsective")) -> 
    js_divergence_means
  
  return(js_divergence_means)
}

## Plot split bar plots ----

single_bar_plot <- function(rating_dist, 
                            dist_color, 
                            facet_by = "Bigram", 
                            bigrams = NULL, 
                            sorted_bigrams = FALSE,
                            vertical = FALSE) {
  long_dist = pivot_dist_longer(rating_dist)
  
  if (!is.null(bigrams)) {
    long_dist %>%
      filter(Bigram %in% bigrams) -> 
      long_dist
    if (sorted_bigrams == TRUE) {
      long_dist %>%
        mutate(Bigram = factor(Bigram, levels=bigrams)) -> long_dist
    } else {
      long_dist %>%
        mutate(Bigram = fct_relevel(Bigram, sort)) -> long_dist
    }
  }
  
  if (facet_by != "Bigram") {
    # Need to calculate averaged probability distributions
    long_dist %>%
      group_by(NumRating) %>%
      group_by_at(facet_by, .add = TRUE) %>%
      summarize(Probability = mean(Probability), .groups = "drop") ->
      long_dist
  }
  
  long_dist %>%
    ggplot(aes(x=NumRating, y=Probability)) + 
    geom_col(position="identity", fill=dist_color) +
    scale_y_continuous(labels = abs, limits=c(0,1)) +
    xlab("Rating") +
    theme_minimal() -> plot
  
  if (vertical == TRUE) {
    plot +
      coord_flip() ->
      plot
  }
  
  if (facet_by == "AdjectiveClass") {
    plot +
      facet_grid(~ AdjectiveClass) ->
      plot
  } else if (facet_by == "Adjective") {
    plot +
      facet_grid(~ Adjective) ->
      plot
  } else if (facet_by == "Bigram" & length(bigrams) > 1) {
    plot +
      facet_grid(~ Bigram) ->
      plot
  }
  
  return(plot)
}

longer_and_name_dist <- function(dist, dist_name) {
  dist %>%
    pivot_dist_longer() %>%
    mutate(Model = dist_name) -> new_dist
  return(new_dist)
}

insert_linebreaks <- function(labels, width = 10) {
  sapply(labels, function(lbl) {
    paste(strwrap(lbl, width = width), collapse = "\n")
  })
}

split_bar_plot <- function(dists, model_names_colors, 
                           human_name = "Human (context not given)",
                           legend_title = "Rating Source",
                           facet_by = "AdjectiveClass",
                           bigrams = NULL,
                           sorted_bigrams = FALSE,
                           facet_wrap_width = NULL,
                           vertical = TRUE,
                           bigram_wrap_length = NULL) {
  model_names = names(model_names_colors)
  named_long_dists <- mapply(longer_and_name_dist, dists, model_names, SIMPLIFY = FALSE)
  
  bind_rows(named_long_dists) %>% mutate(Model = factor(Model, levels = model_names)) %>%
    select(!HumanOrLM) -> 
    human_and_lm_plot_dists
  
  if (!is.null(bigrams)) {
    human_and_lm_plot_dists %>%
      filter(Bigram %in% bigrams) -> 
      human_and_lm_plot_dists 
    if (sorted_bigrams == TRUE) {
      human_and_lm_plot_dists %>%
        mutate(Bigram = factor(Bigram, levels=bigrams)) -> human_and_lm_plot_dists
    } else {
      human_and_lm_plot_dists %>%
        mutate(Bigram = fct_relevel(Bigram, sort)) -> human_and_lm_plot_dists
    }
  }
  
  if (facet_by == "Adjective") {
    human_and_lm_plot_dists %>%
      mutate(Adjective = fct_relevel(Adjective,
                                    "artificial", "counterfeit", "fake", "false", "former", "knockoff")) -> 
      human_and_lm_plot_dists
  }
  
  if (facet_by != "Bigram") {
    # Need to calculate averaged probability distributions
    human_and_lm_plot_dists %>%
      group_by(Model, NumRating) %>%
      group_by_at(facet_by, .add = TRUE) %>%
      summarize(Probability = mean(Probability), .groups = "drop") ->
      human_and_lm_plot_dists
  }
  
  if (!is.null(bigram_wrap_length)) {
    human_and_lm_plot_dists$Bigram <- insert_linebreaks(human_and_lm_plot_dists$Bigram,
                                                        width = bigram_wrap_length)
  }
  
  human_and_lm_plot_dists %>%
    filter(Model != human_name) %>%
    ggplot(aes(x=NumRating, y=Probability)) + 
    geom_col(aes(fill=Model), 
             position="identity") +
    geom_col(data=human_and_lm_plot_dists %>%
               filter(Model == human_name) %>%
               mutate(Probability = -Probability) %>%
               select(!Model),
             aes(fill=human_name),
             position="identity") +
    scale_y_continuous(labels = abs, limits=c(-1,1)) +
    xlab("Rating") +
    scale_fill_manual(name=legend_title,
                      breaks=model_names,
                      values=model_names_colors) +
    theme_minimal() -> plot
  
  if (vertical == TRUE) {
    plot +
      coord_flip() ->
      plot
  }
  
  if (facet_by == "AdjectiveClass") {
    plot +
      facet_grid(AdjectiveClass ~ Model) ->
      plot
  } else if (facet_by == "Adjective") {
    plot +
      facet_grid(Model ~ Adjective) ->
      plot
  } else if (facet_by == "Bigram") {
    if (length(model_names_colors) == 2) {
      plot +
        facet_wrap(~ Bigram, ncol=facet_wrap_width) -> plot
    } else {
      plot + 
        facet_grid(Model ~ Bigram) -> 
        plot
    }
  } else {
    plot +
      facet_grid(~ Model) ->
      plot
  }
    
  return(plot)
}

# Split violin plots ----

##  Code for split violin plot ---- 
# https://stackoverflow.com/questions/35717353/split-violin-plot-with-ggplot2

GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, 
                           draw_group = function(self, data, ..., draw_quantiles = NULL) {
                             data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
                             grp <- data[1, "group"]
                             newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), if (grp %% 2 == 1) y else -y)
                             newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
                             newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
                             
                             if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
                               stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <=
                                                                         1))
                               quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
                               aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
                               aesthetics$alpha <- rep(1, nrow(quantiles))
                               both <- cbind(quantiles, aesthetics)
                               quantile_grob <- GeomPath$draw_panel(both, ...)
                               ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
                             }
                             else {
                               ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
                             }
                           })

geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., 
                              draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE, 
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes, 
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}

## Human vs LLM split violin plot

human_lm_split_violin_poster <- function(isa_data, lm_data, nouns, adjective="fake", poster=TRUE, title=TRUE) {
  human_lm_ratings <- build_human_lm_ratings(isa_data, lm_data, "LLM")
  human_lm_ratings %>%
    filter(HumanOrLM %in% c(label.human, "LLM")) %>%
    filter(Adjective == adjective) %>%
    #  filter(Bigram %in% c("fake laugh", "fake concert", "fake jacket")) %>%
    filter(Noun %in% nouns) -> filtered_ratings
  
  filtered_ratings %>%
    ggplot(aes(x=reorder_within(x=Noun,by=NumRating,
                                within=Adjective,fun=median), #Noun, 
               y=NumRating, fill=HumanOrLM)) +
    geom_split_violin(adjust=3) +
    xlab("Noun") +
    ylab("Rating") + 
    scale_fill_manual(name='Data Source',
                      breaks=c('Human', 
                               'LLM'),
                      values=c('Human'='#1E88E5', 
                               'LLM'='#D81B60'
                      )) +
    scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
    guides(x = guide_axis(angle = 90), 
           fill=guide_legend(ncol=2)) +
    theme_minimal() +
    theme(legend.position="bottom") -> plot
  
  if (title == TRUE) {
    plot +
      ggtitle("Human vs. LLM distribution for 'Is a fake N still an N?") ->
      plot
  }
  
  if (poster == TRUE) {
    plot +
      theme(text=element_text(size=36, color="#2C365E"),
            
            panel.grid.major.x = element_blank(),
            #        panel.grid.minor.x = element_blank(),
            panel.grid.major.y = element_blank(),
            #        panel.grid.minor.y = element_blank()
      ) -> plot
  }
  
  return(plot)
}
