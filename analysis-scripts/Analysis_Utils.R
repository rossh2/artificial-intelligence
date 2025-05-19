library(tidyverse)
library(tidytext)
library(extrafont) # Make Windows fonts available

# Colour scale ----

magenta_color = "#DC267F"
light_blue_color = "#1E88E5"
dark_blue_color = "#2C365E"
dark_yellow_color = "#FFC107"
mid_green_color = "#009E73"
paper_color_scale = c(magenta_color, light_blue_color, dark_yellow_color, mid_green_color,
                      "#5D3A9B", "#E66100", "#004D40", "#CC79A7", "#F0E442")

names_to_colors <- function(names_list) {
  named_colors <- paper_color_scale[1:length(names_list)]
  names(named_colors) <- names_list
  return(named_colors)
}

# Qualtrics preprocessing ----

preprocess_qualtrics_wide <- function(wide_data) {
  # Remove the first two rows, which contain the header again and the full question text
  wide_data <- wide_data[-c(1,2),]
  rownames(wide_data) <- NULL
  
  # Select only question columns; drop all user information & comments except demographic questionnaire
  wide_data <- subset(wide_data, 
                      select = grep("^(Q.|EnglishBefore5|Dialect|OtherLanguages|Comments)", 
                                    names(wide_data)))
  
  # Add fake UserID (don't want to use actual Prolific ID)
  wide_data$UserId <- 1:nrow(wide_data)
  # Drop actual Prolific ID
  if ("Q1" %in% names(wide_data)) {
    wide_data %>% select(!Q1) -> wide_data
  }
  
  return(wide_data)
}

# Merge frequency ----

add_frequency <- function(isa_data) {
  if (!("Adjective" %in% names(isa_data))) {
    isa_data %>%
      separate_wider_delim(Bigram, names=c("Adjective", "Noun"), delim=" ",
                           cols_remove = FALSE,
                           too_many = "merge") %>%
      mutate(across(c(Adjective, Noun), as.factor)) ->
      isa_data
  }
  
  bigrams_freqs <- read.csv("../Adjectives-PythonCode/output/filtering_data/all_c4_3979_bigrams_with_frequencies.csv", header = TRUE)
  
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
  
  adjective_classes <- read.csv("../Adjectives-PythonCode/data/adjective_classes.csv", header = TRUE)
  
  if (!("AdjectiveClass" %in% names(isa_data))) {
    isa_data %>% 
      merge(adjective_classes, by=c("Adjective"), all.x=TRUE) %>%
      mutate_at(c("AdjectiveClass"), factor) -> isa_data_combined
  } else {
    isa_data -> isa_data_combined
  }
  
  if (!("Frequency" %in% names(isa_data))) {
    isa_data_combined %>%
      merge(unique_bigrams_freqs %>% dplyr::select("Bigram", "Count", "Frequency", "CoarseFrequency"), 
            by=c("Bigram"),
            all.x=TRUE) %>%
      mutate(Frequency = droplevels(Frequency)) -> isa_data_combined
  }

  return(isa_data_combined)

}

# Add mean and variance ----

calculate_variance <- function(isa_data) {
  if('Experiment' %in% names(isa_data)) {
    isa_variance <- isa_data %>%
      group_by(Bigram) %>%
      summarise(Mean = mean(NumRating), SD = sd(NumRating), SE = SD / sqrt(n()), Variance = var(NumRating),
                Adjective = unique(Adjective), Noun = unique(Noun), AdjectiveClass = unique(AdjectiveClass),
                Frequency = unique(Frequency), Count = unique(Count), CoarseFrequency = unique(CoarseFrequency),
                Experiment = first(Experiment)) %>%
      ungroup()
  } else {
    isa_variance <- isa_data %>%
      group_by(Bigram) %>%
      summarise(Mean = mean(NumRating), SD = sd(NumRating), SE = SD / sqrt(n()), Variance = var(NumRating),
                Adjective = unique(Adjective), Noun = unique(Noun), AdjectiveClass = unique(AdjectiveClass),
                Frequency = unique(Frequency), Count = unique(Count), CoarseFrequency = unique(CoarseFrequency)) %>%
      ungroup()
  }
  
  return(isa_variance)
}

merge_variance <- function(isa_data, isa_variance) {
  isa_data %>%
    merge(isa_variance %>% dplyr::select(Bigram, Mean, SD, SE, Variance), 
          by = "Bigram") -> isa_data_combined
  return(isa_data_combined)
}

# Plots ----

plot_single_adjective_poster <- function(isa_data_12_capped, adjective, adj_det, nouns) {
  return(isa_data_12_capped %>%
           filter(Adjective == adjective) %>%
           filter(Noun %in% nouns) %>%
           ggplot(aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
                      y=NumRating)) + 
           geom_jitter(width=0.2, height=0.2, alpha=0.4, size=4, color="#2C365E") + 
           geom_point(aes(y=Mean), size=7, position=position_dodge(width=0.2), color="#2C365E") +
           geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                         width = 0.2, 
                         linewidth=2, color="#2C365E") +
           ggtitle(sprintf("Human ratings for 'Is %s %s N still an N?'", adj_det, adjective)) + 
           xlab("Noun") +
           ylab("Rating") + 
           scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
           scale_y_continuous(breaks=1:5, limits = c(0.5,5.5)) +
           guides(x = guide_axis(angle = 90)) +
           theme_minimal() + 
           theme(text=element_text(size=36, color="#2C365E"),
                 legend.position="bottom",
                 #        panel.grid.major.x = element_blank(),
                 panel.grid.minor.x = element_blank(),
                 #        panel.grid.major.y = element_blank(),
                 panel.grid.minor.y = element_blank()
           )
  )
}

plot_josh_scatter_plot <- function(isa_data, rater_name = NULL, 
                                   nouns = NULL, adjectives = NULL, 
                                   facet_by_source = FALSE, 
                                   x_axis_bigrams = FALSE,
                                   poster = FALSE) {
  if (!is.null(nouns)) {
    isa_data %>%
      filter(Noun %in% nouns) ->
      isa_data
  }
  
  if (!is.null(adjectives)) {
    isa_data %>%
      filter(Adjective %in% adjectives) ->
      isa_data
  }
  
  isa_data %>% 
    mutate(CoarseFrequency = fct_recode(CoarseFrequency,
                                        "25th-50th pct." = "25th-50th percentile",
                                        "50th-75th pct." = "50th-75th percentile",
                                        "75th-90th pct." = "75th-90th percentile",
                                        "90th-99th pct." = "90th-99th percentile"
    )) -> isa_data
  
  if (poster == TRUE) {
    mean_dot_size = 6
    jitter_dot_size = 4
    error_bar_width = 0.6
    error_linewidth = 1.5
  } else {
    mean_dot_size = 3
    jitter_dot_size = 1
    error_linewidth = 1
    if (x_axis_bigrams == TRUE) {
      error_bar_width = 0.25
    } else {
      error_bar_width = 0.5
    }
  }
  
  if (facet_by_source == TRUE) {
    ggplot(isa_data, 
           aes(x=reorder_within(x=Noun,by=NumRating,within=HumanOrLM),
               y=NumRating, color=CoarseFrequency)) +
      facet_wrap(~ HumanOrLM, scales = "free_x") +
      guides(x = guide_axis(angle = 90)) +
      xlab("Noun") -> plot
  } else if (x_axis_bigrams == TRUE) {
    ggplot(isa_data %>%
             mutate(Bigram = fct_relevel(Bigram, sort)), 
           aes(x=Bigram,
               y=NumRating)) +
      guides(x = guide_axis(angle = 0)) +
      xlab("Bigram") -> plot
  } else {
    ggplot(isa_data, 
           aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
               y=NumRating, color=CoarseFrequency)) +
      guides(x = guide_axis(angle = 90)) +
      xlab("Noun") -> plot
    
    if (length(adjectives) > 1) {
      plot + 
        facet_wrap(~ Adjective, scales = "free_x")  -> plot
    }
  }
  
  plot + 
    geom_jitter(width=0.2, height=0.2, alpha=0.2, size=jitter_dot_size) + 
    geom_point(aes(y=Mean), size=mean_dot_size, position=position_dodge(width=0.2)) +
    geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                  width = error_bar_width, linewidth=error_linewidth,
                  position=position_dodge(width=0.2)) +
    
    ylab("Rating") + 
    theme_minimal() + 
    scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) +
    scale_y_continuous(breaks=1:5, limits = c(0.5,5.5)) +
    scale_color_discrete(name="Frequency") -> plot
  
  if (!is.null(rater_name)) {
    plot +
      ggtitle(sprintf("%s ratings for 'Is an AN still an N?'", rater_name)) -> plot
  } else {
    plot + 
      ggtitle("Ratings for 'Is an AN still an N?'", rater_name) -> plot
  }
  
  return(plot)
}
