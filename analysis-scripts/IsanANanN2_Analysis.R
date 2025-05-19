library(tidyverse)
library(tidytext)
library(ordinal)

source("Analysis_Utils.R")

# Preprocess analogy ISA experiment

isa2_wide <- read.csv("results/human/exp2_isa/isa_responses_part2_raw.csv",
                          fileEncoding = "UTF-8-BOM")

# Remove the first two rows, which contain the header again and the full question text
isa2_wide <- isa2_wide[-c(1,2),]
rownames(isa2_wide) <- NULL

colnames(isa2_wide)

# Select only question columns; drop all user information & comments except demographic questionnaire
isa2_wide <- subset(isa2_wide, 
                        select = grep("^(Q.|EnglishBefore5|Dialect|OtherLanguages|Comments)", 
                                      names(isa2_wide)))

# Add fake UserID (don't want to use actual Prolific ID)
isa2_wide$UserId <- 1:nrow(isa2_wide)
# Drop actual Prolific ID
isa2_wide %>% select(!Q1) -> isa2_wide

# Rename demographic questions
isa2_wide <- isa2_wide %>%
  rename(OtherEnglish = Dialect_2_TEXT)

# Merge fillers

isa2_wide %>%
  select(sort(tidyselect::peek_vars())) %>%  # Sort columns alphabetically
  unite(Q.knitted.pizza, Q.knitted.pizza:Q.knitted.pizza.9.2, sep="", remove = TRUE) %>%
  unite(Q.temporary.breakage, Q.temporary.breakage:Q.temporary.breakage.9.2, sep="", remove = TRUE) %>%
  unite(Q.wooden.pear, Q.wooden.pear:Q.wooden.pear.9.2, sep="", remove = TRUE) %>%
  unite(Q.orange.mouse, Q.orange.mouse:Q.orange.mouse.9.2, sep="", remove = TRUE) ->
  isa2_wide

# Exclusion criteria ----

## Demographic ----

dem_excluded_ids <- isa2_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & OtherEnglish != "American English")) %>% 
  pull(UserId)

dem_excluded_ids

## Attention checks ----

isa2_wide <- isa2_wide %>%
  mutate(AttnFailed = ifelse(Q.wooden.pear %in% c("Definitely yes", "Probably yes"), 1, 0)
  )

attn_excluded_ids <- isa2_wide %>%
  filter(AttnFailed >= 1) %>%
  pull(UserId)

attn_excluded_ids

excluded_ids <- isa2_wide %>%
  filter(UserId %in% dem_excluded_ids | UserId %in% attn_excluded_ids) %>%
  select(EnglishBefore5, Dialect, OtherEnglish, OtherLanguages, Q.orange.mouse, Q.wooden.pear)

isa2_wide_excl <- isa2_wide %>%
  filter(!UserId %in% dem_excluded_ids & !UserId %in% attn_excluded_ids)

nrow(isa2_wide_excl)

## Select target questions and pivot ----

names(isa2_wide_excl)

isa_analogy_data <- isa2_wide_excl %>%
  dplyr::select(!Comments:OtherLanguages) %>% # Exclude demographics
  dplyr::select(!c("Q.melted.ice", "Q.miniature.stepladder", "Q.green.square")) %>%  # Exclude training
  dplyr::select(!c("Q.knitted.pizza", "Q.temporary.breakage", "Q.orange.mouse", "Q.wooden.pear")) %>%  # Exclude fillers
  dplyr::select(!c("AttnFailed")) %>%
  pivot_longer(
    cols = Q.artificial.abundance:Q.wooden.horse,
    names_to = c("Adjective","Noun"),
    names_pattern = "([a-z]+).([a-z.]+)",
    values_to = "Rating"
  ) %>%
  mutate(Noun = fct_recode(Noun, 
                                "spring water" = "spring.water",

  )) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate_at(c("UserId", "Adjective", "Noun", "Bigram"), factor) 

isa_analogy_data <- isa_analogy_data %>%
  filter(Rating!="") %>%
  mutate(Rating = factor(Rating, levels = c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"))) %>%
  mutate(NumRating = as.integer(Rating))


str(isa_analogy_data)

## Balance ----

isa_analogy_data %>% 
  group_by(Bigram) %>%
  summarise(RatingCount = n()) %>%
  mutate(RatingThreshold = case_when(
    RatingCount > 12 ~ ">12",
    RatingCount == 12 ~ "12",
    RatingCount >= 10 ~ "10-11",
    RatingCount < 10 ~ "<10"
  )) %>%
  mutate(RatingThreshold = factor(RatingThreshold, c(">12", "12", "10-11", "<10"))) %>%
  group_by(RatingThreshold) %>%
  summarize(BigramCount = n())
# One list (8 bigrams) only has 9 ratings

max_participants_per_group = 12

set.seed(42)
isa_analogy_data_capped <- isa_analogy_data %>%
  group_by(Bigram) %>%
  slice_sample(n = max_participants_per_group) %>%
  ungroup()

# Combine with frequencies and original ISA experiment ----

isa_data_capped_copy <- as_tibble(read.csv("isa_data_capped.csv")) %>%
  mutate_at(c("Adjective", "Noun", "Bigram", "IsaRating", "ParticipantId"), factor)

isa_analogy_data_capped %>%
  mutate(UserId = paste0('ISA-A-', UserId),
         Experiment = "Analogy") %>%
  bind_rows(isa_data_capped_copy %>%
          select("Adjective", "Noun", "Bigram", "IsaRating", "NumIsaRating", "ParticipantId") %>%
          mutate(UserId = paste0('ISA-', ParticipantId),
                 Rating = IsaRating,
                 NumRating = NumIsaRating,
                 Experiment = "Original") %>%
          select(!c(ParticipantId, IsaRating, NumIsaRating))
  ) %>%
  mutate_at(c("Adjective", "Noun", "Bigram", "Rating", "UserId", "Experiment"), factor) -> 
  isa_data_combined

str(isa_data_combined)

add_frequency(isa_data_combined) -> isa_data_combined

## Add mean and variance ----

calculate_variance(isa_data_combined) -> isa_variance_combined
merge_variance(isa_data_combined, isa_variance_combined) -> isa_data_combined

str(isa_data_combined)

write.csv(isa_variance_combined, "isa_variance_combined.csv", row.names = FALSE)
write.csv(isa_data_combined, "isa_data_combined.csv", row.names = FALSE)

## Remove extra bigrams ----

exp_adjectives <- read.csv('../Adjectives-PythonCode/data/adjectives.txt', header = FALSE)

extra_nouns <- c("spring water", "intelligence")

isa_data_12_combined <- isa_data_combined %>%
  filter(Adjective %in% exp_adjectives$V1) %>%
  filter(!(Noun %in% extra_nouns))

isa_variance_12_combined <- isa_variance_combined %>%
  filter(Adjective %in% exp_adjectives$V1) %>%
  filter(!(Noun %in% extra_nouns))

write.csv(isa_variance_12_combined, "isa_variance_12_combined.csv", row.names = FALSE)

## Counts ----

isa_data_12_combined %>% 
  select(Noun) %>% 
  distinct(Noun) %>% 
  summarize(NounCount = n())

isa_data_12_combined %>% 
  group_by(Experiment) %>%
  select(Noun) %>% 
  distinct(Noun) %>% 
  summarize(NounCount = n())

isa_data_12_combined %>% 
  select(Bigram) %>% 
  distinct(Bigram) %>% 
  summarize(BigramCount = n())

isa_data_12_combined %>% 
  group_by(Experiment) %>%
  select(Bigram) %>% 
  distinct(Bigram) %>% 
  summarize(BigramCount = n())

isa_data_12_combined %>% 
  group_by(Experiment, Bigram) %>%
  summarise(RatingCount = n()) %>%
  mutate(RatingThreshold = case_when(
    RatingCount > 12 ~ ">12",
    RatingCount == 12 ~ "12",
    RatingCount >= 10 ~ "10-11",
    RatingCount < 10 ~ "<10"
  )) %>%
  mutate(RatingThreshold = factor(RatingThreshold, c(">12", "12", "10-11", "<10"))) %>%
  group_by(Experiment, RatingThreshold) %>%
  summarize(BigramCount = n())

# Plot ----

## Josh-style scatter plots with error bars ----

### Both experiments ----

ggplot(isa_data_12_combined, 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating, color=Experiment)) + 
  geom_jitter(width=0.2, height=0.2, alpha=0.2) + 
  geom_point(aes(y=Mean), size=3, position=position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                width = 0.2, position=position_dodge(width=0.2)) +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for 'Is an AN still an N?' (colored by experiment)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_combined_bigrams_by_adjective_by_experiment_errorbars.png", 
       width=7000, height=2500, dpi=300, units="px")

ggplot(isa_data_12_combined, 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating, color=CoarseFrequency)) + 
  geom_jitter(width=0.2, height=0.2, alpha=0.2) + 
  geom_point(aes(y=Mean), size=3, position=position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                width = 0.2, position=position_dodge(width=0.2)) +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for 'Is an AN still an N?'") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_combined_bigrams_by_adjective_with_frequency_errorbars.png", 
       width=7000, height=2500, dpi=300, units="px")

plot_josh_scatter_plot(isa_data_12_combined, 
                       adjectives=c("fake", "counterfeit"), nouns=nouns,
                       poster = TRUE) +
  theme(text=element_text(size=24))
ggsave("plots/isa_plot_combined_bigrams_fake_cf_with_frequency_errorbars.png", 
       width=17, height=7, dpi=300, units="in")

plot_josh_scatter_plot(isa_data_12_combined, 
                       adjectives=c("fake", "counterfeit"), nouns=nouns) +
  ggtitle(NULL)
ggsave("plots/isa_plot_combined_bigrams_fake_cf_with_frequency_errorbars_paper.png", 
       width=9, height=3, dpi=300, units="in")


plot_josh_scatter_plot(isa_data_12_combined, 
                       adjectives=c("fake", "counterfeit"), nouns=nouns) +
  ggtitle(NULL) +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=9),
        legend.margin = margin(t = 6, r = 0, b = 6, l = 0),
        legend.box.margin = margin(t = 6, r = 0, b = 6, l = 0)
        )
ggsave("plots/isa_plot_combined_bigrams_fake_cf_with_frequency_errorbars_diss.png", 
       width=6.5, height=2.5, dpi=300, units="in")


plot_josh_scatter_plot(isa_data_combined, 
                       adjectives=c("fake", "stone", "wooden", "velvet"), 
                       nouns=c("gun", "lion", "horse", "rabbit"),
                       x_axis_bigrams=TRUE) +
  ggtitle(NULL) +
  theme(text = element_text(size=12, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  )
ggsave("plots/isa_plot_combined_bigrams_material_errorbars_diss.png", 
       width=5, height=2.5, dpi=300, units="in")

### Original experiment ----

ggplot(isa_data_12_combined %>% 
         filter(Experiment == "Original"), 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating, color=CoarseFrequency)) + 
  geom_jitter(width=0.2, height=0.2, alpha=0.2) + 
  geom_point(aes(y=Mean), size=3, position=position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                width = 0.2,
                position=position_dodge(width=0.2)) +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for 'Is an AN still an N?' (Original experiment)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_original_bigrams_by_adjective_with_frequency_errorbars.png", 
       width=6000, height=2500, dpi=300, units="px")

ggplot(isa_data_12_combined %>% 
         filter(Experiment == "Original") %>%
         filter(Adjective %in% c("tiny", "homemade")) %>%
         mutate(Adjective = fct_relevel(fct_drop(Adjective), "tiny")) %>%
         mutate(CoarseFrequency = fct_recode(CoarseFrequency,
                                       "25th-50th pct." = "25th-50th percentile",
                                       "50th-75th pct." = "50th-75th percentile",
                                       "75th-90th pct." = "75th-90th percentile",
                                       "90th-99th pct." = "90th-99th percentile"
         )), 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating, color=CoarseFrequency)) + 
  geom_jitter(width=0.2, height=0.2, alpha=0.2) + 
  geom_point(aes(y=Mean), size=3, position=position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                width = 0.5,
                position=position_dodge(width=0.2)) +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for 'Is an AN still an N?'") + # (Original experiment)") + 
  xlab("Noun") +
  ylab("Rating") + 
  theme_minimal() + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  scale_color_discrete(name="Frequency") +
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_original_bigrams_subsective_by_adjective_with_frequency_errorbars.png", 
       width=8, height=3, dpi=300, units="in")

ggplot(isa_data_12_combined %>% 
         filter(Experiment == "Original") %>%
         filter(Adjective %in% c("fake", "counterfeit")) %>%
         mutate(CoarseFrequency = fct_recode(CoarseFrequency,
                                             "25th-50th pct." = "25th-50th percentile",
                                             "50th-75th pct." = "50th-75th percentile",
                                             "75th-90th pct." = "75th-90th percentile",
                                             "90th-99th pct." = "90th-99th percentile"
         )), 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating, color=CoarseFrequency)) + 
  geom_jitter(width=0.2, height=0.2, alpha=0.2) + 
  geom_point(aes(y=Mean), size=3, position=position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                width = 0.5,
                position=position_dodge(width=0.2)) +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for 'Is an AN still an N?'") + # (Original experiment)") + 
  xlab("Noun") +
  ylab("Rating") + 
  theme_minimal() + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) +
  scale_color_discrete(name="Frequency") +
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_original_bigrams_privative_by_adjective_with_frequency_errorbars.png", 
       width=8, height=3, dpi=300, units="in")

### Analogy experiment ----

ggplot(isa_data_12_combined %>% 
         filter(Experiment == "Analogy"), 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating, color=CoarseFrequency)) + 
  geom_jitter(width=0.2, height=0.2, alpha=0.2) + 
  geom_point(aes(y=Mean), size=3, position=position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                width = 0.2,
                position=position_dodge(width=0.2)) +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for 'Is an AN still an N?' (Analogy experiment)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_analogy_bigrams_by_adjective_with_frequency_errorbars.png", 
       width=6000, height=2500, dpi=300, units="px")

### Intelligence ----

ggplot(isa_data_combined %>% filter(Noun == "intelligence"), 
       aes(x=Bigram, y=NumRating)) + 
  geom_jitter(width=0.2, height=0.2, alpha=0.2) + 
  geom_point(aes(y=Mean), size=3, position=position_dodge(width=0.2)) +
  geom_errorbar(aes(ymin = Mean - 1.96 * SE, ymax = Mean + 1.96 * SE), 
                width = 0.2, position=position_dodge(width=0.2)) +
  ggtitle("Ratings for 'Is A intelligence still intelligence?'") + 
  xlab("Bigram") +
  ylab("Rating") + 
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_intelligence_ratings.png")

## Violin/box plots over means ----

### Overall ----

ggplot(isa_variance_12_combined,
       aes(x=Adjective, y=Mean)) +
  geom_violin(data=isa_variance_12_combined, scale="area", linewidth=0.5) +
  geom_jitter(aes(col=AdjectiveClass), height = 0, width = 0.05, size = 0.75, alpha=0.4) +
  # geom_boxplot(aes(alpha=0.01), outlier.shape = NA) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  xlab("Adjective") +
  ylab("Mean rating") +
  facet_wrap(~ AdjectiveClass, scales = "free_x") +
  scale_color_manual(values=c('Subsective'=light_blue_color, 'Privative'=magenta_color)) +
  theme_minimal() +
  theme(legend.position='none') -> combined_mean_violin_plot
combined_mean_violin_plot +
  theme(text=element_text(size=14), strip.text.x = element_blank()) +
  ggtitle("Mean bigram ratings for 'Is an A N an N?'")
ggsave(filename="plots/isa_plot_combined_mean_ratings_violin_no_frequency.png",
       units="in", width=9, height=3.5, dpi=300)

combined_mean_violin_plot +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(),
  )
ggsave(filename="plots/isa_plot_combined_mean_ratings_violin_no_frequency_paper.png",
       units="in", width=8, height=2, dpi=300)

combined_mean_violin_plot +
  theme(
    strip.background = element_blank(),
    strip.text.x = element_blank(),
    text=element_text(size=12, family = "Palatino Linotype"),
    axis.text.x = element_text(size=10, angle=90, hjust=1)
  )
ggsave(filename="plots/isa_plot_combined_mean_ratings_violin_no_frequency_diss.png",
       units="in", width=6.5, height=2.5, dpi=300)

### Comparison between experiments ----

ggplot(isa_variance_12_combined,
       aes(x=Adjective, y=Mean, color=Experiment)) +
  geom_violin() +
  facet_wrap(~ AdjectiveClass, scales = "free_x") +
  guides(x = guide_axis(angle = 90))

ggplot(isa_variance_12_combined,
       aes(x=Adjective, y=Mean, color=Experiment)) +
  geom_boxplot() +
  facet_wrap(~ AdjectiveClass, scales = "free_x") +
  guides(x = guide_axis(angle = 90))
ggsave("plots/isa_plot_analogy_vs_original_means_boxplot.png")

## Variance vs. count ----

ggplot(isa_variance_12_combined %>%
         mutate(Count = replace(Count, Count == 0, 1)), # Pretend zero counts are 1 for log-scale
       aes(x=Count, y=Variance)) + 
  geom_point(aes(color=Frequency)) + 
  geom_smooth(method='lm', formula = y ~ x) +
  ggtitle("Effect of frequency on variance") +
  scale_x_continuous(trans='log10') +
  xlab("Bigram Count (log scale)") +
  ylab("Rating Variance") +
  facet_wrap(~ AdjectiveClass) +
  theme_minimal() +
  theme(text=element_text(size=14)) +
ggsave(filename="plots/isa_plot_combined_count_variance_scatter_with_freqs_fitted.png", 
       units="in", width = 7, height = 4, dpi=300)

# Frequency regression ----

combined_variance_freq_lm <- lm(Variance ~ CoarseFrequency, data = isa_variance_combined)
summary(combined_variance_freq_lm)

combined_variance_count_lm <- lm(Variance ~ Count + AdjectiveClass, data = isa_variance_combined)
summary(combined_variance_count_lm)

combined_int_variance_count_lm <- lm(Variance ~ Count, data = isa_variance_combined %>% filter(AdjectiveClass == "Subsective"))

summary(combined_int_variance_count_lm)

combined_int_variance_count_rsquared <- summary(combined_int_variance_count_lm)$r.squared
combined_int_variance_count_adj_rsquared <- summary(combined_int_variance_count_lm)$adj.r.squared

combined_priv_variance_count_lm <- lm(Variance ~ Count, data = isa_variance_combined %>% filter(AdjectiveClass == "Privative"))

summary(combined_priv_variance_count_lm)

combined_priv_variance_count_rsquared <- summary(combined_priv_variance_count_lm)$r.squared
combined_priv_variance_count_adj_rsquared <- summary(combined_priv_variance_count_lm)$adj.r.squared

## Explore variance/frequency ----

isa_variance_combined %>% filter(AdjectiveClass == "Subsective") %>% arrange(Mean) %>% View()

isa_variance_combined %>% filter(Frequency == "Zero") %>% arrange(Variance) %>% View()

isa_variance_combined %>% filter(Frequency == "Zero" & AdjectiveClass == "Privative") %>% arrange(Variance) %>% View()

isa_variance_combined %>% filter(CoarseFrequency == "90th-99th percentile") %>% arrange(desc(Variance)) %>% View()
