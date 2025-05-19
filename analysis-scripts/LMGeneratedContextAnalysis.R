library(tidyverse)
library(tidytext)
library(ordinal)
library(lmerTest)
library(viridis)
library(performance)
library(MuMIn)
library(rcompanion)
library(ggeffects)
library(effects)
library(patchwork)

source("Analysis_Utils.R")
source("LM_Analysis_Utils.R")

# Load data ----

read.csv("results/llm/gen_context/predictions_Meta-Llama-3-70B-Instruct_context-labelledscale-5shot.csv") %>%
  preprocess_labelled_responses() %>%
  select(Bigram, Adjective, Noun, ContextBias, PredictedResponse, NumPredictedResponse) %>%
  rename(NumRating = NumPredictedResponse, Rating = PredictedResponse) ->
  llama3i_gen_context_responses

# All bigrams incl. analogy, with chat template
read.csv("results/llm/isa/predictions_isa_Meta-Llama-3-70B-Instruct_labelledscale-5shot.csv") %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3i_lscale_5shot_responses


# Preprocessing ----

## Normalization ----

llama3i_gen_context_responses %>%
  distinct(Bigram) %>%
  summarize(n())
# 781 bigrams that the LM was willing to generate for

llama3i_gen_context_responses %>%
  group_by(Bigram) %>%
  summarize(n=n()) %>%
  filter(n != 12) %>%
  print(n=Inf)
# Some bigrams only have 3 contexts for some reason, a bunch involve 'illegal' so we were probably hitting guardrails

# Sample just 12 ratings for cases where we have 24 and exclude bigrams that have < 10

llama3i_gen_context_responses %>%
  group_by(Bigram) %>%
  summarize(n=n()) %>%
  filter(n < 10) %>%
  pull(Bigram) ->
  too_few_bigrams


set.seed(42)
llama3i_gen_context_responses %>%
  filter(!(Bigram %in% too_few_bigrams)) %>%
  group_by(Bigram) %>%
  slice_sample(n = 12) %>%
  ungroup() ->
  llama3i_gen_context_responses_capped

llama3i_gen_context_responses_capped %>%
  distinct(Bigram) %>%
  summarize(n())
# 757 bigrams that we have at least 10 ratings for

str(llama3i_gen_context_responses_capped)

## Sampling ----

llama3i_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) -> 
  llama3i_lscale_5shot_sampled_responses

## Add frequency and variance ----

llama3i_gen_context_responses_capped %>%
  add_frequency() ->
  llama3i_gen_context_responses_capped

calculate_variance(llama3i_gen_context_responses_capped) -> llama3i_gen_context_variance
merge_variance(llama3i_gen_context_responses_capped, 
               llama3i_gen_context_variance) -> llama3i_gen_context_responses_combined

str(llama3i_gen_context_responses_capped)

llama3i_lscale_5shot_sampled_responses %>%
  add_frequency() ->
  llama3i_lscale_5shot_sampled_responses

merge_variance(llama3i_lscale_5shot_sampled_responses, calculate_variance(llama3i_lscale_5shot_sampled_responses)) ->
  llama3i_lscale_5shot_sampled_responses

str(llama3i_lscale_5shot_sampled_responses)

# Plots ----

## Scatter plots ----

plot_josh_scatter_plot(llama3i_gen_context_responses_combined, "Llama 3 Instruct")
ggsave('plots/lm_isa_gen_context_scatter_by_frequency.png', width = 20, height = 8, unit='in', dpi=300)

plot_josh_scatter_plot(isa_data_12_combined, "Human")
ggsave('plots/lm_isa_gen_context_human_comparison.png', width = 20, height = 8, unit='in', dpi=300)

plot_josh_scatter_plot(human_lm_gen_ratings %>%
                         mutate(HumanOrLM = fct_recode(HumanOrLM, "Llama 3 70B Instruct (generated contexts)" = "LLM",
                                                       "Human (context not given)" = "Human")), 
                       adjectives=c("fake"), nouns=nouns,
                       facet_by_source=TRUE, poster = TRUE) +
  theme(text=element_text(size=24))
ggsave("plots/lm_isa_gen_context_human_vs_llm_fake_scatter_by_freq.png", 
       width=17, height=7, dpi=300, units="in")


## Violin plots ----

human_lm_split_violin_poster(isa_data_12_combined, llama3i_gen_context_responses_combined, llama3i_gen_context_responses_combined %>% pull(Noun))

human_lm_split_violin_poster(isa_data_12_combined, llama3i_gen_context_responses_combined, llama3i_gen_context_responses_combined %>% pull(Noun),
                             adjective="useful")

nouns = c("dollar", "concert", "fire", "abundance", "air", "door",
          "gun", "crowd", "reef", "photograph","information",
          "jacket", "report", "laugh")
human_lm_split_violin_poster(isa_data_12_combined, llama3i_gen_context_responses_combined, nouns)

spv_nouns_1 = c("dollar", "fire",  "reef",  
              "rumor")
spv_nouns_2 = c("dollar", "fire",  "reef",  
                "crowd", "concert")
human_lm_split_violin_poster(isa_data_12_combined, 
                             llama3i_lscale_5shot_sampled_responses,
                             spv_nouns_1, poster=FALSE)
ggsave("plots/lm_isa_splitviolin_fake_selected_nouns_llama3i_sampled_1.png", units='in', width=4, height=4)

human_lm_split_violin_poster(isa_data_12_combined, 
                             llama3i_lscale_5shot_sampled_responses,
                             spv_nouns_2, title = FALSE) +
  theme(text=element_text(size=24))
ggsave("plots/lm_isa_splitviolin_fake_selected_nouns_llama3i_sampled_2.png", units='in', width=8, height=7)

human_lm_split_violin_poster(isa_data_12_combined, 
                             llama3i_gen_context_responses_capped,
                             spv_nouns_2, title = FALSE) +
  theme(text=element_text(size=24))
ggsave("plots/lm_isa_splitviolin_fake_selected_nouns_llama3i_gen.png", units='in', width=8, height=7)

spv_nouns_useful = c("fire",  "reef", "sign", "watch", "attack")

human_lm_split_violin_poster(isa_data_12_combined, 
                             llama3i_lscale_5shot_sampled_responses,
                             adjective="useful",
                             spv_nouns_useful, title = FALSE) +
  theme(text=element_text(size=24))
ggsave("plots/lm_isa_splitviolin_useful_selected_nouns_llama3i_sampled.png", units='in', width=8, height=7)

human_lm_split_violin_poster(isa_data_12_combined, 
                             llama3i_gen_context_responses_capped,
                             adjective="useful",
                             spv_nouns_useful, title = FALSE) +
  theme(text=element_text(size=24))
ggsave("plots/lm_isa_splitviolin_useful_selected_nouns_llama3i_gen.png", units='in', width=8, height=7)


## Heat maps ----

isa_data_12_combined %>%
  merge(llama3i_gen_context_responses_capped %>%
          select(Bigram, Rating, NumRating) %>%
          rename(PredictedResponse = Rating,
                 NumPredictedResponse = NumRating),
        .by="Bigram") %>%
  group_by(AdjectiveClass, Rating, PredictedResponse) %>%
  summarize(Count = dplyr::n(), .groups = "drop") -> human_vs_llama3i_gen_responses

isa_data_12_combined %>%
  merge(llama3i_lscale_5shot_sampled_responses %>%
          select(Bigram, Rating, NumRating) %>%
          rename(PredictedResponse = Rating,
                 NumPredictedResponse = NumRating) %>%
          mutate(PredictedResponse = factor(PredictedResponse, levels=c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"))),
        .by="Bigram") %>%
  group_by(AdjectiveClass, Rating, PredictedResponse) %>%
  summarize(Count = dplyr::n(), .groups = "drop") -> human_vs_llama3i_sampled_responses

human_vs_llama3i_gen_responses %>%
  ggplot(aes(x=PredictedResponse, y=Rating, fill=Count)) +
  geom_tile() +
  facet_wrap(~ AdjectiveClass) +
  labs(title="Correlation between LLM and human ratings, by adjective type",
       x="Llama 3 Instruct (context generation)", y="Human Rating (context not given)") +
  guides(x = guide_axis(angle = 90)) + 
  theme_minimal() + 
  theme(legend.position="right")  +
  scale_fill_viridis(option = "plasma")


human_vs_llama3i_sampled_responses %>%
  ggplot(aes(x=PredictedResponse, y=Rating, fill=Count)) +
  geom_tile() +
  facet_wrap(~ AdjectiveClass) +
  labs(title="Correlation between LLM and human ratings, by adjective type",
       x="Llama 3 Instruct (logprob sampling)", y="Human Rating (context not given)") +
  guides(x = guide_axis(angle = 90)) + 
  theme_minimal() + 
  theme(legend.position="right") +
  scale_fill_continuous()
  scale_fill_viridis(option = "plasma")
  
## Distributions ----

llama3i_gen_context_responses_capped %>%
  filter(Bigram == "fake crowd") %>%
  ggplot(aes(x=NumRating)) +
  geom_density(fill="#D81B60", alpha=1) +
  labs(y="Density", x="Rating", title="") +
  ggtitle("") +
  xlim(c(1,5)) +   
  theme_minimal() + 
  theme(text=element_text(size=36, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank()
  )
ggsave("plots/llama3i_fake_crowd_generated_rating_density_poster.png", units='in', height=6, width=8, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake crowd")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_crowd_rating_bar_plot.png", units='in', height=3, width=4, dpi=300)

human_dist %>%
  filter(Bigram == "fake crowd") %>%
  mutate("Definitely not" = 1/6, "Probably yes" = 0.25) %>%
  single_bar_plot(dist_color=light_blue_color, bigrams = c("fake crowd")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/hypothetical_fake_crowd_rating_bar_plot_blue.png", units='in', height=3, width=4, dpi=300)

llama3i_gen_dist %>%
  single_bar_plot(dist_color=light_blue_color, bigrams = c("fake crowd")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),

        panel.grid.minor.y = element_blank())
ggsave("plots/llama3i_fake_crowd_generated_rating_bar_plot.png", units='in', height=3, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake concert")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_concert_rating_bar_plot.png", units='in', height=3, width=4, dpi=300)

an_context_combined_plus %>%
  add_frequency() %>%
  filter(ContextBias == "subsective") %>%
  build_human_dist() %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake scarf")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_scarf_subsective_rating_bar_plot.png", units='in', height=3, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake scarf")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_scarf_rating_bar_plot.png", units='in', height=3, width=4, dpi=300)

an_context_combined_plus %>%
  add_frequency() %>%
  filter(ContextBias == "subsective") %>%
  build_human_dist() %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake reef")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_reef_subsective_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake reef")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_reef_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

an_context_combined_plus %>%
  add_frequency() %>%
  filter(ContextBias == "privative") %>%
  build_human_dist() %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake fire")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_fire_privative_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("fake fire")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_fake_fire_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

human_dist_extra %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("artificial intelligence")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_ai_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

llama3i_dist %>%
  single_bar_plot(dist_color=light_blue_color, bigrams = c("artificial intelligence")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        
        panel.grid.minor.y = element_blank())
ggsave("plots/llama3i_ai_logprob_rating_bar_plot.png", units='in', height=3, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("homemade cat")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_homemade_cat_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("counterfeit watch")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_counterfeit_watch_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("homemade sweater")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_homemade_sweater_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

human_dist %>%
  single_bar_plot(dist_color=magenta_color, bigrams = c("counterfeit scarf")) +
  theme(text=element_text(size=24, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/human_counterfeit_scarf_rating_bar_plot.png", units='in', height=2.5, width=4, dpi=300)

# JS divergence ----

## Calculate ----

human_lm_gen_ratings <- build_human_lm_ratings(isa_data_12_combined, llama3i_gen_context_responses_combined, "LLM")
human_lm_sampled_ratings <- build_human_lm_ratings(isa_data_12_combined, llama3i_lscale_5shot_sampled_responses, "LLM")

calculate_divergences(human_lm_gen_ratings, "LLM", js_only = TRUE) -> gen_divergences
gen_divergences

gen_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

gen_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

gen_divergences %>%
  mutate(CoarseFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency",
    .default = "Low frequency"
  )) %>%
  group_by(CoarseFrequency) %>% 
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

calculate_divergences(human_lm_sampled_ratings, "LLM", js_only = TRUE) -> sampled_divergences
sampled_divergences

bind_rows(gen_divergences %>% 
            mutate(Method = "Context generation"), 
          #sampled_divergences %>%
          llama3i_human_lm_divergences %>%
            mutate(Method = "Log-probability")
) -> both_divergences

isa_data_12_combined %>%
  group_by(AdjectiveClass, Rating, .drop=FALSE) %>%
  summarize(Count = dplyr::n(), .groups = "drop_last") %>%
  mutate(HumanOrLM = "Human (no context)") ->
  total_human_distribution

llama3i_gen_context_responses_combined %>%
  group_by(AdjectiveClass, Rating, .drop=FALSE) %>%
  summarize(Count = dplyr::n(), .groups = "drop_last") %>%
  mutate(HumanOrLM = "LLM (generated)") ->
  total_gen_distribution

llama3i_lscale_5shot_sampled_responses %>%
  group_by(AdjectiveClass, Rating, .drop=FALSE) %>%
  summarize(Count = dplyr::n(), .groups = "drop_last") %>%
  mutate(HumanOrLM = "LLM (sampled)") ->
  total_sampled_distribution

bind_rows(total_human_distribution, total_gen_distribution, total_sampled_distribution) ->
  total_distributions

isa_data_12_combined %>%
  group_by(AdjectiveClass, Rating, .drop=FALSE) %>%
  summarize(Count = dplyr::n(), .groups = "drop_last") %>%
  mutate(HumanOrLM = "Human (no context)") ->
  total_human_distribution

llama3i_gen_context_responses_combined %>%
  group_by(AdjectiveClass, Rating, .drop=FALSE) %>%
  summarize(Count = dplyr::n(), .groups = "drop_last") %>%
  mutate(HumanOrLM = "LLM (generated)") ->
  total_gen_distribution

llama3i_lscale_5shot_sampled_responses %>%
  group_by(AdjectiveClass, Rating, .drop=FALSE) %>%
  summarize(Count = dplyr::n(), .groups = "drop_last") %>%
  mutate(HumanOrLM = "LLM (sampled)") ->
  total_sampled_distribution

bind_rows(total_human_distribution, total_gen_distribution, total_sampled_distribution) ->
  total_distributions

bind_rows(isa_data_12_combined %>%
            mutate(Method = "Human (no context)"),
          llama3i_gen_context_responses_combined %>%
            mutate(Method = "LLM (generated)"),
          llama3i_lscale_5shot_sampled_responses %>%
            mutate(Method = "LLM (sampled)")
          ) -> human_and_lm_12_ratings

## Plot ----

### Histograms ----

both_divergences %>%
  ggplot(aes(x=JSDivergence)) +
  geom_histogram(bins = 20) +
  facet_wrap(~ Method) +
  ggtitle('Histogram of Jensen-Shannon divergence (per bigram) between human and LM distributions')

both_divergences %>%
  ggplot(aes(x=CoarseFrequency, y=JSDivergence)) +
  geom_violin() +
  facet_wrap(~ Method * AdjectiveClass)

both_divergences %>%
  ggplot(aes(x=CoarseFrequency, y=JSDivergence, fill=Method)) +
  geom_split_violin() +
  facet_wrap(~ AdjectiveClass) + 
  guides(x = guide_axis(angle = 90)) +
  theme_minimal()

### Split violins ----

both_divergences %>%
  mutate(Method = fct_recode(Method,
                             "Context" = "Context generation",
                             "Log-prob" = "Log-probability")) %>%
  ggplot(aes(x=AdjectiveClass, y=JSDivergence, fill=Method)) +
  geom_split_violin() +
  theme_minimal() +
  labs(x = "Adjective Type",
       y = "JS Divergence") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        #legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')
        ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5)) +
  scale_fill_manual(values = names_to_colors(c("Human", "Context", "Log-prob"))) +
  scale_y_reverse() ->
  js_splitviolin_plot
js_splitviolin_plot
ggsave('plots/jsdivergence_2methods_by_adjective_class.png', width=4, height=1.75, units='in')

js_splitviolin_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  )
ggsave('plots/jsdivergence_2methods_by_adjective_class_diss.png', width=4, height=1.75, units='in')


both_divergences %>%
  ggplot(aes(x=Adjective, y=JSDivergence, fill=Method)) +
  geom_split_violin() +
  theme_minimal() +
  labs(x = "Adjective",
       y = "Jensen-Shannon Divergence") + 
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5)) +
  scale_fill_discrete(type = c("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#81ba31", "#5a18dd"))

both_divergences %>%
  filter(AdjectiveClass == "Privative") %>%
  ggplot(aes(x=CoarseFrequency, y=JSDivergence, fill=Method)) +
  geom_split_violin() +
  facet_wrap(~ Adjective) + 
  guides(x = guide_axis(angle = 90))

gen_divergences %>%
  filter(AdjectiveClass == "Privative") %>%
  arrange(JSDivergence) %>%
  head(20)

sampled_divergences %>%
  filter(AdjectiveClass == "Privative") %>%
  arrange(JSDivergence) %>%
  head(20)

### Distributions (violin) ----

total_distributions %>%
  ggplot(aes(x=Rating, y=Count)) +
  geom_col(position="dodge") +
  facet_wrap(~ AdjectiveClass * HumanOrLM) +
  ggtitle("Rating distributions by adjective class")

human_and_lm_12_ratings %>%
  ggplot(aes(x=AdjectiveClass, y=NumRating, fill=Method)) +
  geom_violin(adjust=2, position="dodge") +
  labs(x = "Adjective Type",
       y = "Rating") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5))
ggsave('plots/rating_distribution_by_sampling_method.png', width=4, height=3.25, units='in')

human_and_lm_12_ratings %>%
#  filter(AdjectiveClass == "Privative") %>%
  ggplot(aes(x=Adjective, y=NumRating, fill=Method)) +
  geom_violin(adjust=2, position="dodge") +
  labs(x = "Adjective",
       y = "Rating") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  facet_wrap(~ AdjectiveClass, nrow=2, scale="free_x")  + 
  theme(
    strip.text.x = element_blank()
  ) +
  scale_fill_discrete(type = c("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#81ba31", "#5a18dd", "#BD45F1")) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))
ggsave('plots/all_a_rating_distribution_by_sampling_method.png', width=8, height=4, units='in')


### Distributions (bar) ----


llama3i_gen_dist <- build_human_dist(llama3i_gen_context_responses_combined)

split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))
ggsave('plots/rating_distribution_by_method_by_adj_class_splitbar_llama3i.png', width=4.25, height=3.25, units='in')

split_bar_plot(list(human_dist, llama3i_gen_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation")),
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))
ggsave('plots/rating_distribution_context_only_by_adj_class_splitbar_llama3i.png', width=3.25, height=3.25, units='in')

split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Adjective",
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))
ggsave('plots/rating_distribution_by_method_by_adj_splitbar_llama3i.png', width=9.25, height=4, units='in')

spb_bigrams = c("fake dollar", "illegal currency",
                "fake crowd", "fake concert", "fake idea",
                 "useful reef", "useful watch", "homemade cat",
                "former job", "counterfeit watch")
split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=spb_bigrams) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))
ggsave('plots/rating_distribution_by_method_by_bigram_splitbar_llama3i.png', width=10.75, height=4, units='in')

both_divergences %>%
  select(Bigram, JSDivergence, Adjective, Noun, AdjectiveClass, Method) %>%
  pivot_wider(names_from = Method, values_from = JSDivergence, names_glue = "{Method} JS") %>%
  rename(ContextJS = "Context generation JS", LogprobJS = "Log-probability sampling JS") %>%
  filter(!is.na(ContextJS)) %>%
  mutate(Match = case_when((ContextJS > 0.5  & LogprobJS > 0.5) ~ "Both bad",
                           (ContextJS > 0.5  & LogprobJS < 0.25) ~ "Logprob better",
                           (ContextJS < 0.25 & LogprobJS > 0.5) ~ "Context better",
                           (ContextJS < 0.25 & LogprobJS < 0.25) ~ "Both good",
                           .default = "Murky"
                           ),
         Delta = ContextJS - LogprobJS,
  ) ->
  both_divergences_classified

table(both_divergences_classified[, c("Match", "AdjectiveClass")])

table((both_divergences_classified %>% filter(str_length(Bigram) <= 15))[, c("Match", "AdjectiveClass")])

both_divergences_classified %>%
  ggplot(aes(x=Adjective, y=Delta)) +
  geom_violin(fill="#2C365E") +
  theme_minimal() +
  labs(x = "Adjective",
       y = "Delta between context-generation and log-probability JS divergence")
#  theme(legend.position = "bottom",
#        legend.key.size = unit(0.4, 'cm'),
#        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')) +
#  guides(fill = guide_legend(title.position="top", title.hjust = 0.5)) +
#  scale_fill_discrete(type = c("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#81ba31", "#5a18dd"))

set.seed(123)
both_divergences_classified %>%
  filter(Match != "Murky") %>%
  select(Bigram, Match, AdjectiveClass) %>%
  filter(str_length(Bigram) <= 15) %>%
  group_by(AdjectiveClass, Match) %>%
  sample_n(2) ->
  spb_bigrams_by_match
spb_bigrams_by_match %>% print(n=Inf)

spb_bigrams_by_match %>%
  pull(Bigram) ->
  spb_random_bigrams

split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=spb_random_bigrams,
               sorted_bigrams = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))
ggsave('plots/rating_distribution_by_method_by_random_bigram_splitbar_llama3i.png', width=15.25, height=5, units='in')

spb_bigrams2 = c("counterfeit dollar", "counterfeit watch", 
                 "fake lifestyle", "useful heart",
                 "homemade cat", "fake crowd",
                 "false market", "homemade bus"
                 )
split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=spb_bigrams2,
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')
        # text=element_text(size=15)
        ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE)) 
ggsave('plots/rating_distribution_by_method_by_bigram2_splitbar_llama3i.png', width=8.75, height=3.5, units='in')

split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=spb_bigrams2,
               vertical = TRUE,
               bigram_wrap_length = 11) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')
        # text=element_text(size=15)
  ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE)) +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=9.5),
        axis.text = element_text(size=10)
  ) 
ggsave('plots/rating_distribution_by_method_by_bigram2_splitbar_llama3i_diss.png', width=6.5, height=3.5, units='in')


split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=spb_bigrams2,
               vertical = TRUE,
               sorted_bigrams = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')
        # text=element_text(size=15)
  ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE)) +
  theme(text=element_text(size=18, color="#2C365E"))
ggsave('plots/rating_distribution_by_method_by_bigram2_sorted_splitbar_llama3i.png', width=14, height=5.5, units='in')

spb_bigrams_ctxt = c("counterfeit dollar", "counterfeit watch", "fake crowd", "homemade cat",
                     "false market", "fake concert", "homemade bus", "useful heart"  )
split_bar_plot(list(human_dist, llama3i_gen_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation")),
               human_name = "Human (context not given)",
               facet_by = "Bigram",
               bigrams=spb_bigrams_ctxt,
               sorted_bigrams = TRUE,
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')
        # text=element_text(size=15)
  ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE)) +
  theme(text=element_text(size=18, color="#2C365E"))
ggsave('plots/rating_distribution_context_only_by_bigram2_splitbar_llama3i.png', width=14, height=3.5, units='in')

spb_bigrams3 = c("homemade cat", "homemade bus", "homemade flower", "illegal currency",
                 "former currency")
split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=spb_bigrams3) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')
        # text=element_text(size=15)
  ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))

split_bar_plot(list(human_dist_extra, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=c("artificial intelligence")) +
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')
        # text=element_text(size=15)
  ) 
#  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=1, byrow=TRUE))
ggsave('plots/rating_distribution_by_method_artificial_intelligence_splitbar_llama3i.png', width=4, height=2.25, units='in')

spb_bigrams4 = c("counterfeit watch", "homemade bus")
split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=spb_bigrams4,
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt'),
        legend.justification = "center"
        # text=element_text(size=15)
  ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=2, byrow=TRUE))
ggsave('plots/rating_distribution_by_method_by_bigram_two_splitbar_llama3i.png', width=3.5, height=3.75, units='in')

split_bar_plot(list(human_dist, llama3i_gen_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation", 
                                 "Log-probability")),
               legend_title = "Method",
               facet_by = "Bigram",
               bigrams=c("fake reef"),
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt'),
        legend.justification = "center"
        # text=element_text(size=15)
  ) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=2, byrow=TRUE))
ggsave('plots/rating_distribution_by_method_fake_reef_two_splitbar_llama3i.png', width=2, height=3.75, units='in')

### JS divergence compared to logprob scaling ----

summarize_js_divergences(gen_divergences %>%
                           mutate(Method = "Llama 3 Instruct",
                                  Parameters = 70)) %>%
  mutate(Experiment = "Context generation") -> 
  gen_divergence_means


logprob_js_divergence_means %>%
  filter(Parameters != 0) %>%
  mutate(Experiment = "Log-probability") %>%
  ggplot(aes(x=Parameters, y=Mean, color=Method)) +
  geom_hline(aes(yintercept = 0, color="Human")) +
  geom_hline(data = logprob_js_divergence_means %>%
               filter(Method == "Uniform Baseline"), aes(yintercept=Mean, color=Method)) +
  geom_hline(data = logprob_js_divergence_means %>%
               filter(Method == "\"Majority\" Baseline"), aes(yintercept=Mean, color=Method)) +
  facet_wrap(~ AdjectiveClass) +
  geom_point(data = gen_divergence_means, aes(shape=Experiment), size=5) + 
  geom_point(alpha=0.4, aes(shape=Experiment)) +
  geom_line(alpha=0.4) +
  labs(x = "Model Parameters (B)", y = "JS Divergence", 
       color="Rating Source", shape="Method") +
  theme_minimal() +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_reverse() +
  expand_limits(y = c(0, 1)) +
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "\"Majority\" Baseline", "Uniform Baseline"))) +
  scale_shape_manual(values = c(18, 16)) +
  guides(color = guide_legend(override.aes = list(size = 2))) +
  theme(legend.title = element_text(margin = margin(t = -10, b=5)),
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'))
ggsave('plots/logprob_context_jsdivergence_scaling.png', width=6, height=2.5, units='in')


## Statistics ----

### Tables ----

both_divergences %>%
  select(Bigram, JSDivergence, Adjective, Noun, AdjectiveClass, Method) %>%
  pivot_wider(names_from = Method, values_from = JSDivergence, names_glue = "{Method} JS") %>%
  rename(ContextJS = "Context generation JS", LogprobJS = "Log-probability JS") %>%
  filter(!is.na(ContextJS)) %>%
  mutate(Match = case_when((ContextJS < 0.25 & LogprobJS < 0.25) ~ "Both good",
                           (ContextJS > 0.5  & LogprobJS > 0.5) ~ "Both bad",
                           (LogprobJS <= ContextJS) ~ "Logprob better",
                           (ContextJS < LogprobJS) ~ "Context better",
                           .default = "Murky"
  ),
  Delta = ContextJS - LogprobJS,
  ) ->
  both_divergences_classified2

table(both_divergences_classified2[, c("Match", "AdjectiveClass")])

table(both_divergences_classified2[, c("Match", "Adjective")])

both_divergences_classified2 %>%
  ggplot(aes(x=Adjective, fill=Match)) +
  geom_bar(position="stack")

both_divergences_classified2 %>%
  merge(isa_variance_12_combined %>% select(Bigram, Mean, SD), .by="Bigram") %>%
  filter(Match == "Both bad") %>%
  View()

both_divergences_classified2 %>%
  merge(isa_variance_12_combined %>% select(Bigram, Mean, SD), .by="Bigram") %>%
  filter(Match == "Context better") %>%
  View()

both_divergences_classified2 %>% filter(Bigram %in% spb_bigrams2)

both_divergences %>%
  group_by(Method) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

both_divergences %>%
  group_by(AdjectiveClass, Method) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

both_divergences_classified %>%
  merge(isa_variance_combined %>% select(Bigram, Mean, SD, Variance), by = "Bigram") %>% 
  rename(HumanMean = Mean, HumanSD = SD, HumanVariance = Variance) %>%
  add_frequency() %>%
  mutate(CoarseFrequency = fct_relevel(CoarseFrequency, rev)) ->
  both_divergences_why_data

both_divergences %>%
  merge(isa_variance_combined %>% select(Bigram, Mean, SD, Variance), by = "Bigram") %>% 
  rename(HumanMean = Mean, HumanSD = SD, HumanVariance = Variance) %>%
  filter(AdjectiveClass == "Subsective") %>%
  mutate(MeanClass = case_when(HumanMean >= 4 ~ "4+",
                               3.5 < HumanMean & HumanMean < 4 ~ "3.5-4",
                               .default="<3.5")) %>%
  group_by(MeanClass, Method) %>%
  summarize(MeanJS=mean(JSDivergence))

### Fit regressions on JS divergences ----

divergence_lm <- lmer(JSDivergence ~ Method + (1 | Adjective) + (1 | Noun) + (1 | CoarseFrequency), 
                      data = both_divergences %>%
                        mutate(Method = fct_relevel(Method, "Log-probability sampling")))
summary(divergence_lm)

why_delta_divergence_lm <- lm(Delta ~ AdjectiveClass * HumanMean + HumanSD + CoarseFrequency,
                          data = both_divergences_why_data)
summary(why_delta_divergence_lm)
plot(allEffects(why_delta_divergence_lm))
plot(predict_response(why_delta_divergence_lm, terms=c("HumanMean", "AdjectiveClass")), 
     dot_size = 1, show_data = TRUE, facets = TRUE) +
  labs(title = NULL, y = "JS Divergence", x = "Human Mean") +
  theme_minimal()


why_delta_divergence_lm2 <- lm(Delta ~ Adjective * HumanMean + HumanSD  + CoarseFrequency,
                              data = both_divergences_why_data %>%
                                mutate(Adjective = fct_relevel(Adjective, "useful")))
summary(why_delta_divergence_lm2)
# This is really hard to interpret as a bigger delta doesn't mean that one
# method is better in some cases unless one method is better all the time
# A third linear regression fitting the delta between the JS divergences for the two methods to the same predictors shows a significant effect for privative adjective type and for human mean, as well as a significant interaction, but no significant effect for human SD or bigram frequency. 
# We see that the difference between the two methods is smaller for privative-type adjectives than for subsective ones on average, but increases as the human mean increases.
# % Delta = Context - Logprob
# % So positive slope in delta = ??

why_logprob_divergence_lm <- lm(LogprobJS ~ AdjectiveClass * HumanMean + HumanSD + CoarseFrequency,
                                data = both_divergences_why_data)
summary(why_logprob_divergence_lm)

(plot(predict_response(why_logprob_divergence_lm, terms=c("HumanMean", "AdjectiveClass")), 
      dot_size = 1, show_data = TRUE, facets = TRUE) +
    labs(title = NULL, y = "JS Divergence", x = "Human Mean") +
    theme_minimal()
  ) +
(plot(predict_response(why_logprob_divergence_lm, terms=c("HumanSD")), 
      dot_size = 1, show_data = TRUE, jitter=0.01) +
   labs(title = NULL, y = "JS Divergence", x = "Human SD") +
   theme_minimal()
  ) +
  plot_annotation(title="Predicted JS divergence (log-probability)") +
  plot_layout(widths = c(2, 1))
ggsave("plots/lm_logprobs_js_divergence_effects_plot.png", width=4, height=3, units="in")

(plot(predict_response(why_logprob_divergence_lm, terms=c("HumanMean", "AdjectiveClass")), 
      dot_size = 1, show_data = TRUE, facets = TRUE) +
    labs(title = NULL, y = "JS Divergence", x = "Human Mean") +
    theme_minimal() + 
    theme(text = element_text(size=10, family="Palatino Linotype"),
          strip.text.x = element_text(size=10),
          axis.text = element_text(size=10)
    ) +
    scale_color_manual(name="AdjectiveClass",
                       values=c('Subsective'=light_blue_color, 'Privative'=magenta_color)) +
    scale_fill_manual(name="AdjectiveClass",
                       values=c('Subsective'=light_blue_color, 'Privative'=magenta_color))
) +
  (plot(predict_response(why_logprob_divergence_lm, terms=c("HumanSD")), 
        dot_size = 1, show_data = TRUE, jitter=0.01) +
     labs(title = NULL, y = "JS Divergence", x = "Human SD") +
     theme_minimal() + 
     theme(text = element_text(size=10, family="Palatino Linotype"),
           strip.text.x = element_text(size=10),
           axis.text = element_text(size=10)
     )
  ) +
#  plot_annotation(title="Predicted JS divergence (log-probability)") +
  plot_layout(widths = c(2, 1))
ggsave("plots/lm_logprobs_js_divergence_effects_plot_diss.png", width=4.25, height=1.75, units="in")


why_logprob_divergence_lm2 <- lm(LogprobJS ~ Adjective * HumanMean + HumanSD + CoarseFrequency,
                                data = both_divergences_why_data)
summary(why_logprob_divergence_lm2)
plot(predict_response(why_logprob_divergence_lm2, terms=c("HumanMean", "Adjective")), show_data = TRUE, facets = TRUE)
  scale_fill_paletteer_d("rcartocolor::Prism")
plot(predict_response(why_logprob_divergence_lm2, terms=c("HumanSD")), show_data = TRUE, jitter=0.05)

why_logprob_divergence_lm3 <- lm(LogprobJS ~ AdjectiveClass * HumanMean + HumanVariance + CoarseFrequency,
                                data = both_divergences_why_data)
summary(why_logprob_divergence_lm3)
plot(predict_response(why_logprob_divergence_lm3, terms=c("HumanMean", "AdjectiveClass")), show_data = TRUE, facets = TRUE)
plot(predict_response(why_logprob_divergence_lm3, terms=c("HumanVariance")), show_data = TRUE, jitter=0.05)
# Variance is a slightly worse fit than SD


why_context_divergence_lm <- lm(ContextJS ~ AdjectiveClass * HumanMean + HumanSD + CoarseFrequency,
                                data = both_divergences_why_data)
summary(why_context_divergence_lm)

(plot(predict_response(why_context_divergence_lm, terms=c("HumanMean", "AdjectiveClass")), 
      dot_size = 1, show_data = TRUE, facets = TRUE) +
    labs(title = NULL, y = "JS Divergence", x = "Human Mean") +
    theme_minimal()
) +
  (plot(predict_response(why_context_divergence_lm, terms=c("HumanSD")), 
       dot_size = 1, show_data = TRUE, jitter=0.01) +
     labs(title = NULL, y = "JS Divergence", x = "Human SD") +
     theme_minimal()
  ) +
  plot_annotation(title="Predicted JS divergence (context generation)") +
  plot_layout(widths = c(2, 1))
ggsave("plots/lm_context_js_divergence_effects_plot.png", width=4, height=3, units="in")

(plot(predict_response(why_context_divergence_lm, terms=c("HumanMean", "AdjectiveClass")), 
      dot_size = 1, show_data = TRUE, facets = TRUE) +
    labs(title = NULL, y = "JS Divergence", x = "Human Mean") +
    theme_minimal() +
    theme(text = element_text(size=10, family="Palatino Linotype"),
          strip.text.x = element_text(size=10),
          axis.text = element_text(size=10)
    ) +
    scale_color_manual(name="AdjectiveClass",
                       values=c('Subsective'=light_blue_color, 'Privative'=magenta_color)) +
    scale_fill_manual(name="AdjectiveClass",
                      values=c('Subsective'=light_blue_color, 'Privative'=magenta_color))
) +
  (plot(predict_response(why_context_divergence_lm, terms=c("HumanSD")), 
        dot_size = 1, show_data = TRUE, jitter=0.01) +
     labs(title = NULL, y = "JS Divergence", x = "Human SD") +
     theme_minimal() +
     theme(text = element_text(size=10, family="Palatino Linotype"),
           strip.text.x = element_text(size=10),
           axis.text = element_text(size=10)
     )
  ) +
#  plot_annotation(title="Predicted JS divergence (context generation)") +
  plot_layout(widths = c(2, 1))
ggsave("plots/lm_context_js_divergence_effects_plot_diss.png", width=4.25, height=1.75, units="in")


why_context_divergence_lm2 <- lm(ContextJS ~ Adjective * HumanMean + HumanSD + CoarseFrequency,
                                 data = both_divergences_why_data)
summary(why_context_divergence_lm2)
plot(predict_response(why_context_divergence_lm2, terms=c("HumanMean", "Adjective")), show_data = TRUE, facets = TRUE) +
  scale_fill_paletteer_d("rcartocolor::Prism")
plot(predict_response(why_context_divergence_lm2, terms=c("HumanSD")), show_data = TRUE, jitter=0.05)

plot(predict_response(why_logprob_divergence_lm, terms=c("HumanMean", "AdjectiveClass")), show_data = TRUE, facets = TRUE) +
plot(predict_response(why_context_divergence_lm, terms=c("HumanMean", "AdjectiveClass")), show_data = TRUE, facets = TRUE)
  
why_context_divergence_lm3 <- lm(ContextJS ~ AdjectiveClass + HumanMean + HumanSD + CoarseFrequency
                                 + AdjectiveClass:HumanMean + AdjectiveClass:CoarseFrequency,
                                data = both_divergences_why_data)
summary(why_context_divergence_lm3)

# Statistics and counts ----

## Counts and plots of counts ----

llama3i_gen_context_responses_combined %>%
  filter(AdjectiveClass == "Privative") %>%
  mutate(SubsectiveInference = NumRating >= 3) %>%
  group_by(SubsectiveInference) %>%
  summarize(Count = n()) %>%
  mutate(Percent = prop.table(Count)) 

isa_data_12_combined %>%
  filter(AdjectiveClass == "Privative") %>%
  mutate(SubsectiveInference = NumRating >= 3) %>%
  group_by(SubsectiveInference) %>%
  summarize(Count = n()) %>%
  mutate(Percent = prop.table(Count)) 

bind_rows(
  llama3i_gen_context_responses_combined %>%
    filter(AdjectiveClass == "Privative") %>%
    filter(NumRating >= 3) %>%
    group_by(Adjective) %>%
    summarize(Count = n()) %>%
    mutate(Method="LLM (generated)"),
  isa_data_12_combined %>%
    filter(AdjectiveClass == "Privative") %>%
    filter(NumRating >= 3) %>%
    group_by(Adjective) %>%
    summarize(Count = n())  %>%
    mutate(Method="Human (no context)")
) %>%
  ggplot(aes(x=Adjective,y=Count,fill=Method)) +
  geom_col(position="dodge") +
  labs(y = "Count of subsective inferences for privative adjectives")
  

isa_data_12_combined %>%
  filter(AdjectiveClass == "Privative") %>%
  mutate(SubsectiveInference = NumRating >= 3) %>%
  group_by(Adjective, SubsectiveInference) %>%
  summarize(Count = n()) %>%
  mutate(Percent = prop.table(Count)) 

## Fit models ----

stacked_gen_isa_data = merge(
  isa_data_12_combined %>% 
    group_by(Bigram) %>% 
    mutate(ContextBias = row_number()) %>% 
    ungroup() %>%
    select(!c(SE, Variance, Experiment, UserId, Frequency, Count)),
  llama3i_gen_context_responses_combined %>%
    rename(GenRating = Rating,
           GenNumRating = NumRating,
           GenMean = Mean,
           GenSD = SD) %>%
    select(!c(SE, Variance,  Frequency, Count)),
  by = c('Bigram', 'ContextBias', 'Adjective', 'Noun', 'CoarseFrequency', 'AdjectiveClass')
) %>%
  merge(llama3i_lscale_5shot_sampled_responses %>%
          group_by(Bigram) %>% 
          mutate(ContextBias = row_number()) %>% 
          ungroup() %>%
          rename(SampledRating = Rating,
                 SampledNumRating = NumRating,
                 SampledMean = Mean,
                 SampledSD = SD) %>%
          select(!c(SE, Variance,  Frequency, Count)),
        by = c('Bigram', 'ContextBias', 'Adjective', 'Noun', 'CoarseFrequency', 'AdjectiveClass')
        )

str(stacked_gen_isa_data)

# Deliberately fit rating as number so that it can get collapsed to mean per bigram
gen_mean_fit_lm = lmer(NumRating ~ GenNumRating + (1 | Bigram),
                       data = stacked_gen_isa_data)
summary(gen_mean_fit_lm)
r.squaredGLMM(gen_mean_fit_lm)

sampled_mean_fit_lm = lmer(NumRating ~ SampledNumRating  + (1 | Bigram),
                     data = stacked_gen_isa_data)
summary(sampled_mean_fit_lm)
r.squaredGLMM(sampled_mean_fit_lm)

model.null = clmm(Rating ~ 1 + (1 | Bigram), data = stacked_gen_isa_data)

gen_mean_fit_clm = clmm(Rating ~ GenRating  + (1 | Bigram),
                     data = stacked_gen_isa_data)
summary(gen_mean_fit_clm)
nagelkerke(fit = gen_mean_fit_clm, null = model.null)

sampled_mean_fit_clm = clm(Rating ~ SampledRating + (1 | Bigram),
                     data = stacked_gen_isa_data)
summary(sampled_mean_fit_clm)
nagelkerke(fit = sampled_mean_fit_clm, null = model.null)

# Look at contexts ----

read.csv("results/llm/gen_context/predictions_Meta-Llama-3-70B-Instruct_context-labelledscale-5shot.csv") %>%
  preprocess_labelled_responses() %>%
  select(Bigram, Adjective, Noun, Context, PredictedResponse, NumPredictedResponse) %>%
  add_frequency() ->
  llama3i_contexts

llama3i_contexts %>% 
  filter(Adjective == "fake") %>%
  slice_sample(n = 30) %>%
  View()

llama3i_contexts %>% 
  filter(Adjective == "fake" & NumPredictedResponse >= 4) %>%
  select(Bigram, Context) %>%
  View()


llama3i_contexts %>% 
  filter(Bigram == "fake crowd") %>%
  View()

llama3i_contexts %>% 
  group_by(Adjective) %>%
  slice_sample(n = 1) %>%
  View()

llama3i_contexts %>%
  filter(Bigram %in% c("useful heart", "fake lifestyle", "false market", "fake leg", "homemade cat")) %>%
  group_by(Bigram) %>%
  slice_sample(n = 6) %>%
  View()


llama3i_contexts %>%
  filter(Bigram %in% (gen_divergences %>% slice_max(JSDivergence, n=20) %>% pull(Bigram))) %>%
  group_by(Bigram) %>%
  slice_sample(n = 3) %>%
  select(Bigram, Context, PredictedResponse, AdjectiveClass) %>%
  View()


llama3i_contexts %>%
  filter(Noun %in% c("air") & AdjectiveClass == "Privative") %>%
  group_by(Bigram) %>%
  slice_sample(n = 6) %>%
  View()
