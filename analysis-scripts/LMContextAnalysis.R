library(tidyverse)
library(tidytext)
library(ordinal)
library(broom)
library(effects)
library(paletteer)

source("LM_Analysis_Utils.R")

# Load data ----

## Define locations ----

# Llama 2 Chat, 5-shot

llama2c_5shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-70b-chat-hf_context-labelledscale-5shot.csv")
llama2c_7b_5shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-7b-chat-hf_context-labelledscale-qa-5shot.csv")
llama2c_13b_5shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-13b-chat-hf_context-labelledscale-qa-5shot.csv")

# Llama 2 Chat, 0-shot

llama2c_0shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-70b-chat-hf_context-labelledscale-qa.csv")
llama2c_7b_0shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-7b-chat-hf_context-labelledscale-qa.csv")
llama2c_13b_0shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-13b-chat-hf_context-labelledscale-qa.csv")

# Llama 2, 5-shot

llama2_5shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-70b-hf_context-labelledscale-qa-5shot.csv")
llama2_7b_5shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-7b-hf_context-labelledscale-qa-5shot.csv")
llama2_13b_5shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-13b-hf_context-labelledscale-qa-5shot.csv")

# Llama 2, 0-shot

llama2_0shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-70b-hf_context-labelledscale-qa.csv")
llama2_7b_0shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-7b-hf_context-labelledscale-qa.csv")
llama2_13b_0shot_context_responses = read.csv("results/llm/context/predictions_Llama-2-13b-hf_context-labelledscale-qa.csv")

# Llama 3 Instruct, 5-shot

llama3i_5shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-70B-Instruct_context-labelledscale-5shot.csv")
llama3i_8b_5shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-8B-Instruct_context-labelledscale-5shot.csv")

# Llama 3 Instruct, 0-shot

# N.B. 0-shot response for 70B uses hand-written chat template
llama3i_0shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-70B-Instruct_context-labelledscale-llama3.csv")
llama3i_8b_0shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-8B-Instruct_context-labelledscale-qa.csv")

# Llama 3, 5-shot

llama3_5shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-70B_context-labelledscale-qa-5shot.csv")
llama3_8b_5shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-8B_context-labelledscale-qa-5shot.csv")

# Llama 3, 0-shot

llama3_0shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-70B_context-labelledscale-qa.csv")
llama3_8b_0shot_context_responses = read.csv("results/llm/context/predictions_Meta-Llama-3-8B_context-labelledscale-qa.csv")


# Mixtral and Qwen, 5-shot

mixtral_8x7b_5shot_context_responses = read.csv("results/llm/context/predictions_Mixtral-8x7B-Instruct-v0.1_context-labelledscale-5shot.csv")
qwen2i_5shot_context_responses = read.csv("results/llm/context/predictions_Qwen2-72B-Instruct_context-labelledscale-5shot.csv")

# Mixtral and Qwen, 0-shot

mixtral_8x7b_0shot_context_responses = read.csv("results/llm/context/predictions_Mixtral-8x7B-Instruct-v0.1_context-labelledscale.csv")
qwen2i_0shot_context_responses = read.csv("results/llm/context/predictions_Mixtral-8x7B-Instruct-v0.1_context-labelledscale.csv")

## Preprocessing ----

preprocess_context_responses <- function(responses) {
  responses %>%
    preprocess_labelled_responses() %>%
    mutate(ContextBias = factor(ContextBias)) %>%
    add_frequency -> p_responses
  return(p_responses)
}


# Llama 2 Chat, 5-shot

preprocess_context_responses(llama2c_5shot_context_responses) ->
  llama2c_5shot_context_responses

preprocess_context_responses(llama2c_7b_5shot_context_responses) ->
  llama2c_7b_5shot_context_responses

preprocess_context_responses(llama2c_13b_5shot_context_responses) ->
  llama2c_13b_5shot_context_responses

# Llama 2 Chat, 0-shot

preprocess_context_responses(llama2c_0shot_context_responses) ->
  llama2c_0shot_context_responses

preprocess_context_responses(llama2c_7b_0shot_context_responses) ->
  llama2c_7b_0shot_context_responses

preprocess_context_responses(llama2c_13b_0shot_context_responses) ->
  llama2c_13b_0shot_context_responses

# Llama 2, 5-shot

preprocess_context_responses(llama2_5shot_context_responses) ->
  llama2_5shot_context_responses

preprocess_context_responses(llama2_7b_5shot_context_responses) ->
  llama2_7b_5shot_context_responses

preprocess_context_responses(llama2_13b_5shot_context_responses) ->
  llama2_13b_5shot_context_responses

# Llama 2, 0-shot

preprocess_context_responses(llama2_0shot_context_responses) ->
  llama2_0shot_context_responses

preprocess_context_responses(llama2_7b_0shot_context_responses) ->
  llama2_7b_0shot_context_responses

preprocess_context_responses(llama2_13b_0shot_context_responses) ->
  llama2_13b_0shot_context_responses

# Llama 3 Instruct, 5-shot


preprocess_context_responses(llama3i_5shot_context_responses) ->
  llama3i_5shot_context_responses

preprocess_context_responses(llama3i_8b_5shot_context_responses) ->
  llama3i_8b_5shot_context_responses

# Llama 3 Instruct, 5-shot

preprocess_context_responses(llama3i_0shot_context_responses) ->
  llama3i_0shot_context_responses


preprocess_context_responses(llama3i_8b_0shot_context_responses) ->
  llama3i_8b_0shot_context_responses

# Llama 3, 5-shot

preprocess_context_responses(llama3_5shot_context_responses) ->
  llama3_5shot_context_responses

preprocess_context_responses(llama3_8b_5shot_context_responses) ->
  llama3_8b_5shot_context_responses

# Llama 3, 0-shot

preprocess_context_responses(llama3_0shot_context_responses) ->
  llama3_0shot_context_responses

preprocess_context_responses(llama3_8b_0shot_context_responses) ->
  llama3_8b_0shot_context_responses

# Mixtral & Qwen, 5-shot

preprocess_context_responses(mixtral_8x7b_5shot_context_responses) ->
  mixtral_8x7b_5shot_context_responses

preprocess_context_responses(qwen2i_5shot_context_responses) ->
  qwen2i_5shot_context_responses

# Mixtral & Qwen, 0-shot

preprocess_context_responses(mixtral_8x7b_0shot_context_responses) ->
  mixtral_8x7b_0shot_context_responses

preprocess_context_responses(qwen2i_0shot_context_responses) ->
  qwen2i_0shot_context_responses

# Add no-context data ----

# N.B. Llama 2 Chat 0-shot responses don't use chat template
llama2c_lscale_0shot_all_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/37778207/predictions.csv")

llama2c_lscale_5shot_all_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43509547/predictions.csv")
llama2c_7b_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43498921/predictions.csv")
llama2c_13b_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43498922/predictions.csv")

llama2_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43544291/predictions.csv")
llama2_7b_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43544294/predictions.csv")
llama2_13b_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43544293/predictions.csv")

llama3i_lscale_0shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/37357508/predictions_labelledscale.csv")

llama3i_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/41772823/predictions.csv")
llama3i_8b_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43498919/predictions.csv")

llama3_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43544292/predictions.csv")
llama3_8b_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43544295/predictions.csv")

mixtral_8x7b_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43413906/predictions.csv")
qwen2i_lscale_5shot_responses = read.csv("../Adjectives-PythonCode/output/lm_isa_data/43501963/predictions.csv")

llama2c_lscale_0shot_all_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2c_lscale_0shot_all_responses

llama3i_lscale_0shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3i_lscale_0shot_responses

llama2c_lscale_5shot_all_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2c_lscale_5shot_all_responses

llama2c_7b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2c_7b_lscale_5shot_responses

llama2c_13b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2c_13b_lscale_5shot_responses

llama2_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2_lscale_5shot_responses

llama2_7b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2_7b_lscale_5shot_responses

llama2_13b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2_13b_lscale_5shot_responses

llama3i_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3i_lscale_5shot_responses

llama3i_8b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3i_8b_lscale_5shot_responses

llama3_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3_lscale_5shot_responses

llama3_8b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3_8b_lscale_5shot_responses

mixtral_8x7b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  mixtral_8x7b_lscale_5shot_responses

qwen2i_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  qwen2i_lscale_5shot_responses

merge_no_context <- function(context_responses, plain_responses) {
  context_responses %>%
    select('Bigram', 'Adjective', 'Noun', 'ContextBias', 'PredictedResponse', 'NumPredictedResponse', 'CoarseFrequency') %>%
    bind_rows(plain_responses %>%
            mutate(ContextBias = "no context") %>%
            mutate(ContextBias = factor(ContextBias)) %>%
            select(c('Bigram', 'Adjective', 'Noun', 'ContextBias', 'PredictedResponse', 'NumPredictedResponse')) %>%
            filter(Bigram %in% context_responses$Bigram)
          ) %>%
    mutate(ContextBias = factor(ContextBias),
           Bigram = fct_drop(Bigram),
           Noun = fct_drop(Noun)) %>%
    mutate(ContextBias = fct_recode(ContextBias, "privative" = "Privative", 
                                             "subsective" = "Subsective")) ->
    merged_responses
  return(merged_responses)
}

llama2c_0shot_context_responses %>%
  merge_no_context(llama2c_lscale_0shot_all_responses) ->
  llama2c_0shot_combined_responses

llama3i_0shot_context_responses %>%
  merge_no_context(llama3i_lscale_0shot_responses) ->
  llama3i_0shot_combined_responses

llama2c_5shot_context_responses %>%
  merge_no_context(llama2c_lscale_5shot_all_responses) ->
  llama2c_5shot_combined_responses

llama2c_7b_5shot_context_responses %>%
  merge_no_context(llama2c_7b_lscale_5shot_responses) ->
  llama2c_7b_5shot_combined_responses

llama2c_13b_5shot_context_responses %>%
  merge_no_context(llama2c_13b_lscale_5shot_responses) ->
  llama2c_13b_5shot_combined_responses

llama2_5shot_context_responses %>%
  merge_no_context(llama2_lscale_5shot_responses) ->
  llama2_5shot_combined_responses

llama2_7b_5shot_context_responses %>%
  merge_no_context(llama2_7b_lscale_5shot_responses) ->
  llama2_7b_5shot_combined_responses

llama2_13b_5shot_context_responses %>%
  merge_no_context(llama2_13b_lscale_5shot_responses) ->
  llama2_13b_5shot_combined_responses

llama3i_5shot_context_responses %>%
  merge_no_context(llama3i_lscale_5shot_responses) ->
  llama3i_5shot_combined_responses

llama3i_8b_5shot_context_responses %>%
  merge_no_context(llama3i_8b_lscale_5shot_responses) ->
  llama3i_8b_5shot_combined_responses

llama3_5shot_context_responses %>%
  merge_no_context(llama3_lscale_5shot_responses) ->
  llama3_5shot_combined_responses

llama3_8b_5shot_context_responses %>%
  merge_no_context(llama3_8b_lscale_5shot_responses) ->
  llama3_8b_5shot_combined_responses

mixtral_8x7b_5shot_context_responses %>%
  merge_no_context(mixtral_lscale_5shot_responses) ->
  mixtral_8x7b_5shot_combined_responses

qwen2i_5shot_context_responses %>%
  merge_no_context(qwen2i_lscale_5shot_responses) ->
  qwen2i_5shot_combined_responses

# Explore no-context data ----

get_ps_percentages <- function(combined_responses) {
  combined_responses %>%
    filter(ContextBias == "no context") %>%
    mutate(PSRating = factor(case_when(NumPredictedResponse >= 4 ~ "Subsective",
                                       NumPredictedResponse <= 2 ~ "Privative",
                                       .default = "Unsure"))) %>%
    group_by(PSRating) %>%
    summarize(Count=n()) %>%
    mutate(Percent = prop.table(Count)) ->
    percentages
  return(percentages)
}

all_ps_percentages <- bind_rows(
  get_ps_percentages(llama2c_5shot_combined_responses) %>%
    mutate(Model = "Llama 2 70B Chat",
           Instruct = TRUE,
           Parameters = 70,
           Shots = 5),
  get_ps_percentages(llama2c_7b_5shot_combined_responses) %>%
    mutate(Model = "Llama 2 7B Chat",
           Instruct = TRUE,
           Parameters = 7,
           Shots = 5),
  get_ps_percentages(llama2c_13b_5shot_combined_responses) %>%
    mutate(Model = "Llama 2 13B Chat",
           Instruct = TRUE,
           Parameters = 13,
           Shots = 5),
  get_ps_percentages(llama3i_5shot_combined_responses) %>%
    mutate(Model = "Llama 3 70B Instruct",
           Instruct = TRUE,
           Parameters = 70,
           Shots = 5),
  get_ps_percentages(llama3i_8b_5shot_combined_responses) %>%
    mutate(Model = "Llama 3 8B Instruct",
           Instruct = TRUE,
           Parameters = 8,
           Shots = 5),
  get_ps_percentages(llama3_5shot_combined_responses) %>%
    mutate(Model = "Llama 3 70B",
           Instruct = FALSE,
           Parameters = 70,
           Shots = 5),
  get_ps_percentages(llama3_8b_5shot_combined_responses) %>%
    mutate(Model = "Llama 3 8B",
           Instruct = FALSE,
           Parameters = 8,
           Shots = 5),
  get_ps_percentages(llama2_5shot_combined_responses) %>%
    mutate(Model = "Llama 2 70B",
           Instruct = FALSE,
           Parameters = 70,
           Shots = 5),
  get_ps_percentages(llama2_7b_5shot_combined_responses) %>%
    mutate(Model = "Llama 2 7B",
           Instruct = FALSE,
           Parameters = 7,
           Shots = 5),
  get_ps_percentages(llama2_13b_5shot_combined_responses) %>%
    mutate(Model = "Llama 2 13B",
           Instruct = FALSE,
           Parameters = 13,
           Shots = 5),
  get_ps_percentages(mixtral_8x7b_5shot_combined_responses) %>%
    mutate(Model = "Mixtral 8x7B Instruct",
           Instruct = TRUE,
           Parameters = 56,
           Shots = 5),
  get_ps_percentages(qwen2i_5shot_combined_responses) %>%
    mutate(Model = "Qwen 2 72B Instruct",
           Instruct = TRUE,
           Parameters = 72,
           Shots = 5),
  get_ps_percentages(an_context2_plus_target %>%
                       rename(NumPredictedResponse = NumRating)) %>%
    mutate(Model = "Human",
           Instruct = FALSE,
           Parameters = 0,
           Shots = 0)
) %>%
  mutate(Model = factor(Model)) %>%
  mutate(Model = fct_relevel(Model, c("Human", 
                                      "Qwen 2 72B Instruct",
                                      "Llama 3 70B Instruct",
                                      "Llama 2 70B Chat",
                                      "Mixtral 8x7B Instruct",
                                      "Llama 3 8B Instruct",
                                      "Llama 2 13B Chat",
                                      "Llama 2 7B Chat",
                                      "Llama 3 70B",
                                      "Llama 2 70B",
                                      "Llama 3 8B",
                                      "Llama 2 13B",
                                      "Llama 2 7B"
  ))) %>%
  mutate(PSRating = factor(PSRating, levels=c("Subsective", "Unsure", "Privative")))
all_ps_percentages

all_ps_percentages %>%
  filter(Instruct==FALSE)

# Plot data ----

## No-context percentages ----

all_ps_percentages %>%
  filter(Instruct==TRUE | Model == "Human") %>%
  mutate(Model = fct_recode(Model,
                            "Qwen 2 72B" = "Qwen 2 72B Instruct",
                            "Llama 3 70B" = "Llama 3 70B Instruct",
                            "Llama 2 70B" = "Llama 2 70B Chat",
                            "Mixtral 8x7B" = "Mixtral 8x7B Instruct",
                            "Llama 3 8B" = "Llama 3 8B Instruct",
                            "Llama 2 13B" = "Llama 2 13B Chat",
                            "Llama 2 7B" = "Llama 2 7B Chat")) %>%
  ggplot(aes(x=Model,y=Percent,fill=PSRating)) +
  geom_col(position="stack") +
  labs(fill="Rating category", y="Rating percentage", x="Rating source") +
  scale_fill_discrete(type=c(light_blue_color, dark_yellow_color, magenta_color)) +
  guides(x = guide_axis(angle = 45)) +
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')
        ) ->
  lm_isa_rating_plot
lm_isa_rating_plot
ggsave("plots/lm_isa_rating_percentages.png", width=4, height=2.25, units="in")

lm_isa_rating_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  )
ggsave("plots/lm_isa_rating_percentages_diss.png", width=4, height=2.25, units="in")

all_ps_percentages %>%
  filter(Instruct==FALSE | Model == "Human") %>%
  ggplot(aes(x=Model,y=Percent,fill=PSRating)) +
  geom_col(position="stack") +
  labs(fill="Rating category", y="Rating percentage", x="Rating source") +
  scale_fill_discrete(type=c(light_blue_color, dark_yellow_color, magenta_color)) +
  guides(x = guide_axis(angle = 45)) +
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        #        legend.box.spacing = unit(0, 'pt')
  )
ggsave("plots/lm_isa_rating_percentages_base.png", width=4, height=2.25, units="in")

## 1 SD plots ----

all_adjectives = context_variance_combined %>% distinct(Adjective) %>% pull(Adjective)
all_nouns = context_variance_combined %>% distinct(Noun) %>% pull(Noun)

nouns = c("dollar", "concert", "fire", "abundance", "air", "door",
          "gun", "crowd", "reef", "photograph","information", "fact",
          "jacket", "report", "laugh", "bed", "lion", "watch", 
          "business", "art", "celebration",
          "truck", "chair", "handbag", "scarcity", "scarf", "fruit",
          "rumor", "painting", "flower", "hand")

# c("fake", "counterfeit")
plot_1sd(context_variance_combined, 
         llama2c_0shot_combined_responses, lm_name="Llama 2 Chat (0-shot)",
         adjectives=all_adjectives, nouns=all_nouns, context=TRUE)

plot_1sd(context_variance_combined, 
         llama3i_0shot_combined_responses, lm_name="Llama 3 Instruct (0-shot)",
         adjectives=all_adjectives, nouns=all_nouns, context=TRUE)

plot_1sd(context_variance_combined, 
         llama2c_5shot_combined_responses, lm_name="Llama 2 Chat (5-shot)",
         adjectives=all_adjectives, nouns=all_nouns, context=TRUE)

plot_1sd(context_variance_combined, 
         llama3i_5shot_combined_responses, lm_name="Llama 3 Instruct (5-shot)",
         adjectives=all_adjectives, nouns=all_nouns, context=TRUE)

plot_1sd(context_variance_combined %>% filter(ContextBias != "no context"), 
         llama3i_5shot_combined_responses %>% filter(ContextBias != "no context"), 
         lm_name="Llama 3 Instruct",
         adjectives=c("fake"), nouns=all_nouns, context=TRUE, thresholds=TRUE, poster=TRUE)
ggsave("plots/lm_isa_context_fake_1sd.png", width=6, height=4, units="in")
ggsave("plots/lm_isa_context_fake_1sd_poster.png", width=14, height=9, units="in")


## Effect of context plots ----

plot_context_effect_by_adjective <- function(lm_responses, lm_name) {
  lm_responses %>%
    ggplot(aes(x=ContextBias, y=NumPredictedResponse, col=ContextBias)) +
    geom_boxplot() +
    labs(col='Context bias', x='Context bias', y='Rating') + 
    scale_color_manual(name="Context bias",
                       values=c('subsective'='#F8766D', 'privative'='#00BFC4')) +
    facet_wrap(~ Adjective) +
    ggtitle(sprintf("%s ratings for 'In this setting, is an AN still an N?'", lm_name))
}

plot_context_effect_by_adjective(llama2c_0shot_combined_responses, 
                                 lm_name='Llama 2 Chat (0 shot)')

plot_context_effect_by_adjective(llama2c_5shot_combined_responses, 
                                 lm_name='Llama 2 Chat 70B (5 shot)')

plot_context_effect_by_adjective(llama2c_13b_5shot_context_responses, 
                                 lm_name='Llama 2 Chat 13B (5 shot)')

plot_context_effect_by_adjective(llama3i_0shot_combined_responses, 
                                 lm_name='Llama 3 Instruct (0 shot)')

plot_context_effect_by_adjective(llama3i_5shot_combined_responses, 
                                 lm_name='Llama 3 Instruct (5 shot)')

# Effect of context models ----

llama2c_0shot_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                                   data = llama2c_0shot_combined_responses %>%  
                                     mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                   link = "logit")

summary(llama2c_0shot_bias_works_lm)

llama2c_5shot_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                                   data = llama2c_5shot_combined_responses %>%  
                                     mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                   link = "logit")

summary(llama2c_5shot_bias_works_lm)

llama2c_7b_5shot_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                                       data = llama2c_7b_5shot_combined_responses %>%  
                                         mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                       link = "logit")

summary(llama2c_7b_5shot_bias_works_lm)

llama2c_13b_5shot_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                                       data = llama2c_13b_5shot_combined_responses %>%  
                                         mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                       link = "logit")

summary(llama2c_13b_5shot_bias_works_lm)

llama3i_0shot_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                                    data = llama3i_0shot_combined_responses %>%  
                                      mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                    link = "logit")

summary(llama3i_0shot_bias_works_lm)

llama3i_5shot_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + CoarseFrequency + (1 | Adjective) + (1 | Noun),  
                                      data = llama3i_5shot_combined_responses %>%  
                                        mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                      link = "logit")

summary(llama3i_5shot_bias_works_lm)

llama3i_8b_5shot_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                                    data = llama3i_8b_5shot_combined_responses %>%  
                                      mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                    link = "logit")

summary(llama3i_8b_5shot_bias_works_lm)

mixtral_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                                    data = mixtral_8x7b_5shot_combined_responses %>%  
                                      mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                                    link = "logit")

summary(mixtral_bias_works_lm)

qwen_bias_works_lm <- clmm(PredictedResponse ~ ContextBias + (1 | Adjective) + (1 | Noun),  
                              data = qwen2i_5shot_combined_responses %>%  
                                mutate(ContextBias = fct_relevel(ContextBias, 'no context')), 
                              link = "logit")

summary(qwen_bias_works_lm)

# Accuracy within 1 SD ----

## Calculate ----

# Llama 2 Chat, 5-shot

llama2c_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2c_5shot_context_1sd_acc
llama2c_5shot_context_1sd_acc %>% print(n = Inf)

llama2c_7b_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2c_7b_5shot_context_1sd_acc
llama2c_7b_5shot_context_1sd_acc %>% print(n = Inf)

llama2c_13b_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2c_13b_5shot_context_1sd_acc
llama2c_13b_5shot_context_1sd_acc %>% print(n = Inf)

# Llama 2 Chat, 0-shot

llama2c_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2c_0shot_context_1sd_acc
llama2c_0shot_context_1sd_acc %>% print_1sd_paper_results()

llama2c_7b_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2c_7b_0shot_context_1sd_acc
llama2c_7b_0shot_context_1sd_acc %>% print_1sd_paper_results()

llama2c_13b_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2c_13b_0shot_context_1sd_acc
llama2c_13b_0shot_context_1sd_acc %>% print_1sd_paper_results()

# Llama 2, 5-shot

llama2_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2_5shot_context_1sd_acc
llama2_5shot_context_1sd_acc %>% print(n = Inf)

llama2_7b_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2_7b_5shot_context_1sd_acc
llama2_7b_5shot_context_1sd_acc %>% print(n = Inf)

llama2_13b_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2_13b_5shot_context_1sd_acc
llama2_13b_5shot_context_1sd_acc %>% print(n = Inf)

# Llama 2, 0-shot

llama2_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2_0shot_context_1sd_acc
llama2_0shot_context_1sd_acc %>% print_1sd_paper_results()

llama2_7b_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2_7b_0shot_context_1sd_acc
llama2_7b_0shot_context_1sd_acc %>% print_1sd_paper_results()

llama2_13b_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama2_13b_0shot_context_1sd_acc
llama2_13b_0shot_context_1sd_acc %>% print_1sd_paper_results()


# Llama 3 Instruct, 5-shot

llama3i_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3i_5shot_context_1sd_acc
llama3i_5shot_context_1sd_acc %>% print(n = Inf)

llama3i_8b_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3i_8b_5shot_context_1sd_acc
llama3i_8b_5shot_context_1sd_acc %>% print(n = Inf)

# Llama 3 Instruct, 0-shot

llama3i_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3i_0shot_context_1sd_acc
llama3i_0shot_context_1sd_acc %>% print_1sd_paper_results()

llama3i_8b_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3i_8b_0shot_context_1sd_acc
llama3i_8b_0shot_context_1sd_acc %>% print_1sd_paper_results()

# Llama 3, 5-shot

llama3_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3_5shot_context_1sd_acc
llama3_5shot_context_1sd_acc %>% print(n = Inf)

llama3_8b_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3_8b_5shot_context_1sd_acc
llama3_8b_5shot_context_1sd_acc %>% print(n = Inf)

# Llama 3, 0-shot

llama3_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3_0shot_context_1sd_acc
llama3_0shot_context_1sd_acc %>% print_1sd_paper_results()

llama3_8b_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  llama3_8b_0shot_context_1sd_acc
llama3_8b_0shot_context_1sd_acc %>% print_1sd_paper_results()

# Mixtral & Qwen, 5-shot

mixtral_8x7b_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  mixtral_8x7b_5shot_context_1sd_acc
mixtral_8x7b_5shot_context_1sd_acc %>% print(n = Inf)

qwen2i_5shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  qwen2i_5shot_context_1sd_acc
qwen2i_5shot_context_1sd_acc %>% print(n = Inf)

# Mixtral & Qwen, 0-shot

mixtral_8x7b_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  mixtral_8x7b_0shot_context_1sd_acc
mixtral_8x7b_0shot_context_1sd_acc %>% print_1sd_paper_results()

qwen2i_0shot_context_responses %>%
  accuracy_1sd_context(context_variance_combined) ->
  qwen2i_0shot_context_1sd_acc
qwen2i_0shot_context_1sd_acc %>% print_1sd_paper_results()

# Humans and baselines

an_context_combined_plus %>% 
  rename(NumIsaRating = NumRating) %>%
  accuracy_1sd_context(context_variance_combined, lm_responses=FALSE) ->
  human_context_1sd_acc
human_context_1sd_acc %>% print(n = Inf)

calculate_random_baseline_context(context_variance_combined) ->
  random_context_1sd_acc

print_1sd_paper_results <- function(sd_acc) {
  sd_acc %>%
  filter(AdjectiveClass %in% c("Overall", "High frequency", "Zero frequency")) %>% 
    filter((AdjectiveClass == "Overall" | ContextBias == "both"))
}

print_1sd_paper_results(random_context_1sd_acc)

all_context_1sd_accuracies <- bind_rows(llama2c_0shot_context_1sd_acc %>%
                              mutate(Model = "Llama 2 Chat",
                                     Parameters = 70,
                                     Shots = 0),
                            llama2c_5shot_context_1sd_acc %>%
                              mutate(Model = "Llama 2 Chat",
                                     Parameters = 70,
                                     Shots = 5),
                            llama2c_7b_5shot_context_1sd_acc %>%
                              mutate(Model = "Llama 2 Chat",
                                     Parameters = 7,
                                     Shots = 5),
                            llama2c_13b_5shot_context_1sd_acc %>%
                              mutate(Model = "Llama 2 Chat",
                                     Parameters = 13,
                                     Shots = 5),
                            llama3i_0shot_context_1sd_acc %>%
                              mutate(Model = "Llama 3 Instruct",
                                     Parameters = 70,
                                     Shots = 0),
                            llama3i_5shot_context_1sd_acc %>%
                              mutate(Model = "Llama 3 Instruct",
                                     Parameters = 70,
                                     Shots = 5),
                            llama3i_8b_5shot_context_1sd_acc %>%
                              mutate(Model = "Llama 3 Instruct",
                                     Parameters = 8,
                                     Shots = 5),
                            mixtral_8x7b_5shot_context_1sd_acc %>%
                              mutate(Model = "Mixtral Instruct",
                                     Parameters = 56,
                                     Shots = 5),
                            qwen2i_5shot_context_1sd_acc %>%
                              mutate(Model = "Qwen 2 Instruct",
                                     Parameters = 72,
                                     Shots = 5),
                            human_context_1sd_acc %>%
                              mutate(Model = "Human",
                                     Parameters = 0,
                                     Shots = 0),
                            sentiment_context_1sd_acc %>%
                              mutate(Model = "Sentiment Baseline",
                                     Parameters = 70,
                                     Shots = 0),
                            random_context_1sd_acc %>%
                              mutate(Model = "Random Baseline",
                                     Parameters = 0,
                                     Shots = 0)
                            ) %>%
  mutate(Model = factor(Model))

all_context_base_1sd_accuracies <- bind_rows(llama2_5shot_context_1sd_acc %>%
                                          mutate(Model = "Llama 2",
                                                 Parameters = 70,
                                                 Shots = 5),
                                        llama2_7b_5shot_context_1sd_acc %>%
                                          mutate(Model = "Llama 2",
                                                 Parameters = 7,
                                                 Shots = 5),
                                        llama2_13b_5shot_context_1sd_acc %>%
                                          mutate(Model = "Llama 2",
                                                 Parameters = 13,
                                                 Shots = 5),
                                        llama3_5shot_context_1sd_acc %>%
                                          mutate(Model = "Llama 3",
                                                 Parameters = 70,
                                                 Shots = 5),
                                        llama3_8b_5shot_context_1sd_acc %>%
                                          mutate(Model = "Llama 3",
                                                 Parameters = 8,
                                                 Shots = 5)
) %>%
  mutate(Model = factor(Model))

human_total_1sd_acc <- all_context_1sd_accuracies %>%
  filter(AdjectiveClass == "Overall" & Model == "Human") %>%
  select(Model, ContextBias, Accuracy)

## Plot ----

all_context_1sd_accuracies %>%
  filter(Shots == 5 & AdjectiveClass == "Overall") %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=3) +
  geom_hline(data = human_total_1sd_acc, aes(yintercept=Accuracy, color=Model)) +
  facet_grid(~ ContextBias) +
  theme_minimal()

all_context_1sd_accuracies %>%
  filter(Shots == 5 & AdjectiveClass == "Overall" & ContextBias == "both") %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=3) +
  geom_hline(data = human_total_1sd_acc %>% filter(ContextBias == "both"), aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,10)) +
  theme_minimal() +
  labs(color='Rating Source', y='Accuracy within 1SD') + 
#  theme(legend.position = "bottom") +
  guides(color = guide_legend(title.position="top", title.hjust = 0.5, 
#                              nrow=2, byrow=TRUE
                              )) +
  scale_color_discrete(type = c("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#81ba31", "#5a18dd"))
ggsave('plots/context_1sd_accuracy_scaling.png', width=4, height=2, units='in')
  

# Filtered to bigrams where humans are accurate ----

context_variance_combined %>%
  filter(ContextBias != "no context") %>%
  filter(SD <= 1) %>%
  select(Bigram, ContextBias) ->
  high_quality_contexts

llama3i_5shot_context_responses %>%
  mutate(ContextBias = fct_recode(ContextBias, "privative" = "Privative", 
                                  "subsective" = "Subsective")) %>%
  merge(high_quality_contexts, by = c("Bigram", "ContextBias")) ->
  llama3i_5shot_hq_context_responses

llama3i_5shot_hq_context_responses %>% 
  group_by(ContextBias) %>%
  summarize(n())

an_context2_plus_target %>% 
  merge(high_quality_contexts, by = c("Bigram", "ContextBias")) ->
human_hq_context_responses 

# Accuracy by effect of context ----

## Humans ----

an_context_combined_plus_target %>%
  mutate(across(c("Bigram", "Adjective", "Noun", "ContextBias", "Rating", "UserId"), as.factor)) %>%
  add_frequency() %>%
  context_direct_accuracy() ->
  human_context_acc
human_context_acc

## Models ----

llama2c_0shot_context_responses %>%
  context_direct_accuracy() ->
  llama2c_0shot_context_acc
llama2c_0shot_context_acc

llama2c_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama2c_5shot_context_acc
llama2c_5shot_context_acc

llama2c_7b_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama2c_7b_5shot_context_acc
llama2c_7b_5shot_context_acc

llama2c_13b_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama2c_13b_5shot_context_acc
llama2c_13b_5shot_context_acc

llama2_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama2_5shot_context_acc
llama2_5shot_context_acc

llama2_7b_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama2_7b_5shot_context_acc
llama2_7b_5shot_context_acc

llama2_13b_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama2_13b_5shot_context_acc
llama2_13b_5shot_context_acc

llama3i_0shot_context_responses %>%
  context_direct_accuracy() ->
  llama3i_0shot_context_acc
llama3i_0shot_context_acc

llama3i_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama3i_5shot_context_acc
llama3i_5shot_context_acc

llama3i_8b_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama3i_8b_5shot_context_acc
llama3i_8b_5shot_context_acc

llama3_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama3_5shot_context_acc
llama3_5shot_context_acc

llama3_8b_5shot_context_responses %>%
  context_direct_accuracy() ->
  llama3_8b_5shot_context_acc
llama3_8b_5shot_context_acc

mixtral_8x7b_5shot_context_responses %>%
  context_direct_accuracy() ->
  mixtral_8x7b_5shot_context_acc
mixtral_8x7b_5shot_context_acc

qwen2i_5shot_context_responses %>%
  context_direct_accuracy() ->
  qwen2i_5shot_context_acc
qwen2i_5shot_context_acc

all_context_accuracies <- bind_rows(llama2c_5shot_context_acc %>%
                                      mutate(Model = "Llama 2 Chat",
                                             Parameters = 70,
                                             Shots = 5),
                                    llama2c_7b_5shot_context_acc %>%
                                      mutate(Model = "Llama 2 Chat",
                                             Parameters = 7,
                                             Shots = 5),
                                    llama2c_13b_5shot_context_acc %>%
                                      mutate(Model = "Llama 2 Chat",
                                             Parameters = 13,
                                             Shots = 5),
                                    llama3i_5shot_context_acc %>%
                                      mutate(Model = "Llama 3 Instruct",
                                             Parameters = 70,
                                             Shots = 5),
                                    llama3i_8b_5shot_context_acc %>%
                                      mutate(Model = "Llama 3 Instruct",
                                             Parameters = 8,
                                             Shots = 5),
                                    mixtral_8x7b_5shot_context_acc %>%
                                      mutate(Model = "Mixtral Instruct",
                                             Parameters = 56,
                                             Shots = 5),
                                    qwen2i_5shot_context_acc %>%
                                      mutate(Model = "Qwen 2 Instruct",
                                             Parameters = 72,
                                             Shots = 5),
                                    human_context_acc %>%
                                      mutate(Model = "Human",
                                             Parameters = 0,
                                             Shots = 0),
                                    sentiment_context_acc %>% 
                                      mutate(Model = "Sentiment Baseline",
                                             Parameters = 70,
                                             Shots = 0)
) %>%
  mutate(Model = factor(Model))

all_context_base_accuracies <- bind_rows(llama2_5shot_context_acc %>%
                                      mutate(Model = "Llama 2",
                                             Parameters = 70,
                                             Shots = 5),
                                    llama2_7b_5shot_context_acc %>%
                                      mutate(Model = "Llama 2",
                                             Parameters = 7,
                                             Shots = 5),
                                    llama2_13b_5shot_context_acc %>%
                                      mutate(Model = "Llama 2",
                                             Parameters = 13,
                                             Shots = 5),
                                    llama3_5shot_context_acc %>%
                                      mutate(Model = "Llama 3",
                                             Parameters = 70,
                                             Shots = 5),
                                    llama3_8b_5shot_context_acc %>%
                                      mutate(Model = "Llama 3",
                                             Parameters = 8,
                                             Shots = 5)
) %>%
  mutate(Model = factor(Model))

human_total_acc <- all_context_accuracies %>%
  filter(Class == "Overall" & Model == "Human") %>%
  select(Model, Class, Accuracy)

sentiment_total_acc <- all_context_accuracies %>%
  filter(Class == "Overall" & Model == "Sentiment Baseline") %>%
  select(Model, Class, Accuracy)

## Plot ----

all_context_accuracies %>%
  filter(Shots == 5 & Class %in% c("Overall", "Privative", "Subsective")) %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=3) +
  geom_hline(data = human_total_acc, aes(yintercept=Accuracy, color=Model)) +
  facet_grid(~ Class) +
  theme_minimal()

all_context_accuracies %>%
  filter(Shots == 5 & Class == "Overall") %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=3) +
  geom_hline(data = human_total_acc %>% filter(Class == "Overall"), aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = sentiment_total_acc %>% filter(Class == "Overall"), aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,10)) +
  theme_minimal() +
  labs(color='Rating Source') + 
  #  theme(legend.position = "bottom") +
  guides(color = guide_legend(title.position="top", title.hjust = 0.5, 
                              #                              nrow=2, byrow=TRUE
  ))
ggsave('plots/context_accuracy_scaling.png', width=4, height=2, units='in')

all_context_accuracies %>%
  filter(Shots == 5 & Class == "Overall") %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model,linetype='Accuracy')) +
  geom_line(aes(color=Model,linetype='Accuracy within 1 SD'), 
            data=all_context_1sd_accuracies %>%
              filter(Shots == 5 & AdjectiveClass == "Overall" & ContextBias == "both")) +
  geom_point(aes(color=Model), size=2) +
  geom_point(aes(color=Model), size=2, shape=1,
             data=all_context_1sd_accuracies %>%
               filter(Shots == 5 & AdjectiveClass == "Overall" & ContextBias == "both")) +
  geom_hline(data = human_total_acc %>% filter(Class == "Overall"), 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = sentiment_total_acc %>% filter(Class == "Overall"), 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = human_total_1sd_acc %>% filter(ContextBias == "both"),
             aes(yintercept=Accuracy, color=Model),
             linetype = "dashed") +
  scale_x_continuous(breaks=seq(0,75,10)) +
  scale_linetype(name='Metric',
                         breaks=c('Accuracy', 'Accuracy within 1 SD')) +
  theme_minimal() +
  labs(color='Rating Source') + 
  #  theme(legend.position = "bottom") +
  guides(color = guide_legend(title.position="top", title.hjust = 0.5, 
                              #                              nrow=2, byrow=TRUE
  ))
ggsave('plots/context_accuracy_scaling_2metric.png', width=4, height=2.5, units='in')

## Thresholded examples ----

llama3i_5shot_combined_responses %>% 
  filter(ContextBias != "no context") %>%
  mutate(ContextBias = fct_recode(ContextBias,
                                  "Privative context bias" = "privative", "Subsective context bias" = "subsective"
  )) %>%
  filter(Adjective %in% c("fake")) %>%
  ggplot(aes(x=Noun, 
             y=NumPredictedResponse,
             color=ContextBias)) +
  geom_hline(yintercept=2, color="#F8766D", linewidth=2, linetype=2) +
  geom_hline(yintercept=4, color="#00BFC4", linewidth=2, linetype=2) +
  geom_point(shape=18, size=8) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  labs(x="fake {noun}", y="Rating", color="Context bias", 
       title = "Ratings for Llama 3 70B Instruct") +
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ ContextBias) +
  theme_minimal() +
  theme(legend.position="none") +
  theme(text=element_text(size=36, color="#2C365E"))
ggsave("plots/lm_isa_context_fake_threshold_only.png", width=14, height = 9, units="in")

### Five-way plot for paper ----

prepare_1sd_accuracies <- function(sd_accuracies) {
  sd_accuracies %>%
    filter(AdjectiveClass %in% c("Overall" , "High frequency", "Zero frequency") & ContextBias == "both") %>%
    mutate(Class = AdjectiveClass) %>%
    bind_rows(
      sd_accuracies %>%
        filter(AdjectiveClass %in% c("Overall") & ContextBias != "both") %>%
        mutate(Class = fct_recode(ContextBias, "Privative context" = "privative", "Subsective context" = "subsective"))
    ) %>%
    mutate(Class=fct_relevel(Class, "Privative context", "Subsective context", "High frequency", "Zero frequency", "Overall")) ->       
    sd_accuracies_plot
  return(sd_accuracies_plot)
}

prepare_1sd_accuracies(all_context_1sd_accuracies %>% filter(Shots == 5)) ->       
  all_context_1sd_accuracies_plot

prepare_1sd_accuracies(all_context_1sd_accuracies %>% filter(Model == "Human")) ->       
  human_total_1sd_acc_by_class

prepare_1sd_accuracies(all_context_1sd_accuracies %>% filter(Model == "Sentiment Baseline")) ->       
  sentiment_1sd_acc_by_class

prepare_1sd_accuracies(all_context_1sd_accuracies %>% filter(Model == "Random Baseline")) ->       
  random_1sd_acc_by_class



all_context_accuracies %>%
  filter(Shots == 5 & Class %in% c("Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class=fct_relevel(Class, "Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class = fct_recode(Class, "Privative context" = "Privative", "Subsective context" = "Subsective")) ->       
  all_context_accuracies_plot

all_context_accuracies %>%
  filter(Model == "Human" & Class %in% c("Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class=fct_relevel(Class, "Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class = fct_recode(Class, "Privative context" = "Privative", "Subsective context" = "Subsective")) ->       
  human_total_acc_by_class

all_context_accuracies %>%
  filter(Model == "Sentiment Baseline" & Class %in% c("Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class=fct_relevel(Class, "Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class = fct_recode(Class, "Privative context" = "Privative", "Subsective context" = "Subsective")) ->       
  sentiment_acc_by_class

data.frame(Class = c("Privative", "Subsective", "High frequency", "Zero frequency", "Overall"),
           Accuracy = 0.4, Model = "Random Baseline") %>%
  mutate(Class=fct_relevel(Class, "Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class = fct_recode(Class, "Privative context" = "Privative", "Subsective context" = "Subsective")) ->
  random_acc_by_class

all_context_accuracies_plot %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model,linetype='Accuracy')) +
  geom_line(aes(color=Model,linetype='Accuracy within 1 SD'), 
            data=all_context_1sd_accuracies_plot) +
  geom_point(aes(color=Model), size=2) +
  geom_point(aes(color=Model), size=2, shape=1,
             data=all_context_1sd_accuracies_plot) +
  geom_hline(data = human_total_acc_by_class, 
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy")) +
  geom_hline(data = sentiment_acc_by_class, 
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy")) +
  geom_hline(data = human_total_1sd_acc_by_class,
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy within 1 SD")
             ) +
  geom_hline(data = sentiment_1sd_acc_by_class,
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy within 1 SD")
             ) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  scale_linetype(name='Metric',
                 breaks=c('Accuracy', 'Accuracy within 1 SD')) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)") + 
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE),
         linetype = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "Sentiment Baseline", "Random baseline"))) +
  facet_grid(~ Class)
ggsave('plots/context_accuracy_scaling_2metric_all.png', width=8, height=3.25, units='in')


all_context_accuracies_plot %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=2) +
  geom_hline(data = human_total_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
#  geom_hline(data = sentiment_acc_by_class, 
#             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = random_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')
        ) +
#  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE),
#         linetype = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "Random Baseline"))) +
  facet_grid(~ Class) ->
  context_accuracy_scaling_1metric_plot
context_accuracy_scaling_1metric_plot
ggsave('plots/context_accuracy_scaling_1metric_all.png', width=8, height=2.25, units='in')

context_accuracy_scaling_1metric_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  facet_grid(~ Class, 
             labeller = labeller(Class = c("Privative context" = "Priv. context",
                                           "Subsective context" = "Subs. context",
                                           "High frequency" = "High freq.",
                                           "Zero frequency" = "Zero freq.",
                                           "Overall" = "Overall")))
ggsave('plots/context_accuracy_scaling_1metric_all_diss.png', width=6.5, height=2, units='in')


all_context_accuracies_plot %>%
  filter(Class %in% c("High frequency", "Zero frequency", "Overall")) %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=2) +
  geom_hline(data = human_total_acc_by_class %>%
               filter(Class %in% c("High frequency", "Zero frequency", "Overall")), 
             aes(yintercept=Accuracy, color=Model)) +
  #  geom_hline(data = sentiment_acc_by_class, 
  #             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = random_acc_by_class %>%
               filter(Class %in% c("High frequency", "Zero frequency", "Overall")), 
             aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        #        legend.box.spacing = unit(0, 'pt')
  ) +
  #  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE),
  #         linetype = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "Random Baseline"))) +
  facet_grid(~ Class)
ggsave('plots/context_accuracy_scaling_1metric_3way.png', width=6, height=2.25, units='in')

all_context_accuracies_plot %>%
  filter(Class %in% c("Overall")) %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=2) +
  geom_hline(data = human_total_acc_by_class %>%
               filter(Class %in% c("Overall")), 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = random_acc_by_class %>%
               filter(Class %in% c("Overall")), 
             aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
  ) +
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "Random Baseline"))) +
  facet_grid(~ Class)
ggsave('plots/context_accuracy_scaling_1metric_overall.png', width=3, height=2, units='in')

all_context_accuracies_plot %>%
  filter(Class %in% c("Zero frequency")) %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=2) +
  geom_hline(data = human_total_acc_by_class %>%
               filter(Class %in% c("Zero frequency")), 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = random_acc_by_class %>%
               filter(Class %in% c("Zero frequency")), 
             aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
  ) +
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "Random Baseline"))) +
  facet_grid(~ Class)
ggsave('plots/context_accuracy_scaling_1metric_zero.png', width=3, height=2, units='in')

all_context_accuracies_plot %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model), 
            data=all_context_1sd_accuracies_plot) +
  geom_point(aes(color=Model), size=2,
             data=all_context_1sd_accuracies_plot) +
  geom_hline(data = human_total_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)
  ) +
  geom_hline(data = random_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)
  ) +
#  geom_hline(data = sentiment_1sd_acc_by_class,
#             aes(yintercept=Accuracy, color=Model)
#  ) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)", y="Accuracy within 1 SD") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')
        ) +
#  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE),
#         linetype = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "Random Baseline"))) +
  facet_grid(~ Class) ->
  ctxt_1sd_plot
ctxt_1sd_plot
ggsave('plots/context_accuracy_scaling_1sd_metric_all.png', width=8, height=2.25, units='in')

ctxt_1sd_plot + 
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  facet_grid(~ Class, 
             labeller = labeller(Class = c("Privative context" = "Priv. context",
                                           "Subsective context" = "Subs. context",
                                           "High frequency" = "High freq.",
                                           "Zero frequency" = "Zero freq.",
                                           "Overall" = "Overall")))
ggsave('plots/context_accuracy_scaling_1sd_metric_all_diss.png', width=6.5, height=2, units='in')


prepare_1sd_accuracies(all_context_base_1sd_accuracies %>% filter(Shots == 5)) ->       
  all_context_base_1sd_accuracies_plot

all_context_base_accuracies %>%
  filter(Shots == 5 & Class %in% c("Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class=fct_relevel(Class, "Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class = fct_recode(Class, "Privative context" = "Privative", "Subsective context" = "Subsective")) ->       
  all_context_base_accuracies_plot

all_context_base_accuracies_plot %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model,linetype='Accuracy')) +
  geom_line(aes(color=Model,linetype='Accuracy within 1 SD'), 
            data=all_context_base_1sd_accuracies_plot) +
  geom_point(aes(color=Model), size=2) +
  geom_point(aes(color=Model), size=2, shape=1,
             data=all_context_base_1sd_accuracies_plot) +
  geom_hline(data = human_total_acc_by_class, 
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy")) +
  geom_hline(data = sentiment_acc_by_class, 
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy")) +
  geom_hline(data = human_total_1sd_acc_by_class,
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy within 1 SD")
  ) +
  geom_hline(data = sentiment_1sd_acc_by_class,
             aes(yintercept=Accuracy, color=Model, linetype = "Accuracy within 1 SD")
  ) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  scale_linetype(name='Metric',
                 breaks=c('Accuracy', 'Accuracy within 1 SD')) +
  theme_minimal() +
  labs(color='Rating Source') + 
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE),
         linetype = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_discrete(type = c("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#81ba31", "#5a18dd", "#BD45F1")) +
  facet_grid(~ Class)
ggsave('plots/context_accuracy_scaling_2metric_base.png', width=8, height=3.25, units='in')

all_context_base_accuracies_plot %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=2) +
  geom_hline(data = human_total_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
#  geom_hline(data = sentiment_acc_by_class, 
#             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = random_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')
        ) +
#  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE),
#         linetype = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3", "Qwen 2", "Llama 2", "Mixtral", "Random Baseline"))) +
  
  facet_grid(~ Class) ->
  ctxt_1metric_base_plot
ctxt_1metric_base_plot
ggsave('plots/context_accuracy_scaling_1metric_base.png', width=8, height=2.25, units='in')

ctxt_1metric_base_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  facet_grid(~ Class, 
             labeller = labeller(Class = c("Privative context" = "Priv. context",
                                           "Subsective context" = "Subs. context",
                                           "High frequency" = "High freq.",
                                           "Zero frequency" = "Zero freq.",
                                           "Overall" = "Overall")))
ggsave('plots/context_accuracy_scaling_1metric_base_diss.png', width=6.5, height=2, units='in')


all_context_base_accuracies_plot %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model), 
            data=all_context_base_1sd_accuracies_plot) +
  geom_point(aes(color=Model), size=2,
             data=all_context_base_1sd_accuracies_plot) +
  geom_hline(data = human_total_1sd_acc_by_class,
             aes(yintercept=Accuracy, color=Model)
  ) +
#  geom_hline(data = sentiment_1sd_acc_by_class,
#             aes(yintercept=Accuracy, color=Model)
#  ) +
  geom_hline(data = random_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)
  ) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)", y="Accuracy within 1 SD") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')
        ) +
#  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE),
#         linetype = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3", "Qwen 2", "Llama 2", "Mixtral", "Random Baseline"))) +
  facet_grid(~ Class) ->
  ctxt_1sd_base_plot
ctxt_1sd_base_plot
ggsave('plots/context_accuracy_scaling_1sd_metric_base.png', width=8, height=2.25, units='in')

ctxt_1sd_base_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  facet_grid(~ Class, 
             labeller = labeller(Class = c("Privative context" = "Priv. context",
                                           "Subsective context" = "Subs. context",
                                           "High frequency" = "High freq.",
                                           "Zero frequency" = "Zero freq.",
                                           "Overall" = "Overall")))
ggsave('plots/context_accuracy_scaling_1sd_metric_base_diss.png', width=6.5, height=2, units='in')


## High quality contexts only ----

llama3i_5shot_hq_context_responses %>%
  context_direct_accuracy() ->
  llama3i_5shot_hq_context_acc
llama3i_5shot_hq_context_acc

human_hq_context_responses %>%
  context_direct_accuracy() ->
  human_hq_context_acc
human_hq_context_acc


# Effect of sentiment ----

## Load ratings ----

sentiment_ratings <- read.csv("../Adjectives-PythonCode/output/lm_sentiment_data/llama3i_context_ratings.csv")

sentiment_ratings %>%
  pivot_longer(
    cols = SentimentPrivative:SentimentSubsective,
    names_to = c("ContextBias"),
    names_pattern = "Sentiment([A-Za-z]+)",
    values_to = "SentRating"
  ) %>%
  mutate(SentRating = factor(SentRating, 
                        levels=c("Negative", "Slightly negative", 
                                 "Neutral", 
                                 "Slightly positive", "Positive")),
         NumSentRating = as.integer(SentRating)) %>%
  mutate(Rating = case_when(SentRating == "Negative" ~ "Definitely not",
                            SentRating == "Slightly negative" ~ "Probably not",
                            SentRating == "Neutral" ~ "Unsure",
                            SentRating == "Slightly positive" ~ "Probably yes",
                            SentRating == "Positive" ~ "Definitely yes"
                            )) %>%
  mutate(Rating = factor(Rating, 
                        levels=c("Definitely not", "Probably not", 
                                 "Unsure", 
                                 "Probably yes", "Definitely yes")),
         NumRating = NumSentRating) %>%
  separate_wider_delim(Bigram, names=c("Adjective", "Noun"), delim=" ",
                       cols_remove = FALSE,
                       too_many = "merge") %>%
  mutate(across(c("Bigram", "ContextBias", "Adjective", "Noun"), as.factor)) %>%
  add_frequency() ->
  sentiment_ratings

## Accuracy as baseline ----

sentiment_ratings %>%
  context_direct_accuracy() ->
  sentiment_context_acc
sentiment_context_acc

sentiment_ratings %>%
  accuracy_1sd_context(context_variance_combined) ->
  sentiment_context_1sd_acc
sentiment_context_1sd_acc

sentiment_ratings %>% 
  mutate(ContextBias = fct_recode(ContextBias, "privative" = "Privative", 
                                  "subsective" = "Subsective")) %>%
  merge(high_quality_contexts, by = c("Bigram", "ContextBias")) ->
  sentiment_hq_context_responses 

sentiment_hq_context_responses  %>%
  context_direct_accuracy() ->
  sentiment_hq_context_acc
sentiment_hq_context_acc


## Fit model ----

llama3i_lscale_5shot_responses %>%
  merge(sentiment_ratings %>% select(Bigram, ContextBias, SentRating, NumSentRating), .by = c("Bigram", "ContextBias")) %>%
  add_frequency() ->
  llama3i_lscale_5shot_sent_responses

# None of these are significant without the no_context ratings, not even ContextBias...
sentiment_lm <- clmm(PredictedResponse ~ SentRating + ContextBias + CoarseFrequency + (1 | Adjective) ,  
                     data = llama3i_lscale_5shot_sent_responses, 
                     link = "logit")
summary(sentiment_lm)
tidy(sentiment_lm) 

sentiment_lm3 <- clmm(PredictedResponse ~ SentRating + (1 | Noun),  
                     data = llama3i_lscale_5shot_sent_responses, 
                     link = "logit")
summary(sentiment_lm3)
tidy(sentiment_lm3) 

# There's no significant correlation numerically either and the slope even goes in the wrong direction
sentiment_lm2 <- lmer(NumPredictedResponse ~ NumSentRating + (1 | Adjective) + (1 | Noun),  
                      data = llama3i_lscale_5shot_sent_responses)
summary(sentiment_lm2)
plot(allEffects(sentiment_lm2))

# Positive sentiment is correlated with human-written stories where the privative adjective behaves subsectively
context_sentiment_lm <- clm(ContextBias ~ SentRating,
                             data = llama3i_lscale_5shot_sent_responses)
summary(context_sentiment_lm)

llama3i_lscale_5shot_sent_responses %>%
  filter(NumSentRating >= 4) %>%
  select(Bigram, ContextBias, PredictedResponse, SentRating) %>%
  View()
