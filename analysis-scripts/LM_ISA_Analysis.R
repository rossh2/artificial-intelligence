library(tidyverse)
library(tidytext)
library(effects)
library(lme4)
library(lmerTest)
library(distr)
library(ordinal)
library(MASS)
library(emmeans)
library(ggeffects)
library(philentropy)
select = dplyr::select

source('LM_Analysis_Utils.R')

# Original ISA experiment bigrams, Llama 2 
# NOT PROVIDED, BUT VARIABLES KEPT SINCE REFERENCED BELOW
llama2c_lscale_0shot_responses = read.csv("")
llama2c_nscale_0shot_responses = read.csv("")
llama2c_lscale_5shot_responses = read.csv("")
llama2c_nscale_5shot_responses = read.csv("")
llama2c_nostill_nscale_5shot_responses = read.csv("")

# Original + analogy ISA experiment bigrams, Llama 2 (but without chat template)
# NOT PROVIDED, BUT VARIABLES KEPT SINCE REFERENCED BELOW
llama2c_lscale_0shot_all_responses = read.csv("")

# Original + analogy ISA experiment bigrams, Llama 3 (but without chat template)
# NOT PROVIDED, BUT VARIABLES KEPT SINCE REFERENCED BELOW
llama3i_lscale_0shot_responses = read.csv("")
llama3i_nscale_5shot_responses = read.csv("")

# Original + analogy ISA experiment bigrams, Llama 3, with chat template
llama2c_lscale_5shot_all_responses = read.csv("results/llm/isa/predictions_isa_Llama-2-70b-chat-hf_labelledscale-5shot.csv")
llama3i_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Meta-Llama-3-70B-Instruct_labelledscale-5shot.csv")

# Original + analogy ISA, new models
llama2c_7b_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Llama-2-7b-chat-hf_labelledscale-5shot.csv")
llama2c_13b_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Llama-2-13b-chat-hf_labelledscale-5shot.csv")
llama3i_8b_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Meta-Llama-3-8B-Instruct_labelledscale-5shot.csv")
mixtral_8x7b_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Mixtral-8x7B-Instruct-v0.1-labelledscale-5shot.csv")
qwen2i_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Qwen2-72B-Instruct-labelledscale-5shot.csv")

llama2_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Llama-2-70b-hf_labelledscale-qa-5shot.csv")
llama2_7b_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Llama-2-7b-hf_labelledscale-qa-5shot.csv")
llama2_13b_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Llama-2-13b-hf_labelledscale-qa-5shot.csv")

llama3_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Meta-Llama-3-70B_labelledscale-qa-5shot.csv")
llama3_8b_lscale_5shot_responses = read.csv("results/llm/isa/predictions_isa_Meta-Llama-3-8B_labelledscale-qa-5shot.csv")

isa_data_12_capped = read.csv("isa_data_12_capped.csv")
isa_variance_12_capped = read.csv("isa_variance_12_capped.csv")
isa_variance_12_combined = read.csv("isa_variance_12_combined.csv")

## Preprocessing ----

isa_data_12_capped %>%
  mutate(across(c(ParticipantId, Adjective, Noun, Bigram, IsaRating, AdjectiveClass, CoarseFrequency), as.factor)) ->
  isa_data_12_capped

bigram_count = nrow(isa_variance_12_capped)
scale_length = 5
ratings_per_item = 12
isa_variance_12_capped %>%
  mutate(across(c(Adjective, Noun, Bigram, AdjectiveClass, CoarseFrequency), as.factor)) %>%
  rename(HumanMean = Mean, HumanVariance = Variance, HumanSD = SD) ->
  isa_variance_12_capped

#merge(isa_data_12_capped, 
#      isa_variance_12_capped %>% select(Bigram, HumanMean, HumanVariance, HumanSD), 
#      by= "Bigram") %>%
#  mutate(NumIsaRatingScaled = NumIsaRating / scale_length) ->
#  isa_data_12_capped

llama2c_lscale_0shot_responses %>%
  preprocess_labelled_responses() ->
  llama2c_lscale_0shot_responses

llama2c_lscale_5shot_responses %>%
  preprocess_labelled_responses() ->
  llama2c_lscale_5shot_responses

llama2c_lscale_5shot_all_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2c_lscale_5shot_all_responses

llama3i_lscale_0shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3i_lscale_0shot_responses

llama3i_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3i_lscale_5shot_responses

llama3i_nscale_5shot_responses %>%
  remove_duplicate_bigrams() ->
  llama3i_nscale_5shot_responses

llama2c_7b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2c_7b_lscale_5shot_responses

llama2c_13b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama2c_13b_lscale_5shot_responses

llama3i_8b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  llama3i_8b_lscale_5shot_responses

mixtral_8x7b_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  mixtral_8x7b_lscale_5shot_responses

qwen2i_lscale_5shot_responses %>%
  preprocess_labelled_responses() %>%
  remove_duplicate_bigrams() ->
  qwen2i_lscale_5shot_responses

## Troubleshooting

llama2c_lscale_5shot_all_responses %>%
  remove_duplicate_bigrams() %>%
  group_by(Bigram) %>%
  summarize(Count = n()) %>%
  filter(Count > 1) %>%
  pull(Bigram) -> analogy_original_duplicate_bigrams

llama3i_lscale_5shot_responses %>%
  remove_duplicate_bigrams() %>%
  filter(Bigram %in% analogy_original_duplicate_bigrams) %>%
  select(Bigram, PredictedResponse)

## Sampling ----

llama2c_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(ratings_per_item) %>%
  merge(isa_variance_12_capped %>% select(Bigram, AdjectiveClass, CoarseFrequency, Count),
        by="Bigram") %>%
  select(ParticipantId, Bigram, Adjective, Noun, IsaRating, NumIsaRating, AdjectiveClass, CoarseFrequency, Count) -> 
  llama2c_lscale_5shot_sampled_responses

llama2c_lscale_5shot_all_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(ratings_per_item) %>%
  merge(isa_variance_12_combined %>% select(Bigram, AdjectiveClass, CoarseFrequency, Count),
        by="Bigram") %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating, AdjectiveClass, CoarseFrequency, Count) -> 
  llama2c_lscale_5shot_all_sampled_responses

llama3i_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() -> 
  llama3i_lscale_5shot_sampled_responses

llama3i_8b_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() ->  
  llama3i_8b_lscale_5shot_sampled_responses

llama2c_7b_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() ->  
  llama2c_7b_lscale_5shot_sampled_responses

llama2c_13b_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() -> 
  llama2c_13b_lscale_5shot_sampled_responses

llama2_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(ratings_per_item) %>%
  merge(isa_variance_12_combined %>% select(Bigram, AdjectiveClass, CoarseFrequency, Count),
        by="Bigram") %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating, AdjectiveClass, CoarseFrequency, Count) -> 
  llama2_lscale_5shot_sampled_responses

llama3_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() -> 
  llama3_lscale_5shot_sampled_responses

llama3_8b_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() ->  
  llama3_8b_lscale_5shot_sampled_responses

llama2_7b_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() ->  
  llama2_7b_lscale_5shot_sampled_responses

llama2_13b_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(12) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating) %>%
  add_frequency() -> 
  llama2_13b_lscale_5shot_sampled_responses

mixtral_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(ratings_per_item) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating, AdjectiveClass, CoarseFrequency, Count) %>%
  add_frequency() -> 
  mixtral_lscale_5shot_all_sampled_responses

qwen2i_lscale_5shot_responses %>%
  sample_preprocess_labelled_responses() %>%
  sample_responses(ratings_per_item) %>%
  select(ParticipantId, Bigram, Adjective, Noun, Rating, NumRating, AdjectiveClass, CoarseFrequency, Count) %>%
  add_frequency() -> 
  qwen2i_lscale_5shot_all_sampled_responses

  
## Combine data ----

label.human = "Human"
# All of the below use "still"
label.llama2c_lscale_0shot = 'Llama 2 Chat (0-shot, labelled)'
label.llama2c_nscale_0shot = 'Llama 2 Chat (0-shot, numeric)'
label.llama2c_lscale_5shot = 'Llama 2 Chat (5-shot, labelled)'
label.llama2c_nscale_5shot = 'Llama 2 Chat (5-shot, numeric)'
label.llama2c_nostill_nscale_5shot = 'Llama 2 Chat (5-shot, numeric, without still)'
label.llama2c_5shot_cot = 'Llama 2 Chat (5-shot, CoT)'
label.llama2c_lscale_5shot_all = 'Llama 2 Chat (5-shot, labelled) [all bigrams]'
label.llama3i_lscale_5shot_all = 'Llama 3 Instruct (5-shot, labelled) [all bigrams]'
label.llama3i_lscale_0shot_all = 'Llama 3 Instruct (0-shot, labelled) [all bigrams]'
label.llama3i_nscale_5shot_all = 'Llama 3 Instruct (5-shot, numeric) [all bigrams]'


human_lm_ratings = bind_rows(isa_data_12_capped %>% 
                               mutate(HumanOrLM = label.human), 
                             llama2c_lscale_5shot_all_sampled_responses %>%
                               mutate(HumanOrLM = label.llama2c_lscale_5shot),
                             # llama2c_nscale_5shot_sampled_responses %>%
                             #   mutate(HumanOrLM = label.llama2c_nscale_5shot),
                             # llama2c_nostill_nscale_5shot_sampled_responses %>%
                             #   mutate(HumanOrLM = label.llama2c_nostill_nscale_5shot),
                             llama3i_lscale_5shot_sampled_responses %>%
                               mutate(HumanOrLM = label.llama3i_lscale_5shot_all)
                             # llama2c_5shot_cot_responses %>%
                             #   mutate(HumanOrLM = label.llama2c_5shot_cot,
                             #          ParticipantId = "Llama2Chat") %>%
                             #   select(!c(HumanMean, HumanSD))
                             ) %>%
  mutate(CoarseFrequency = fct_relevel(CoarseFrequency, "Zero")) %>%
  mutate(across(c(ParticipantId, HumanOrLM, IsaRating), as.factor)) %>%
  mutate(IsaRating = fct_relevel(IsaRating, "Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"),
         AdjectiveClass = fct_relevel(AdjectiveClass, "Subsective"))



# Plots ----

## Density plots ----

human_lm_ratings %>%
  filter(HumanOrLM == label.llama2c_lscale_5shot & Bigram == "fake reef") %>%
  ggplot(aes(x=NumIsaRating)) +
  geom_density(fill="#D81B60", alpha=1) +
  labs(y="Density", x="Rating", title="Distribution of ratings for Llama 2 Chat for \"Is a fake reef a reef?\"") +
  ggtitle("") +
  xlim(c(1,5)) +   
  theme_minimal() + 
  theme(text=element_text(size=36),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank()
)
ggsave("plots/llama2chat_fake_reef_sampled_rating_density.png", units='in', height=3, width=4, dpi=300)
ggsave("plots/llama2chat_fake_reef_sampled_rating_density_poster.png", units='in', height=6, width=8, dpi=300)

human_lm_ratings %>%
  filter(HumanOrLM == label.llama2c_lscale_5shot & Bigram == "fake concert") %>%
  ggplot(aes(x=NumIsaRating)) +
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
ggsave("plots/llama2chat_fake_concert_sampled_rating_density_poster.png", units='in', height=6, width=8, dpi=300)

human_lm_ratings %>%
  filter(HumanOrLM == label.human & Bigram == "fake crowd") %>%
  ggplot(aes(x=NumRating)) +
  geom_density(fill="#1E88E5", alpha=1) +
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
ggsave("plots/human_fake_crowd_rating_density_poster.png", units='in', height=6, width=8, dpi=300)

human_lm_ratings %>%
  filter(HumanOrLM == label.llama2c_nscale_5shot & Bigram == "fake reef") %>%
  ggplot(aes(x=NumIsaRating)) +
  geom_histogram(aes(y = after_stat(count / sum(count))), fill="blue", alpha=0.3, bins = 5) +
  labs(y="Rating Probability", x="Rating", title="Distribution of ratings for Llama 2 Chat for \"Is a fake reef a reef?\"") +
  theme_minimal() + 
  theme(text=element_text(size=36))
ggsave("plots/llama2chat_fake_reef_sampled_rating_histogram.png")

human_lm_ratings %>%
  filter(Bigram == "fake reef") %>%
  ggplot(aes(x=NumIsaRating, fill=HumanOrLM)) +
  geom_density(alpha=0.3) +
  labs(y="Density", x="Rating", title="Distribution of ratings for \"Is a fake reef a reef?\"") +
  xlim(c(0,5)) +
  theme(legend.title = element_blank())

human_lm_ratings %>%
  filter(Bigram == "fake reef") %>%
  ggplot(aes(x=NumIsaRating, fill=HumanOrLM)) +
  geom_histogram(aes(y = after_stat(count / sum(count))), alpha=0.3, bins = 5, position="dodge") +
  labs(y="Rating Probability", x="Rating", title="Distribution of ratings for \"Is a fake reef a reef?\"") +
  theme(legend.title = element_blank())

## Spilt violin plots ----

### Llama 2 Chat numeric scale, 5-shot with still

human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_nscale_5shot)) %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency == "25th-50th percentile" ~ "Low frequency",
    CoarseFrequency %in% c("50th-75th percentile", "75th-90th percentile", "90th-99th percentile") ~ "High frequency"
  )) %>%
  ggplot(aes(x=Adjective, 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin() +
  facet_wrap(~ CoarserFrequency) +
  ggtitle("Human vs. LM ratings") + 
  xlab("Noun") +
  ylab("Rating") + 
  guides(x = guide_axis(angle = 90))

human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_nscale_5shot)) %>%
  filter(Adjective == "fake") %>%
  filter(Bigram %in% c("fake glance", "fake pole", "fake reef", "fake scarf",
                       "fake act", "fake gold", "fake gun", "fake laugh", "fake image", "fake fire",
                       "fake concert", "fake crowd", "fake dollar", "fake lion", "fake jacket", "fake door")) %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency %in% c("25th-50th percentile", "50th-75th percentile") ~ "Low/medium frequency",
    CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency"
  )) %>%
  ggplot(aes(x=Noun, 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin() +
  facet_wrap(~ CoarserFrequency, scales = "free_x", ncol=1) +
  ggtitle("Human vs. LM ratings for 'Is a fake N an N? (selected nouns)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_fill_discrete("Rating Source") +
  guides(x = guide_axis(angle = 90))
ggsave("plots/lm_isa_splitviolin_fake_selected_nouns_llama2chat-numeric-5shot.png", units='in', width=8.5, height=5)

human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_lscale_5shot)) %>%
  mutate(HumanOrLM = fct_recode(HumanOrLM, "Llama 2 Chat (5-shot)" = label.llama2c_lscale_5shot)) %>%
  filter(Adjective == "fake") %>%
#  filter(Bigram %in% c("fake laugh", "fake concert", "fake jacket")) %>%
  filter(Noun %in% c("dollar", "concert", "fire", 
                       "gun", "crowd", 
                       "jacket", "report", "laugh", "image")) %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency %in% c("25th-50th percentile", "50th-75th percentile") ~ "Low/medium frequency",
    CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency"
  )) %>%
  ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                              within=Adjective,fun=median), #Noun, 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin(adjust=3) +
  ggtitle("Human vs. LLM distribution for 'Is a fake N still an N?") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_fill_manual(name='Data Source',
                      breaks=c('Human', 
                               'Llama 2 Chat (5-shot)'),
                      values=c('Human'='#1E88E5', 
                               'Llama 2 Chat (5-shot)'='#D81B60'
                      )) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90), 
         fill=guide_legend(ncol=2)) +
  theme_minimal() + 
  theme(text=element_text(size=36, color="#2C365E"),
        legend.position="bottom",
        panel.grid.major.x = element_blank(),
#        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_blank(),
#        panel.grid.minor.y = element_blank()
  )
#ggsave("plots/lm_isa_splitviolin_fake_selected_nouns_llama2chat-5shot-poster.png", units='in', width=8, height=8)
ggsave("plots/lm_isa_splitviolin_fake_selected_nouns_llama2chat-5shot-poster.png", units='in', width=16, height=7)


human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_nscale_5shot)) %>%
  filter(Adjective == "useful") %>%
  filter(Bigram %in% c("useful reef", "useful sofa", 
                       "useful fire", "useful jacket", "useful pole", "useful scarf",
                       "useful car", "useful fact", "useful plan", "useful idea", "useful sign", "useful work")) %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency %in% c("25th-50th percentile", "50th-75th percentile") ~ "Low/medium frequency",
    CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency"
  )) %>%
  ggplot(aes(x=Noun, 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin(scale = "area") +
  stat_summary(aes(fill=HumanOrLM),
               col="black",
               shape=21,
               geom = "point",
               fun.y = "mean") +
  facet_wrap(~ CoarserFrequency, scales = "free_x", ncol=1) +
  ggtitle("Human vs. LM ratings for 'Is a useful N an N? (selected nouns)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_fill_discrete("Rating Source") +
  guides(x = guide_axis(angle = 90))
ggsave("plots/lm_isa_splitviolin_useful_selected_nouns_llama2chat-numeric-5shot.png", units='in', width=8.5, height=5)

human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_nostill_nscale_5shot)) %>%
  filter(Adjective == "useful") %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency %in% c("25th-50th percentile", "50th-75th percentile") ~ "Low/medium frequency",
    CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency"
  )) %>%
  ggplot(aes(x=Noun, 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin() +
  facet_wrap(~ CoarserFrequency, scales = "free_x", ncol=1) +
  ggtitle("Human vs. LM ratings for 'Is a useful N an N? (selected nouns)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_fill_discrete("Rating Source") +
  guides(x = guide_axis(angle = 90))

human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_nscale_5shot)) %>%
  ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective), 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin() +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Human vs. LM ratings") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))

### Llama 2 Chat labelled scale, 5-shot with still

human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_lscale_5shot)) %>%
  filter(Adjective == "fake") %>%
  filter(Bigram %in% c("fake glance", "fake pole", "fake reef", "fake scarf",
                       "fake act", "fake gold", "fake gun", "fake laugh", "fake image", "fake fire",
                       "fake concert", "fake crowd", "fake dollar", "fake lion", "fake jacket", "fake door")) %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency %in% c("25th-50th percentile", "50th-75th percentile") ~ "Low/medium frequency",
    CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency"
  )) %>%
  ggplot(aes(x=Noun, 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin() +
  stat_summary(aes(fill=HumanOrLM),
               col="black",
               shape=22,
               geom = "point",
               fun.y = "mean") +
  facet_wrap(~ CoarserFrequency, scales = "free_x", ncol=1) +
  ggtitle("Human vs. LM ratings for 'Is a fake N an N? (selected nouns)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_fill_discrete("Rating Source") +
  guides(x = guide_axis(angle = 90))
ggsave("plots/lm_isa_splitviolin_fake_selected_nouns_llama2chat-labelled-5shot.png", units='in', width=8.5, height=5)

human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_lscale_5shot)) %>%
  filter(Adjective == "useful") %>%
  filter(Bigram %in% c("useful reef", "useful sofa", 
                       "useful fire", "useful jacket", "useful pole", "useful scarf",
                       "useful car", "useful fact", "useful plan", "useful idea", "useful sign", "useful work")) %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero frequency",
    CoarseFrequency %in% c("25th-50th percentile", "50th-75th percentile") ~ "Low/medium frequency",
    CoarseFrequency %in% c("75th-90th percentile", "90th-99th percentile") ~ "High frequency"
  )) %>%
  ggplot(aes(x=Noun, 
             y=NumIsaRating, fill=HumanOrLM)) +
  geom_split_violin(scale = "area") +
  stat_summary(aes(fill=HumanOrLM),
               col="black",
               shape=22,
               geom = "point",
               fun.y = "mean") +
  facet_wrap(~ CoarserFrequency, scales = "free_x", ncol=1) +
  ggtitle("Human vs. LM ratings for 'Is a useful N an N? (selected nouns)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_fill_discrete("Rating Source") +
  guides(x = guide_axis(angle = 90))
ggsave("plots/lm_isa_splitviolin_useful_selected_nouns_llama2chat-labelled-5shot.png", units='in', width=8.5, height=5)


## Correlation plots ----

isa_data_12_capped %>%
  merge(llama2c_lscale_5shot_responses %>%
          select(Bigram, PredictedResponse, NumPredictedResponse),
        .by="Bigram") %>%
  mutate(PredictedResponse = factor(PredictedResponse, levels=c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes")),
         IsaRating = fct_relevel(IsaRating, "Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes")) %>%
  group_by(AdjectiveClass, IsaRating, PredictedResponse) %>%
  summarize(Count = dplyr::n(), .groups = "drop") -> human_vs_llama2c_lscale_5shot_responses

human_vs_llama2c_lscale_5shot_responses %>%
  ggplot(aes(x=PredictedResponse, y=IsaRating, fill=Count)) +
  geom_tile() +
  facet_wrap(~ AdjectiveClass) +
  labs(title="Correlation (as counts) between LLM and human ratings",
       x="Llama 2 Chat Rating (5-shot)", y="Human Rating") +
  guides(x = guide_axis(angle = 90)) + 
  theme_minimal() + 
  theme(text=element_text(size=24),
        legend.position="right")
ggsave("plots/lm_isa_llama2c_lscale_5shot_human_vs_prediction_correlation_heatmap.png",
       dpi=300, width=15, height=9, units="in")


## Plot against human data (1 SD from mean) ----

### Old plots ----

isa_variance_12_capped %>%
  rename(NumIsaRating = HumanMean) %>%
  ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                              within=Adjective,fun=median), 
             y = NumIsaRating)) +
  geom_point(aes(color='Human')) +
  geom_errorbar(aes(ymin = round(NumIsaRating - HumanSD), ymax = round(NumIsaRating + HumanSD)), 
                width = 0.5) +
  geom_jitter(aes(color='Llama 2 Chat (0-shot, numeric)'), 
              data = llama2c_nscale_0shot_responses %>%
                rename(NumIsaRating = PredictedResponse), 
              shape=18, height=0.1, width=0.25) +
  geom_jitter(aes(color='Llama 2 Chat (0-shot, labelled)'), 
              data = llama2c_lscale_0shot_responses %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.25) +
  geom_jitter(aes(color='Llama 2 Chat (5-shot, numeric)'), 
              data = llama2c_nscale_5shot_responses %>%
                rename(NumIsaRating = PredictedResponse), 
              shape=18, height=0.1, width=0.25) +
  geom_jitter(aes(color='Llama 2 Chat (5-shot, labelled)'), 
              data = llama2c_lscale_5shot_responses %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.25) +
  geom_jitter(aes(color='Llama 2 Chat (5-shot, CoT)'), 
              data = llama2c_5shot_cot_responses, 
              shape=18, height=0.1, width=0.25) +
  facet_wrap(~ Adjective, scales = "free_x") +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  xlab("Noun") +
  ylab("Rating") +
  ylim(c(0.5, 5.5)) +
  ggtitle("Human vs. Llama 2 Chat ratings for \"Is an AN still an N?\"\n(error bars show one SD above/below mean, rounded to nearest integer rating)") +
  scale_color_manual(name='Data Source',
                     breaks=c('Human', 
                              'Llama 2 Chat (0-shot, numeric)', 
                              'Llama 2 Chat (0-shot, labelled)',
                              'Llama 2 Chat (5-shot, numeric)', 
                              'Llama 2 Chat (5-shot, labelled)',
                              'Llama 2 Chat (5-shot, CoT)'),
                     values=c('Human'='black', 
                              'Llama 2 Chat (0-shot, numeric)'='#ff7f00', 
                              'Llama 2 Chat (0-shot, labelled)'='#984ea3',
                              'Llama 2 Chat (5-shot, numeric)'='#e41a1c',
                              'Llama 2 Chat (5-shot, labelled)'='#377eb8',
                              'Llama 2 Chat (5-shot, CoT)'='#4daf4a'
                     ))
ggsave("plots/lm_isa_humanvsgeneration_allAs_llama2c_allmodels.png", units="in", width=15, height=8)

plot_1sd_all_llama2 <- function(adjective) {
  return(isa_variance_12_capped %>%
           filter(Adjective == adjective) %>%
           rename(NumIsaRating = HumanMean) %>%
           ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                                       within=Adjective,fun=median), 
                      y = NumIsaRating)) +
           geom_point(aes(color='Human')) +
           geom_errorbar(aes(ymin = round(NumIsaRating - HumanSD), ymax = round(NumIsaRating + HumanSD)), 
                         width = 0.2) +
           geom_jitter(aes(color='Llama 2 Chat (0-shot, numeric)'), 
                       data = llama2c_nscale_0shot_responses %>%
                         filter(Adjective == adjective) %>%
                         rename(NumIsaRating = PredictedResponse), 
                       shape=18, height=0.1, width=0.25) +
           geom_jitter(aes(color='Llama 2 Chat (0-shot, labelled)'), 
                       data = llama2c_lscale_0shot_responses %>%
                         filter(Adjective == adjective) %>%
                         rename(NumIsaRating = NumPredictedResponse), 
                       shape=18, height=0.1, width=0.25) +
           geom_jitter(aes(color='Llama 2 Chat (5-shot, numeric)'), 
                       data = llama2c_nscale_5shot_responses %>%
                         filter(Adjective == adjective) %>%
                         rename(NumIsaRating = PredictedResponse), 
                       shape=18, height=0.1, width=0.25) +
           geom_jitter(aes(color='Llama 2 Chat (5-shot, labelled)'), 
                       data = llama2c_lscale_5shot_responses %>%
                         filter(Adjective == adjective) %>%
                         rename(NumIsaRating = NumPredictedResponse), 
                       shape=18, height=0.1, width=0.25) +
           geom_jitter(aes(color='Llama 2 Chat (5-shot, CoT)'), 
                       data = llama2c_5shot_cot_responses %>% 
                         filter(Adjective == adjective), 
                       shape=18, height=0.1, width=0.25) +
           scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
           guides(x = guide_axis(angle = 90)) +
           xlab("Noun") +
           ylab("Rating") +
           ggtitle(sprintf("Human vs. Llama 2 Chat ratings for \"Is a %s N still an N?\"\n(error bars show one SD above/below mean, rounded to nearest integer rating)", adjective)) +
           scale_color_manual(name='Data Source',
                              breaks=c('Human', 
                                       'Llama 2 Chat (0-shot, numeric)', 
                                       'Llama 2 Chat (0-shot, labelled)',
                                       'Llama 2 Chat (5-shot, numeric)', 
                                       'Llama 2 Chat (5-shot, labelled)',
                                       'Llama 2 Chat (5-shot, CoT)'),
                              values=c('Human'='black', 
                                       'Llama 2 Chat (0-shot, numeric)'='#ff7f00', 
                                       'Llama 2 Chat (0-shot, labelled)'='#984ea3',
                                       'Llama 2 Chat (5-shot, numeric)'='#e41a1c',
                                       'Llama 2 Chat (5-shot, labelled)'='#377eb8',
                                       'Llama 2 Chat (5-shot, CoT)'='#4daf4a'
                              )))
  
}

plot_1sd_all_llama3 <- function() {
  return(isa_variance_12_combined %>%
           rename(NumIsaRating = Mean) %>%
           ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                                       within=Adjective,fun=median), 
                      y = NumIsaRating)) +
           geom_point(aes(color='Human')) +
           geom_errorbar(aes(ymin = pmax(1, round(NumIsaRating - SD)), ymax = pmin(5, round(NumIsaRating + SD))), 
                         width = 0.2) +
           geom_jitter(aes(color='Llama 3 Instruct (0-shot)'), 
                       data = llama3i_lscale_0shot_responses %>%
                         rename(NumIsaRating = NumPredictedResponse), 
                       shape=18, height=0.1, width=0.25) +
           geom_jitter(aes(color='Llama 3 Instruct (5-shot)'), 
                       data = llama3i_lscale_5shot_responses %>%
                         rename(NumIsaRating = NumPredictedResponse), 
                       shape=18, height=0.1, width=0.25) +
           scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
           guides(x = guide_axis(angle = 90)) +
           facet_wrap(~ Adjective, scales = "free_x") +
           xlab("Noun") +
           ylab("Rating") +
           ggtitle("Human vs. Llama 3 Instruct ratings for \"Is an AN still an N?\"\n(error bars show one SD above/below mean, rounded to nearest integer rating)") +
           scale_color_manual(name='Data Source',
                              breaks=c('Human', 
                                       'Llama 3 Instruct (0-shot)', 
                                       'Llama 3 Instruct (5-shot)'
                                       ),
                              values=c('Human'='black', 
                                       'Llama 3 Instruct (0-shot)'='#984ea3',
                                       'Llama 3 Instruct (5-shot)'='#377eb8'
                                       )
                              )
         )
}

plot_1sd_2v3 <- function(adjective) {
  return(isa_variance_12_combined%>%
           filter(Adjective == adjective) %>%
           rename(NumIsaRating = Mean) %>%
           ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                                       within=Adjective,fun=median), 
                      y = NumIsaRating)) +
           geom_point(aes(color='Human')) +
           geom_errorbar(aes(ymin = round(NumIsaRating - SD), ymax = round(NumIsaRating + SD)), 
                         width = 0.2) +
           geom_jitter(aes(color='Llama 2 Chat (5-shot, labelled)'), 
                       data = llama2c_lscale_5shot_all_responses %>%
                         filter(Adjective == adjective) %>%
                         rename(NumIsaRating = NumPredictedResponse), 
                       shape=18, height=0.1, width=0.25, size=3) +
           # geom_jitter(aes(color='Llama 2 Chat (5-shot, CoT)'), 
           #             data = llama2c_5shot_cot_responses %>% 
           #               filter(Adjective == adjective), 
           #             shape=18, height=0.1, width=0.25) +
           geom_jitter(aes(color='Llama 3 Instruct (0-shot, labelled)'), 
                       data = llama3i_lscale_0shot_responses %>%
                         filter(Adjective == adjective) %>%
                         rename(NumIsaRating = NumPredictedResponse), 
                       shape=18, height=0.1, width=0.25, size=3) +
           geom_jitter(aes(color='Llama 3 Instruct (5-shot, labelled)'), 
                       data = llama3i_lscale_5shot_responses %>%
                         filter(Adjective == adjective) %>%
                         rename(NumIsaRating = NumPredictedResponse), 
                       shape=18, height=0.1, width=0.25, size=3) +
           geom_jitter(aes(color='Llama 3 Instruct (5-shot, CoT)'), 
                       data = llama3i_5shot_cot_responses %>% 
                         filter(Adjective == adjective), 
                       shape=18, height=0.1, width=0.25, size=3) +
           scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
           guides(x = guide_axis(angle = 90)) +
           xlab("Noun") +
           ylab("Rating") +
           ggtitle(sprintf("Human vs. Llama 2 Chat vs. Llama 3 Instruct ratings for \"Is a %s N still an N?\"\n(error bars show one SD above/below mean, rounded to nearest integer rating)", adjective)) +
           scale_color_manual(name='Data Source',
                              breaks=c('Human', 
                                       'Llama 2 Chat (0-shot, labelled)', 
                                       'Llama 2 Chat (5-shot, labelled)',
                                       'Llama 2 Chat (5-shot, CoT)',
                                       'Llama 3 Instruct (0-shot, labelled)', 
                                       'Llama 3 Instruct (5-shot, labelled)',
                                       'Llama 3 Instruct (5-shot, CoT)'),
                              values=c('Human'='black', 
                                       'Llama 2 Chat (0-shot, labelled)'='#332288', 
                                       'Llama 2 Chat (5-shot, labelled)'='#117733',
                                       'Llama 2 Chat (5-shot, CoT)'='#44AA99',
                                       'Llama 3 Instruct (0-shot, labelled)'='#882255', 
                                       'Llama 3 Instruct (5-shot, labelled)'='#AA4499',
                                       'Llama 3 Instruct (5-shot, CoT)'='#CC6677'
                              )))
  
}

### Poster plots ----

plot_1sd_single_poster <- function(adjective, adj_det, nouns) {
  return(isa_variance_12_capped %>%
      filter(Adjective == adjective) %>%
      filter(Noun %in% nouns) %>%
      rename(NumIsaRating = HumanMean) %>%
      ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                                  within=Adjective,fun=median), 
                 y = NumIsaRating)) +
      geom_point(aes(color='Human'), size=7) +
      geom_errorbar(aes(ymin = round(NumIsaRating - HumanSD), ymax = round(NumIsaRating + HumanSD)), 
                    width = 0.4, linewidth=2, color="#2C365E") +
      geom_jitter(aes(color='Llama 2 Chat (5-shot)'), 
                  data = llama2c_lscale_5shot_responses %>%
                    filter(Adjective == adjective) %>%
                    filter(Noun %in% nouns) %>%
                    rename(NumIsaRating = NumPredictedResponse), 
                  shape=18, height=0.1, width=0.1, size=8) +
      scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
      guides(x = guide_axis(angle = 90)) +
      xlab("Noun") +
      ylab("Rating") +
      ggtitle(sprintf("LLM ratings for \'Is %s %s N still an N?\'", adj_det, adjective)) +
      scale_color_manual(name='Data Source',
                         breaks=c('Human', 
                                  'Llama 2 Chat (5-shot)'),
                         values=c('Human'='#2C365E', 
                                  'Llama 2 Chat (5-shot)'='#D81B60'
                         )) +
      scale_y_continuous(breaks=1:5, limits = c(0.5,5.5)) +
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

plot_1sd_all("fake")
ggsave("plots/lm_isa_humanvsgeneration_fake_llama2c_all-0-5shot.png", width=9, height=3.5, units="in")

plot_1sd_all("useful")
ggsave("plots/lm_isa_humanvsgeneration_useful_llama2c_all-0-5shot.png", width=9, height=3.5, units="in")

plot_1sd_all("illegal")
ggsave("plots/lm_isa_humanvsgeneration_illegal_llama2c_all-0-5shot.png", width=9, height=3.5, units="in")

plot_1sd_single_poster("fake", "a",
                       c("fact", "dollar", "concert", "reef", "fire", 
                         "gun", "door", "scarf", "crowd", "form",
                         "plan", "jacket", "report", "laugh", "image"))
ggsave("plots/lm_isa_1sd_fake_llama2c_5shot_poster.png", dpi=300, units="in",
       width=14, height=7, bg="transparent")

plot_1sd_single_poster("useful", "a", isa_variance_12 %>% distinct(Noun) %>% pull(Noun))
ggsave("plots/lm_isa_1sd_useful_llama2c_5shot_poster.png", dpi=300, units="in",
       width=18, height=9, bg="transparent")

plot_1sd_single_poster("illegal", "an")
ggsave("plots/lm_isa_1sd_illegal_llama2c_5shot_poster.png", dpi=300, units="in",
       width=18, height=9, bg="transparent")

### Llama 3 ----

plot_1sd_all_llama3()

plot_1sd_2v3("fake")

isa_variance_12_combined %>%
  rename(NumIsaRating = Mean) %>%
  ggplot(aes(x=reorder_within(x=Noun,by=NumIsaRating,
                              within=Adjective,fun=median), 
             y = NumIsaRating)) +
  geom_point(aes(color='Human'), size=2) +
  geom_errorbar(aes(ymin = round(NumIsaRating - SD), ymax = round(NumIsaRating + SD)), 
                width = 0.4, linewidth=0.75, color="#2C365E") +
  geom_jitter(aes(color='Llama 2 Chat (5-shot)'), 
              data = llama2c_lscale_5shot_all_responses %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.1, size=2) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  xlab("Noun") +
  ylab("Rating") +
  facet_wrap(~ Adjective, scales = 'free_x') +
  ggtitle("LLM ratings for \'Is an A N still an N?\'") +
  scale_color_manual(name='Data Source',
                     breaks=c('Human', 
                              'Llama 2 Chat (5-shot)'),
                     values=c('Human'='#2C365E', 
                              'Llama 2 Chat (5-shot)'='#D81B60'
                     )) +
  scale_y_continuous(breaks=1:5, limits = c(0.5,5.5)) +
  theme_minimal()

### GenBench plots ----


nouns = c("dollar", "concert", "fire", "abundance", "air", "door",
          "gun", "crowd", "reef", "photograph","information", "fact",
          "jacket", "report", "laugh", "bed", "lion", "watch", 
          "business", "art", "celebration",
          "truck", "chair", "handbag", "scarcity", "scarf", "fruit",
          "rumor", "painting", "flower", "hand")
all_nouns = llama3i_lscale_5shot_responses %>% distinct(Noun) %>% pull()
plot_1sd(isa_variance_12_combined, 
         llama3i_lscale_5shot_responses, lm_name="Llama 3 Instruct 70B",
         adjectives=c("fake"), nouns=nouns, context=FALSE, poster=TRUE) +
  xlab("Noun (fake {noun})")
ggsave('plots/lm_isa_1sd_llama3i_selected_fake_bigrams_poster.png', width=13, height=8, units='in')
plot_1sd(isa_variance_12_combined, 
         llama3i_lscale_5shot_responses, lm_name="Llama 3 Instruct 70B",
         adjectives=c("fake"), nouns=nouns, context=FALSE, title=FALSE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  xlab("Noun (fake {noun})")
ggsave('plots/lm_isa_1sd_llama3i_selected_fake_bigrams.png', width=4, height=2.75, units='in')

plot_1sd(isa_variance_12_combined, 
         llama3i_lscale_5shot_responses, lm_name="Llama 3 Instruct 70B",
         adjectives=c("fake"), nouns=nouns, context=FALSE, title=FALSE,
         human_color=magenta_color, lm_color = light_blue_color) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  xlab("Noun (fake {noun})") ->
  recolored_1sd_plot
ggsave('plots/lm_isa_1sd_llama3i_selected_fake_bigrams_recolored.png', width=4, height=2.75, units='in')

recolored_1sd_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  )
ggsave('plots/lm_isa_1sd_llama3i_selected_fake_bigrams_recolored_diss.png', 
       width=4.25, height=2.75, units='in')

# TODO make this into a function

isa_variance_combined %>%
  filter(Noun == "intelligence") %>%
  rename(NumIsaRating = Mean) %>%
  ggplot(aes(x=Bigram, 
             y = NumIsaRating)) +
  geom_point(aes(color='Human'), size=3) +
  geom_errorbar(aes(ymin = pmax(1, round(NumIsaRating - SD)), ymax = pmin(5, round(NumIsaRating + SD))), 
                width = 0.25, linewidth=1) +
  geom_jitter(aes(color='Llama 3 70B Instruct'), 
              data = llama3i_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.15, size=4) +
  geom_jitter(aes(color='Llama 3 8B Instruct'), 
              data = llama3i_8b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.15, size=4) +
  geom_jitter(aes(color='Llama 2 70B Chat'), 
              data = llama2c_lscale_5shot_all_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.15, size=4) +
  geom_jitter(aes(color='Llama 2 13B Chat'), 
              data = llama2c_13b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.15, size=4) +
  geom_jitter(aes(color='Llama 2 7B Chat'), 
              data = llama2c_7b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.15, size=4) +
  geom_jitter(aes(color='Mixtral 7x8B Instruct'), 
              data = mixtral_8x7b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.15, size=4) +
  geom_jitter(aes(color='Qwen 2 72B Instruct'), 
              data = qwen2i_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumIsaRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.15, size=4) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  xlab("Bigram") +
  ylab("Rating") +
  theme_minimal() +
  # scale_color_discrete(name="Model", type=paletteer_d("RColorBrewer::Set1", n=8))
 scale_color_discrete(name="Model", type = c("#000000", "#D81B60", "#1E88E5", "#FFC107", "#004D40", "#81ba31", "#5a18dd", "#BD45F1"))


isa_data_combined %>%
  filter(Noun == "intelligence") %>%
  ggplot(aes(x=Bigram, 
             y = NumRating)) +
  geom_violin(aes(color='Human'), adjust=3) +
  # geom_errorbar(aes(ymin = pmax(1, round(NumRating - SD)), ymax = pmin(5, round(NumRating + SD))), 
  #               width = 0.25, linewidth=1) +
  geom_jitter(aes(color='Llama 3 70B Instruct'), 
              data = llama3i_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse), 
              shape=18, height=0.2, width=0.3, size=4) +
  geom_jitter(aes(color='Llama 3 8B Instruct'), 
              data = llama3i_8b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.2, size=4) +
  geom_jitter(aes(color='Llama 2 70B Chat'), 
              data = llama2c_lscale_5shot_all_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse), 
              shape=18, height=0.2, width=0.3, size=4) +
  geom_jitter(aes(color='Llama 2 13B Chat'), 
              data = llama2c_13b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse), 
              shape=18, height=0.2, width=0.3, size=4) +
  geom_jitter(aes(color='Llama 2 7B Chat'), 
              data = llama2c_7b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse), 
              shape=18, height=0.2, width=0.3, size=4) +
  geom_jitter(aes(color='Mixtral 7x8B Instruct'), 
              data = mixtral_8x7b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse), 
              shape=18, height=0.2, width=0.3, size=4) +
  geom_jitter(aes(color='Qwen 2 72B Instruct'), 
              data = qwen2i_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse), 
              shape=18, height=0.1, width=0.2, size=4) +
  scale_x_discrete(labels = function(x) lapply(strwrap(x, width = 10, simplify = FALSE), paste, collapse="\n")) +
  # guides(x = guide_axis(angle = 90)) +
  xlab("Bigram") +
  ylab("Rating") +
  theme_minimal() +
  theme(legend.key.size = unit(0.5, 'cm')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, byrow=TRUE)) +
  # scale_color_discrete(name="Model", type=paletteer_d("RColorBrewer::Set1", n=8))
  scale_color_manual(name = "Rating Source", values = names_to_colors(c("Human", "Llama 3 70B Instruct", "Qwen 2 72B Instruct", "Llama 2 70B Chat", "Mixtral 7x8B Instruct", "Llama 3 8B Instruct", "Llama 2 13B Chat", "Llama 2 7B Chat"))) 
ggsave('plots/lm_isa_1sd_intelligence_bigrams.png', width=4, height=2.25, units='in')

human_dist_extra %>%
  filter(Noun == "intelligence") %>%
  longer_and_name_dist("Human") %>%
  mutate(NumRating = as.integer(Rating)) %>%
  ggplot(aes(x=NumRating, y=Probability)) +
  geom_col(aes(color="Human"), fill=magenta_color, alpha=0.1, 
         position="identity") +
  geom_col(data=human_dist_extra %>%
             filter(Noun == "intelligence") %>%
             longer_and_name_dist("Human") %>%
             mutate(NumRating = as.integer(Rating)) %>%
             mutate(Probability = -Probability) %>%
             select(!Model),
           aes(color="Human"), fill=magenta_color, alpha=0.1,
           position="identity") +
  scale_y_continuous(labels = abs, limits=c(-1,1)) +
  coord_flip() +
  theme_minimal() +
  facet_wrap(~ Bigram, labeller = labeller(Bigram = label_wrap_gen(12))) +
  scale_color_manual(name = "Rating Source", values = names_to_colors(c("Human", "Llama 3 70B Instruct", "Qwen 2 72B Instruct", "Llama 2 70B Chat", "Mixtral 7x8B Instruct", "Llama 3 8B Instruct", "Llama 2 13B Chat", "Llama 2 7B Chat"))) +
  geom_jitter(aes(color='Llama 3 70B Instruct'), 
              data = llama3i_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse) %>%
                mutate(Probability = 0), 
              shape=18, height=0.2, width=0.3, size=3) +
  geom_jitter(aes(color='Llama 3 8B Instruct'), 
              data = llama3i_8b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse) %>%
                mutate(Probability = 0), 
              shape=18, height=0.1, width=0.2, size=3) +
  geom_jitter(aes(color='Llama 2 70B Chat'), 
              data = llama2c_lscale_5shot_all_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse) %>%
                mutate(Probability = 0), 
              shape=18, height=0.2, width=0.3, size=3) +
  geom_jitter(aes(color='Llama 2 13B Chat'), 
              data = llama2c_13b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse) %>%
                mutate(Probability = 0), 
              shape=18, height=0.2, width=0.3, size=3) +
  geom_jitter(aes(color='Llama 2 7B Chat'), 
              data = llama2c_7b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse) %>%
                mutate(Probability = 0), 
              shape=18, height=0.2, width=0.3, size=3) +
  geom_jitter(aes(color='Mixtral 7x8B Instruct'), 
              data = mixtral_8x7b_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse) %>%
                mutate(Probability = 0), 
              shape=18, height=0.2, width=0.3, size=3) +
  geom_jitter(aes(color='Qwen 2 72B Instruct'), 
              data = qwen2i_lscale_5shot_responses %>%
                filter(Bigram %in% c("artificial intelligence", "fake intelligence")) %>%
                rename(NumRating = NumPredictedResponse) %>%
                mutate(Probability = 0), 
              shape=18, height=0.1, width=0.2, size=3) +
  # guides(x = guide_axis(angle = 90)) +
  labs(x = "Rating") +
  theme(legend.key.size = unit(0.5, 'cm')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, byrow=TRUE)) ->
  art_int_plot
ggsave('plots/lm_isa_1sd_intelligence_bigrams_splitbar.png', width=4, height=2.25, units='in')

art_int_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  )
ggsave('plots/lm_isa_1sd_intelligence_bigrams_splitbar_diss.png', width=4, height=2.25, units='in')


# Evaluate against human data ----

### Accuracies within 1 SD ----

#### Models ----

# Only first set of bigrams
llama2c_5shot_cot_responses %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_5shot_cot_acc
llama2c_5shot_cot_acc
# Overall 0.636, privative 0.535, subsective 0.74 
# With rounding to nearest integer: overall 0.764, privative 0.716, subsective 0.813

llama2c_5shot_cot_responses %>%
  filter(Adjective == "fake") %>%
  accuracy_1sd(isa_variance_12_combined) 

# Only first set of bigrams
llama3i_5shot_cot_responses %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama3i_5shot_cot_acc
llama3i_5shot_cot_acc

llama2c_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_lscale_5shot_acc
# Overall 0.725, privative 0.626, subsective 0.827 
# With rounding to nearest integer: overall 0.852, privative 0.806, subsective 0.9

llama2c_lscale_5shot_responses %>%
  filter(Adjective == "useful") %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined)

llama2c_nscale_5shot_responses %>%
  mutate(NumIsaRating = PredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_nscale_5shot_acc
# Overall 0.626, privative 0.671, subsective 0.58
# With rounding to nearest integer: overall 0.820, privative 0.852, subsective 0.787

llama2c_lscale_0shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_lscale_0shot_acc
# Overall 0.636, privative 0.368, subsective 0.913 
# With rounding to nearest integer: overall 0.813, privative 0.645, subsective 0.987

llama2c_nscale_0shot_responses %>%
  mutate(NumIsaRating = PredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_nscale_0shot_acc
# Overall 0.541, privative 0.581, subsective 0.5
# With rounding to nearest integer: overall 0.689, privative 0.794, subsective 0.58

llama3i_lscale_0shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama3i_lscale_0shot_all_acc
llama3i_lscale_0shot_all_acc



llama2c_lscale_5shot_all_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_lscale_5shot_all_acc
llama2c_lscale_5shot_all_acc

llama2c_7b_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_7b_lscale_5shot_all_acc
llama2c_7b_lscale_5shot_all_acc

llama2c_13b_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2c_13b_lscale_5shot_all_acc
llama2c_13b_lscale_5shot_all_acc

llama2_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2_lscale_5shot_all_acc
llama2_lscale_5shot_all_acc

llama2_7b_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2_7b_lscale_5shot_all_acc
llama2_7b_lscale_5shot_all_acc

llama2_13b_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama2_13b_lscale_5shot_all_acc
llama2_13b_lscale_5shot_all_acc

llama3i_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama3i_lscale_5shot_all_acc
llama3i_lscale_5shot_all_acc

llama3i_8b_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama3i_8b_lscale_5shot_all_acc
llama3i_8b_lscale_5shot_all_acc

llama3_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama3_lscale_5shot_all_acc
llama3_lscale_5shot_all_acc

llama3_8b_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  llama3_8b_lscale_5shot_all_acc
llama3_8b_lscale_5shot_all_acc

mixtral_8x7b_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  mixtral_8x7b_lscale_5shot_all_acc
mixtral_8x7b_lscale_5shot_all_acc

qwen2i_lscale_5shot_responses %>%
  mutate(NumIsaRating = NumPredictedResponse) %>%
  accuracy_1sd(isa_variance_12_combined) ->
  qwen2i_lscale_5shot_all_acc
qwen2i_lscale_5shot_all_acc

#### Humans vs. 1SD ----

human_mean_acc <- human_accuracy_1sd(isa_data_12_combined, isa_variance_12_combined)
human_mean_acc

#### Baselines ----

random_baseline_acc <- calculate_random_baseline(isa_variance_12_combined)
random_baseline_acc

# Random baseline: Overall 0.458, privative 0.607, subsective 0.324

# Baseline of just guessing 3 for privative and 5 for subsective
majority_baseline_acc <- calculate_majority_baseline(isa_variance_12_combined)
majority_baseline_acc

# Majority-per-class baseline: Overall accuracy 0.751, privative: 0.587, subsective: 0.92
# With rounding to nearest integer: overall 0.882, privative 0.774, subsective 0.993


# Analogy baseline
analogy_baseline_acc <- data.frame(AdjectiveClass = c('Privative', 'Subsective', 'High frequency', 'Zero frequency', 'Overall'),
                                   Accuracy = c(0.648, 0.431, 0.527, 0.567, 0.534))

#### Everything together (plot) ----

# bind_rows(llama2c_5shot_cot_acc %>%
#             mutate(Source = label.llama2c_5shot_cot),
#           llama2c_lscale_0shot_acc %>%
#             mutate(Source = label.llama2c_lscale_0shot),
#           llama2c_nscale_0shot_acc %>%
#             mutate(Source = label.llama2c_nscale_0shot),
#           llama2c_lscale_5shot_acc %>%
#             mutate(Source = label.llama2c_lscale_5shot),
#           llama2c_nscale_5shot_acc %>%
#             mutate(Source = label.llama2c_nscale_5shot),
#           human_mean_acc %>%
#             mutate(Source = label.human),
#           majority_baseline_acc %>%
#             mutate(Source = label.majority_baseline),
#           random_baseline_acc %>%
#             mutate(Source = label.random_baseline)
# ) %>%
#   mutate(AdjectiveClass = fct_relevel(AdjectiveClass, "Overall", "Privative", "Subsective",
#                                       "Privative Zero frequency", "Privative Low frequency", "Privative Medium/High frequency",
#                                       "Subsective Zero frequency", "Subsective Low frequency", "Subsective Medium/High frequency"),
#          Source = fct_relevel(Source, 
#                               label.human, label.majority_baseline,
#                               label.llama2c_5shot_cot,
#                               label.llama2c_lscale_5shot, label.llama2c_nscale_5shot,
#                               label.llama2c_lscale_0shot, label.llama2c_nscale_0shot,
#                               label.random_baseline)
#          ) -> all_accuracies

all_accuracies <- bind_rows(llama2c_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 2 Chat",
                                     Parameters = 70,
                                     Shots = 5),
                            llama2c_7b_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 2 Chat",
                                     Parameters = 7,
                                     Shots = 5),
                            llama2c_13b_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 2 Chat",
                                     Parameters = 13,
                                     Shots = 5),
                            llama3i_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 3 Instruct",
                                     Parameters = 70,
                                     Shots = 5),
                            llama3i_8b_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 3 Instruct",
                                     Parameters = 8,
                                     Shots = 5),
                            mixtral_8x7b_lscale_5shot_all_acc %>%
                              mutate(Model = "Mixtral Instruct",
                                     Parameters = 56,
                                     Shots = 5),
                            qwen2i_lscale_5shot_all_acc %>%
                              mutate(Model = "Qwen 2 Instruct",
                                     Parameters = 72,
                                     Shots = 5),
                            human_mean_acc %>%
                              mutate(Model = "Human",
                                     Parameters = 0,
                                     Shots = 0),
                            majority_baseline_acc %>%
                              mutate(Model = "Majority Baseline",
                                     Parameters = 0,
                                     Shots = 0),
                            random_baseline_acc %>% 
                              mutate(Model = "Random Baseline",
                                     Parameters = 0,
                                     Shots = 0),
                            analogy_baseline_acc %>% 
                              mutate(Model = "Analogy Baseline",
                                     Parameters = 0,
                                     Shots = 0)
) %>%
  mutate(Model = factor(Model)) %>%
  mutate(Class=fct_relevel(AdjectiveClass, "Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class = fct_recode(Class, "Privative adjective" = "Privative", "Subsective adjective" = "Subsective"))

all_accuracies_base <- bind_rows(llama2_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 2",
                                     Parameters = 70,
                                     Shots = 5),
                            llama2_7b_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 2",
                                     Parameters = 7,
                                     Shots = 5),
                            llama2_13b_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 2",
                                     Parameters = 13,
                                     Shots = 5),
                            llama3_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 3",
                                     Parameters = 70,
                                     Shots = 5),
                            llama3_8b_lscale_5shot_all_acc %>%
                              mutate(Model = "Llama 3",
                                     Parameters = 8,
                                     Shots = 5)
) %>%
  mutate(Model = factor(Model)) %>%
  mutate(Class=fct_relevel(AdjectiveClass, "Privative", "Subsective", "High frequency", "Zero frequency", "Overall")) %>%
  mutate(Class = fct_recode(Class, "Privative adjective" = "Privative", "Subsective adjective" = "Subsective"))

all_accuracies %>%
  filter(Model == "Human" & Class %in% c("Privative adjective", "Subsective adjective", "High frequency", "Zero frequency", "Overall")) ->
  human_1sd_acc_by_class

all_accuracies %>%
  filter(Model == "Majority Baseline" & Class %in% c("Privative adjective", "Subsective adjective", "High frequency", "Zero frequency", "Overall")) ->
  majority_1sd_acc_by_class

all_accuracies %>%
  filter(Model == "Random Baseline" & Class %in% c("Privative adjective", "Subsective adjective", "High frequency", "Zero frequency", "Overall")) ->
  random_1sd_acc_by_class

all_accuracies %>%
  filter(Model == "Analogy Baseline" & Class %in% c("Privative adjective", "Subsective adjective", "High frequency", "Zero frequency", "Overall")) ->
  analogy_1sd_acc_by_class

all_accuracies %>%
  filter(Shots == 5 & Class %in% c("Privative adjective", "Subsective adjective", "High frequency", "Zero frequency", "Overall")) %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=2) +
  geom_hline(data = human_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = majority_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = random_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = analogy_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  scale_linetype(name='Metric',
                 breaks=c('Accuracy', 'Accuracy within 1 SD')) +
  theme_minimal() +
  labs(color='Rating Source', x="Model Parameters (B)") + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        # legend.box.spacing = unit(0, 'pt')
        ) +
#  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "Random Baseline", "Majority Baseline", "Analogy Baseline"))) +
  
  facet_grid(~ Class) ->
  scaling_1sd_all_plot
ggsave('plots/1sd_accuracy_scaling_all.png', width=8, height=2.25, units='in')

scaling_1sd_all_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  facet_grid(~ Class, 
             labeller = labeller(Class = c("Privative adjective" = "Priv. adj.",
                                           "Subsective adjective" = "Subs. adj.",
                                           "High frequency" = "High freq.",
                                           "Zero frequency" = "Zero freq.",
                                           "Overall" = "Overall")))
ggsave('plots/1sd_accuracy_scaling_all_diss.png', width=6.5, height=2, units='in')


all_accuracies_base %>%
  filter(Shots == 5 & Class %in% c("Privative adjective", "Subsective adjective", "High frequency", "Zero frequency", "Overall")) %>%
  ggplot(aes(x=Parameters, y = Accuracy, color=Model)) +
  geom_line(aes(color=Model)) +
  geom_point(aes(color=Model), size=2) +
  geom_hline(data = human_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = majority_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  geom_hline(data = random_1sd_acc_by_class, 
             aes(yintercept=Accuracy, color=Model)) +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_continuous(breaks=seq(0.2,1,0.1)) +
  scale_linetype(name='Metric',
                 breaks=c('Accuracy', 'Accuracy within 1 SD')) +
  theme_minimal() +
  labs(color='Rating Source') + 
  theme(legend.position = "right",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
#        legend.box.spacing = unit(0, 'pt')
        ) +
#  guides(color = guide_legend(title.position="top", title.hjust = 0.1,nrow=2, byrow=TRUE)) + 
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3", "Qwen 2", "Llama 2", "Mixtral", "Random Baseline", "Majority Baseline", "Analogy Baseline"))) +
  facet_grid(~ Class) ->
  scaling_1sd_base_plot
scaling_1sd_base_plot
ggsave('plots/1sd_accuracy_scaling_base.png', width=8, height=2.25, units='in')

scaling_1sd_base_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  facet_grid(~ Class, 
             labeller = labeller(Class = c("Privative adjective" = "Priv. adj.",
                                           "Subsective adjective" = "Subs. adj.",
                                           "High frequency" = "High freq.",
                                           "Zero frequency" = "Zero freq.",
                                           "Overall" = "Overall")))
ggsave('plots/1sd_accuracy_scaling_base_diss.png', width=6.5, height=2, units='in')

### Ordinal regressions: Human vs. LM ----

llama2c_lscale_5shot_vs_human_lm <- clmm(IsaRating ~ HumanOrLM * AdjectiveClass + (1 | Adjective) + (1 | Noun) + (1 | Bigram), 
                data = human_lm_ratings %>%
                  filter(HumanOrLM %in% c(label.human, label.llama2c_lscale_5shot)),
                link = "logit")
summary(llama2c_lscale_5shot_vs_human_lm)
exp(llama2c_lscale_5shot_vs_human_lm$coefficients[5])

# Llama 2 Chat, labelled scale, 5-shot is significantly different from humans with an odds ratio of 1.31
# (i.e. roughly speaking, Llama 2 Chat's ratings are higher than human ones)
# The interaction with adjective class is also significant, indicating that subsective adjectives have a different
# effect on ratings for Llama 2 Chat than for humans


llama2c_nscale_5shot_vs_human_lm <- clmm(IsaRating ~ HumanOrLM * AdjectiveClass + (1 | Adjective) + (1 | Noun) + (1 | Bigram), 
                                         data = human_lm_ratings %>%
                                           filter(HumanOrLM %in% c(label.human, label.llama2c_nscale_5shot)),
                                         link = "logit")
summary(llama2c_nscale_5shot_vs_human_lm)
exp(llama2c_nscale_5shot_vs_human_lm$coefficients[5])

# Likewise for the numeric scale, 5-shot

### Regression: effect of frequency ----

llama2c_lscale_5shot_freq_lm <- clmm(IsaRating ~ DataSource + AdjectiveClass + Frequency +
                                       DataSource:AdjectiveClass + DataSource:Frequency +
                                       (1 | Adjective) + (1 | Noun) + (1 | Bigram), 
                                         data = human_lm_ratings %>%
                                           filter(HumanOrLM %in% c(label.human, label.llama2c_lscale_5shot)) %>%
                                       mutate(Frequency = case_when(
                                         CoarseFrequency == "Zero" ~ "Zero",
                                         CoarseFrequency %in% c("50th-75th percentile", "75th-90th percentile", "90th-99th percentile") ~ "High",
                                         .default = "Low"
                                       ),
                                       HumanOrLM = fct_recode(HumanOrLM, "Llama 2 Chat (5-shot)" = label.llama2c_lscale_5shot)) %>%
                                       mutate(Frequency = factor(Frequency, levels=c("High", "Low", "Zero")),
                                              DataSource = fct_drop(HumanOrLM)),
                                     link = "logit")
summary(llama2c_lscale_5shot_freq_lm)
# Significant effect of zero frequency bigrams for humans, which are rated lower
# No significant interaction between zero frequency and LM ratings - LM ratings
# of zero frequency bigrams are not significantly different from human ones, once the general 
# discrepancy between human and LM ratings has been accounted for.
# (Because there is a general discrepancy, post-hoc tests show a difference between zero
# frequency bigrams for human vs. LM)

pred_values <- ggpredict(llama2c_lscale_5shot_freq_lm, terms = c("DataSource", "AdjectiveClass"))
pred_values$x = fct_recode(pred_values$x, "Llama 2 Chat\n(5-shot)" = "Llama 2 Chat (5-shot)")

# Create a stacked bar plot using ggeffects
pred_values %>%
  ggplot(aes(x=x, y=predicted, fill=fct_rev(response.level))) +
  geom_col(position="fill") +
  labs(x="Data Source", y = "Rating (probability)", title="Effects plot for LLM vs. human predictions") +
  facet_wrap(~ group) +
  scale_fill_discrete(name="Rating", type = hcl.colors(5, palette="TealRose")) +
  theme_minimal() + 
  theme(text=element_text(size=36, color="#2C365E"),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank())
ggsave("plots/lm_isa_data_source_regression_effects_plot.png", width=14, height=7.5, units='in')


contrast(emmeans(llama2c_lscale_5shot_freq_lm, ~ HumanOrLM | CoarserFrequency), "pairwise")

llama2c_lscale_5shot_lm_only_freq_lm <- clmm(IsaRating ~ AdjectiveClass + Frequency +
                                       (1 | Adjective) + (1 | Noun) + (1 | Bigram), 
                                     data = human_lm_ratings %>%
                                       filter(HumanOrLM == label.llama2c_lscale_5shot) %>%
                                       mutate(Frequency = case_when(
                                         CoarseFrequency == "Zero" ~ "Zero",
                                         CoarseFrequency %in% c("50th-75th percentile", "75th-90th percentile", "90th-99th percentile") ~ "High",
                                         .default = "Low"
                                       )) %>%
                                       mutate(Frequency = factor(Frequency, levels=c("High", "Low", "Zero"))),
                                     link = "logit")
summary(llama2c_lscale_5shot_lm_only_freq_lm)

# Unfortunately we can't use ggeffects to get a stacked effects plot
gg_predicted = predict_response(llama2c_lscale_5shot_lm_only_freq_lm, 
                                terms = c("Frequency", "AdjectiveClass"))

gg_predicted %>%
  ggplot(aes(x=x, y=response.level)) +
  geom_bar(position="stack")


plot(gg_predicted,
     connect_lines=TRUE)

plot(allEffects(llama2c_lscale_5shot_lm_only_freq_lm), 
     style="stacked",
#     main="",
     ylab="Rating (probability)",
#     xlab=list(CoarserFrequency="Frequency"),
     key.args = list(space="right"),
     colors=rev(hcl.colors(5, palette="TealRose"))
)
dev.copy(png, "lm_isa_lmonly_regression_effects_plot.png", width=18, height=7, res=300, units="in")
dev.off()


llama2c_nscale_5shot_freq_lm <- clmm(IsaRating ~ HumanOrLM + AdjectiveClass + CoarserFrequency +
                                       HumanOrLM:AdjectiveClass + HumanOrLM:CoarserFrequency +
                                       (1 | Adjective) + (1 | Noun) + (1 | Bigram), 
                                     data = human_lm_ratings %>%
                                       filter(HumanOrLM %in% c(label.human, label.llama2c_nscale_5shot)) %>%
                                       mutate(CoarserFrequency = case_when(
                                         CoarseFrequency == "Zero" ~ "Zero",
                                         CoarseFrequency %in% c("50th-75th percentile", "75th-90th percentile", "90th-99th percentile") ~ "High",
                                         .default = "Low"
                                       )),
                                     link = "logit")
summary(llama2c_nscale_5shot_freq_lm)

# Distribution correlation / distance ----

## KL Divergence & JS Divergence ----

### Calculate ----

human_dist <- build_human_dist(isa_data_12_combined)
human_dist_extra <- build_human_dist(isa_data_combined)

llama3i_dist <- build_lm_dist(llama3i_lscale_5shot_responses)
llama3i_human_lm_divergences <- calculate_distribution_js(human_dist, llama3i_dist)

llama3i_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama3i_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))


llama3i_human_lm_divergences %>%
  group_by(CoarseFrequency) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2c_dist <- build_lm_dist(llama2c_lscale_5shot_all_responses)
llama2c_human_lm_divergences <- calculate_distribution_js(human_dist, llama2c_dist)

llama2c_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))
  
llama2c_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2c_7b_dist <- build_lm_dist(llama2c_7b_lscale_5shot_responses)
llama2c_7b_human_lm_divergences <- calculate_distribution_js(human_dist, llama2c_7b_dist)

llama2c_7b_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2c_7b_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2c_13b_dist <- build_lm_dist(llama2c_13b_lscale_5shot_responses)
llama2c_13b_human_lm_divergences <- calculate_distribution_js(human_dist, llama2c_13b_dist)

llama2c_13b_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2c_13b_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama3i_8b_dist <- build_lm_dist(llama3i_8b_lscale_5shot_responses)
llama3i_8b_human_lm_divergences <- calculate_distribution_js(human_dist, llama3i_8b_dist)

llama3i_8b_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama3i_8b_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2_dist <- build_lm_dist(llama2_lscale_5shot_responses)
llama2_human_lm_divergences <- calculate_distribution_js(human_dist, llama2_dist)

llama2_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2_7b_dist <- build_lm_dist(llama2_7b_lscale_5shot_responses)
llama2_7b_human_lm_divergences <- calculate_distribution_js(human_dist, llama2_7b_dist)

llama2_7b_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2_7b_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2_13b_dist <- build_lm_dist(llama2_13b_lscale_5shot_responses)
llama2_13b_human_lm_divergences <- calculate_distribution_js(human_dist, llama2_13b_dist)

llama2_13b_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama2_13b_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama3_dist <- build_lm_dist(llama3_lscale_5shot_responses)
llama3_human_lm_divergences <- calculate_distribution_js(human_dist, llama3_dist)

llama3_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama3_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama3_8b_dist <- build_lm_dist(llama3_8b_lscale_5shot_responses)
llama3_8b_human_lm_divergences <- calculate_distribution_js(human_dist, llama3_8b_dist)

llama3_8b_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

llama3_8b_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

mixtral_dist <- build_lm_dist(mixtral_lscale_5shot_responses)
mixtral_human_lm_divergences <- calculate_distribution_js(human_dist, mixtral_dist)

mixtral_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

mixtral_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

qwen_dist <- build_lm_dist(qwen2i_lscale_5shot_responses)
qwen_human_lm_divergences <- calculate_distribution_js(human_dist, qwen_dist)

qwen_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

qwen_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

random_human_lm_divergences <- calculate_uniform_js(isa_data_12_combined)

random_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

random_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

random_human_lm_divergences %>%
  group_by(CoarseFrequency) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

majority_human_lm_divergences <- calculate_majority_js(isa_data_12_combined)

majority_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

majority_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

majority_human_lm_divergences %>%
  group_by(CoarseFrequency) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

# Leave one participant out, average across
loo_human_lm_divergences <- calculate_human_loo_js(isa_data_12_combined)

loo_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

loo_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

loo_human_lm_divergences %>%
  group_by(CoarseFrequency) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

resampled_human_lm_divergences <- calculate_human_resampled_js(isa_data_12_combined)

resampled_human_lm_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

resampled_human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

resampled_human_lm_divergences %>%
  group_by(CoarseFrequency) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))


bind_rows(llama3i_human_lm_divergences %>%
            mutate(Method = "Llama 3 Instruct",
                   Parameters = 70),
          llama2c_human_lm_divergences %>%
            mutate(Method = "Llama 2 Chat",
                   Parameters = 70),
          qwen_human_lm_divergences %>%
            mutate(Method = "Qwen 2 Instruct",
                   Parameters = 72),
          mixtral_human_lm_divergences %>%
            mutate(Method = "Mixtral Instruct",
                   Parameters = 56),
          llama3i_8b_human_lm_divergences %>%
            mutate(Method = "Llama 3 Instruct",
                   Parameters = 8),
          llama2c_7b_human_lm_divergences %>%
            mutate(Method = "Llama 2 Chat",
                   Parameters = 7),
          llama2c_13b_human_lm_divergences %>%
            mutate(Method = "Llama 2 Chat",
                   Parameters = 13),
          random_human_lm_divergences %>%
            mutate(Method = "Uniform Baseline",
                   Parameters = 0),
          majority_human_lm_divergences %>%
            mutate(Method = "\"Majority\" Baseline",
                   Parameters = 0)
) %>%
  mutate(Method = factor(Method)) -> all_js_divergences

summarize_js_divergences(all_js_divergences) -> logprob_js_divergence_means

all_js_divergences %>%
  filter(Bigram == "homemade cat") %>% 
  select(Method, JSDivergence) %>% print()

### Plot ----

#### Violin plots ----

bind_rows(isa_data_12_combined %>%
            mutate(Method = "Human (no context)"),
          llama3i_lscale_5shot_sampled_responses %>%
            mutate(Method = "Llama 3 70B Instruct"),
          # llama2c_lscale_5shot_all_sampled_responses %>%
          #   mutate(Method = "Llama 2 70B Chat"),
          mixtral_lscale_5shot_all_sampled_responses %>%
            mutate(Method = "Mixtral 8x7B Instruct"),
          qwen2i_lscale_5shot_all_sampled_responses %>%
            mutate(Method = "Qwen 2 72B Instruct")
) %>%
  mutate(Model = factor(Method, levels=c("Human (no context)", 
                                          "Llama 3 70B Instruct", 
                                         "Mixtral 8x7B Instruct",
                                          "Qwen 2 72B Instruct",
                                          "Llama 2 70B Chat"))) -> 
  human_and_lm_sampled_12_ratings

human_and_lm_sampled_12_ratings %>%
  ggplot(aes(x=AdjectiveClass, y=NumRating, fill=Model)) +
  geom_violin(adjust=3, position="dodge") +
  labs(x = "Adjective Type",
       y = "Rating") +
  theme_minimal() +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=2, byrow=TRUE)) +
  scale_fill_discrete(type = c("#D81B60", "#1E88E5", "#FFC107", "#004D40", "#81ba31", "#5a18dd"))
ggsave('plots/plots/rating_distribution_by_llm.png', width=4, height=3.25, units='in')


#### Distribution plots ----

split_bar_plot(list(human_dist, llama3i_dist, qwen_dist, llama2c_dist, mixtral_dist),
               names_to_colors(c("Human (context not given)", 
                 "Llama 3 70B Ins.", 
                 "Qwen 2 72B Ins.",
                 "Llama 2 70B Ch.",
                 "Mixtral 8x7B Ins."
                 )),
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=2, byrow=TRUE))
ggsave('plots/logprob_rating_distribution_by_llm_splitbar.png', width=5, height=3.25, units='in')

split_bar_plot(list(human_dist, llama3i_dist, qwen_dist, llama2c_dist),
               names_to_colors(c("Human (context not given)", 
                 "Llama 3 70B Instruct", 
                 "Qwen 2 72B Instruct",
                 "Llama 2 70B Chat"
               )),
               vertical = TRUE) +
  theme(legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt'),
        text=element_text(size=15),
        strip.text.x = element_blank()
        ) ->
  logprob_llm_splitbar_plot
logprob_llm_splitbar_plot +
  theme(legend.position = "bottom") +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=2, byrow=TRUE))
ggsave('plots/logprob_rating_distribution_by_3llm_splitbar.png', width=5, height=3.5, units='in')

logprob_llm_splitbar_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10),
        legend.position = "right"
  )
ggsave('plots/logprob_rating_distribution_by_3llm_splitbar_diss.png', width=6.5, height=2.5, units='in')


split_bar_plot(list(human_dist, llama3_dist, llama3_8b_dist, llama2_dist, llama2_13b_dist, llama2_7b_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Llama 3 70B", 
                                 "Llama 3 8B",
                                 "Llama 2 70B",
                                 "Llama 2 13B",
                                 "Llama 2 7B"
               )),
               vertical = TRUE) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=2, byrow=TRUE)) ->
  logprob_by_llm_plot
logprob_by_llm_plot
ggsave('plots/logprob_rating_distribution_by_llm_base_splitbar.png', width=8, height=3.25, units='in')

logprob_by_llm_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) 
ggsave('plots/logprob_rating_distribution_by_llm_base_splitbar_diss.png', width=6.5, height=3.5, units='in')


split_bar_plot(list(human_dist, llama3i_dist, qwen_dist, llama2c_dist, mixtral_dist, 
                    llama3i_8b_dist, llama2c_13b_dist, llama2c_7b_dist),
               names_to_colors(c("Human (context not given)", 
                 "Llama 3 70B Ins.", 
                 "Qwen 2 72B Ins.",
                 "Llama 2 70B Ch.",
                 "Mixtral 8x7B Ins.",
                 "Llama 3 8B Ins.",
                 "Llama 2 13B Ch.",
                 "Llama 2 8B Ch."
               )),
               bigrams = c("homemade cat", "illegal currency"), facet_by = "Bigram") +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(fill = guide_legend(title.position="top", title.hjust = 0.5, nrow=2, byrow=TRUE))

split_bar_plot(list(human_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation",
                                 "Log-probability"))[c(1, 3)],
               human_name = "Human (context not given)",
               facet_by = "Bigram",
               bigrams=spb_bigrams2,
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
ggsave('plots/rating_distribution_logprob_only_by_bigram2_splitbar_llama3i.png', width=14, height=3.5, units='in')

split_bar_plot(list(human_dist, llama3i_dist),
               names_to_colors(c("Human (context not given)", 
                                 "Context generation",
                                 "Log-probability"))[c(1, 3)],
               human_name = "Human (context not given)",
               facet_by = "Bigram",
               bigrams=c("counterfeit watch", "homemade cat"),
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
ggsave('plots/rating_distribution_logprob_only_by_bigram_two_splitbar_llama3i.png', width=5, height=3.5, units='in')


#### JS divergence plots ----

logprob_js_divergence_means %>%
  filter(Parameters != 0) %>%
  ggplot(aes(x=Parameters, y=Mean, color=Method)) +
  geom_point() +
  geom_line() +
  geom_hline(aes(yintercept = 0, color="Human")) +
  geom_hline(data = logprob_js_divergence_means %>%
               filter(Method == "Uniform Baseline"), aes(yintercept=Mean, color=Method)) +
  geom_hline(data = logprob_js_divergence_means %>%
               filter(Method == "\"Majority\" Baseline"), aes(yintercept=Mean, color=Method)) +
  facet_wrap(~ AdjectiveClass) +
  labs(x = "Model Parameters (B)", y = "JS Divergence", color="Rating Source") +
  theme_minimal() +
  scale_x_continuous(breaks=seq(0,75,15)) +
  scale_y_reverse() +
  expand_limits(y = c(0, 1)) +
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "\"Majority\" Baseline", "Uniform Baseline")))
ggsave('plots/logprob_jsdivergence_scaling.png', width=6, height=2.25, units='in')

logprob_js_divergence_means %>%
  filter(Parameters != 0) %>%
  ggplot(aes(x=Parameters, y=1 - Mean, color=Method)) +
  geom_point() +
  geom_line() +
  geom_hline(aes(yintercept = 1 - 0, color="Human")) +
  geom_hline(data = logprob_js_divergence_means %>%
               filter(Method == "Uniform Baseline"), aes(yintercept=1 - Mean, color=Method)) +
  geom_hline(data = logprob_js_divergence_means %>%
               filter(Method == "\"Majority\" Baseline"), aes(yintercept=1 - Mean, color=Method)) +
  facet_wrap(~ AdjectiveClass) +
  labs(x = "Model Parameters (B)", y = "1 - JS Divergence", color="Rating Source") +
  theme_minimal() +
  scale_x_continuous(breaks=seq(0,75,15)) +
  #  scale_y_reverse() +
  expand_limits(y = c(0, 1)) +
  scale_color_manual(values = names_to_colors(c("Human", "Llama 3 Instruct", "Qwen 2 Instruct", "Llama 2 Chat", "Mixtral Instruct", "\"Majority\" Baseline", "Uniform Baseline")))
ggsave('plots/logprob_jsdivergence_scaling_flipped.png', width=6, height=2.25, units='in')


#### Original plots ----

human_lm_divergences %>%
  ggplot(aes(x=reorder_within(x=Bigram, by=KLDivergence,
                              within=AdjectiveClass), 
             y=KLDivergence, 
             color=CoarseFrequency)) +
  geom_point() +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ AdjectiveClass, scale="free_x") +
  labs(x="Bigram")

human_lm_divergences %>%
  ggplot(aes(x=reorder_within(x=Bigram, by=JSDivergence,
                              within=AdjectiveClass), 
             y=JSDivergence, 
             color=CoarseFrequency)) +
  geom_point() +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ AdjectiveClass, scale="free_x") +
  labs(x="Bigram")

human_lm_divergences %>%
  ggplot(aes(x=reorder_within(x=Bigram, by=TVDistance,
                              within=AdjectiveClass), 
             y=TVDistance, 
             color=CoarseFrequency)) +
  geom_point() +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ AdjectiveClass, scale="free_x") +
  labs(x="Bigram")

human_lm_divergences %>%
  ggplot(aes(x=reorder_within(x=Noun, by=JSDivergence,
                              within=Adjective), 
             y=JSDivergence, 
             color=CoarseFrequency)) +
  geom_point() +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ Adjective, scale="free_x") +
  labs(x="Noun")

human_lm_divergences %>%
  ggplot(aes(x=reorder_within(x=Noun, by=TVDistance,
                              within=Adjective), 
             y=TVDistance, 
             color=CoarseFrequency)) +
  geom_point() +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ Adjective, scale="free_x") +
  labs(x="Noun")

human_lm_divergences %>%
  ggplot(aes(x=CoarseFrequency, 
             y=JSDivergence, 
             color=AdjectiveClass)) +
  geom_boxplot(notch = TRUE) +
  guides(x = guide_axis(angle = 90))
ggsave('plots/lm_isa_jsdivergence_boxplot.png')

human_lm_divergences %>%
  ggplot(aes(x=JSDivergence,y=KLDivergence)) +
  geom_point()

human_lm_divergences %>%
  ggplot(aes(x=JSDivergence, y=TVDistance, color=AdjectiveClass)) +
  geom_point()

human_lm_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(MeanJS = mean(JSDivergence), JS.SD = sd(JSDivergence), JS.SE = JS.SD / sum(JSDivergence)) -> mean_js
mean_js %>%
  ggplot(aes(x=AdjectiveClass, y=MeanJS)) +
  geom_point() +
  geom_errorbar(aes(ymin = MeanJS - 1.96 * JS.SE, ymax = MeanJS + 1.96 * JS.SE), 
                width = 0.2)

human_lm_divergences %>%
  group_by(AdjectiveClass, CoarseFrequency) %>%
  summarize(MeanJS = mean(JSDivergence), JS.SD = sd(JSDivergence), JS.SE = JS.SD / sum(JSDivergence)) %>%
  ungroup() %>%
  unite(Condition, AdjectiveClass, CoarseFrequency) %>%
  # TODO sort Condition so that zero comes before 25-50
  ggplot(aes(x=Condition, y=MeanJS)) +
  geom_point() +
  geom_errorbar(aes(ymin = MeanJS - 1.96 * JS.SE, ymax = MeanJS + 1.96 * JS.SE), 
                width = 0.2) +
  guides(x = guide_axis(angle = 90)) 
  
  
### Fit model ----

js_lm <- lmer(JSDivergence ~ AdjectiveClass * CoarseFrequency + (1 | Adjective) + (1 | Noun),
            data = human_lm_divergences)
summary(js_lm)

# Nothing is significant except the intercept and the interaction between subsective 
# and privative for the 90th-99th percentile


## Total variation distance ----

## Spearman's Correlation ----

spearman_statistics = function(bigram_ratings, ...) {
  sp.test = cor.test(bigram_ratings$Human, bigram_ratings$Llama, method="spearman", exact=FALSE)
  return(tibble(AdjectiveClass = bigram_ratings$AdjectiveClass, 
                CoarseFrequency = bigram_ratings$CoarseFrequency,
                p.value = sp.test$p.value, S = sp.test$statistic, rho = sp.test$estimate))
}

# Count how often each rating occurs in humans vs. LLMs and see if this is correlated per bigram
human_lm_ratings %>%
  filter(HumanOrLM %in% c(label.human, label.llama2c_lscale_5shot)) %>%
  mutate(HumanOrLM = fct_recode(fct_drop(HumanOrLM), Llama=label.llama2c_lscale_5shot)) %>%
  group_by(HumanOrLM, Bigram, IsaRating, .drop=FALSE) %>%
  summarize(n = dplyr::n(), .groups = "drop_last") %>%
  # Fix adjective classes
  merge(human_lm_ratings %>% select(Bigram, AdjectiveClass, CoarseFrequency) %>% distinct(), .by = "Bigram") %>%
  pivot_wider(names_from=HumanOrLM, values_from=n) %>%
  group_by(Bigram) %>%
  group_modify(spearman_statistics) ->
  llama2c_lscale_5shot_spearman_by_bigram


# Distributions (counts of ratings) are only correlated in 11% of bigrams
llama2c_lscale_5shot_spearman_by_bigram %>%
  group_by(p.value < 0.05) %>%
  summarize(Count = n()) %>%
  mutate(Percent = Count / sum(Count))

llama2c_lscale_5shot_spearman_by_bigram %>%
  group_by(AdjectiveClass, p.value < 0.05) %>%
  summarize(Count = n()) %>%
  mutate(Percent = Count / sum(Count))

llama2c_lscale_5shot_spearman_by_bigram %>%
  mutate(CoarserFrequency = case_when(
    CoarseFrequency == "Zero" ~ "Zero",
    CoarseFrequency %in% c("50th-75th percentile", "75th-90th percentile", "90th-99th percentile") ~ "High",
    .default = "Low"
  )) %>%
  group_by(CoarserFrequency, p.value < 0.05) %>%
  summarize(Count = n()) %>%
  mutate(Percent = Count / sum(Count))



