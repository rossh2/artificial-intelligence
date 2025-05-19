library(tidyverse)
library(ggeffects)
library(lme4)
library(glmmTMB)

source("Analysis_Utils.R")
source("LM_Analysis_Utils.R")

# Preprocess data ----

## Import data ----

analogy_prompting_wide <- read.csv("results/human/exp8_analogy/analogy_prompting_responses_raw.csv",
                      fileEncoding = "UTF-8-BOM")

colnames(analogy_prompting_wide)

analogy_prompting_wide <- preprocess_qualtrics_wide(analogy_prompting_wide)

## Exclusion criteria ----

### Demographic ----

dem_excluded_ids <- analogy_prompting_wide %>%
  filter(EnglishBefore5!="Yes" | !(Dialect=="Yes" | Dialect_2_TEXT %in% c("American English", "American", "american", "Appalachian dialect"))) %>% 
  pull(UserId)

dem_excluded_ids

### Attention checks ----

analogy_prompting_wide <- analogy_prompting_wide %>%
  mutate(AttnFailed = ifelse(Q.stone.strawb.ana %in% c("Definitely yes", "Probably yes"), 1, 0) +
           ifelse(Q.stone.strawb.memo %in% c("Definitely yes", "Probably yes"), 1, 0) +
           ifelse(Q.toy.mammoth.ana %in% c("Definitely yes", "Probably yes"), 1, 0) +
           ifelse(Q.toy.mammoth.memo %in% c("Definitely yes", "Probably yes"), 1, 0) +
           ifelse(Q.orange.fish.ana %in% c("Definitely not", "Probably not"), 1, 0) +
           ifelse(Q.orange.fish.memo %in% c("Definitely not", "Probably not"), 1, 0)
  )

attn_excluded_ids <- analogy_prompting_wide %>%
  filter(AttnFailed >= 2) %>% 
  pull(UserId)

attn_excluded_ids

# Participant 103 does not type adjective-noun phrases in the analogy box
# Participant 101 literally types "adjective" or "noun" in the box
# Participants 43, 9 and 83 type the original bigram into the box, or just the original noun
# Participant 81 tends to use an analogy to the opposite noun (e.g. illegal -> legal) and/or only types an adjective
# Participants 3, 87, 8, 35, 53, 92, 150 type their reasoning into the box
# Participant 146 seems to want to paraphrase nouns with noun-noun compounds (e.g. counterfeit timepiece gadget)
# Participant 17 mostly types single words and thinks that "fruit smoothie" is a good analogy for "artificial image"
# Participant 124 types "still X" into the analogy field
analogy_excluded_ids <- c(101, 103, 43, 83, 9, 81, 3, 87, 8, 35, 53, 92, 150, 146, 17, 124)

# Accidentally gathered 13 of first list, exclude the last one as if never gathered
extra_excluded_ids <- c(175)

### Exclusion ----

excluded_ids <- analogy_prompting_wide %>%
  filter(UserId %in% dem_excluded_ids | UserId %in% attn_excluded_ids) %>%
  select(EnglishBefore5, Dialect, Dialect_2_TEXT, OtherLanguages, 
         Q.stone.strawb.ana, Q.stone.strawb.memo, Q.toy.mammoth.ana, Q.toy.mammoth.memo,
         Q.orange.fish.ana, Q.orange.fish.memo)

analogy_prompting_wide_excl <- analogy_prompting_wide %>%
  filter(!UserId %in% dem_excluded_ids & !UserId %in% attn_excluded_ids & !UserId %in% analogy_excluded_ids & !UserId %in% extra_excluded_ids)

nrow(analogy_prompting_wide_excl)

## Select target questions and pivot ----

names(analogy_prompting_wide_excl)

analogy_prompting_long <- analogy_prompting_wide_excl %>%
  dplyr::select(!EnglishBefore5:Comments) %>% # Exclude demographics
  dplyr::select(!starts_with("Q.melted.plastic") & !starts_with("Q.hardboiled.egg") & !starts_with("Q.decorative.pumpkin")) %>%  # Exclude training
#  dplyr::select(!c("Q.knitted.pizza", "Q.temporary.breakage", "Q.orange.mouse", "Q.wooden.pear")) %>%  # Exclude fillers
  dplyr::select(!c("AttnFailed")) %>%
  pivot_longer(
    cols = Q.false.gentleman.yn:Q.multicolored.image.memo,
    names_to = c("Adjective", "Noun", "QuestionType"),
    names_pattern = "Q.([a-z]+).([a-z]+).([a-zA-Z_0-9]+)",
    values_to = "RatingOrAnalogy"
  ) %>%
  mutate(Noun = ifelse(Noun == "strawb", "strawberry", Noun)) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate(across(c("UserId", "Adjective", "Noun", "Bigram", "QuestionType"), factor)) %>%
  filter(RatingOrAnalogy!="") 

str(analogy_prompting_long)

analogy_prompting_all <- analogy_prompting_long %>%
  pivot_wider(names_from = QuestionType, values_from = RatingOrAnalogy) %>%
  rename(AnalogyAvailable = "yn", AnalogyText = "yn_2_TEXT") %>%
  mutate(Rating = ifelse(!is.na(ana), ana, memo)) %>%
  select(!c("ana", "memo")) %>%
  mutate(Rating = factor(Rating, levels = c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"))) %>%
  mutate(NumRating = as.integer(Rating)) %>%
#  mutate(AnalogyAvailable = fct_recode(factor(AnalogyAvailable, levels = c("No", "Yes, the phrase I'm thinking of is:")),
#                                       Yes = "Yes, the phrase I'm thinking of is:")) %>%
  mutate(AnalogyAvailable = ifelse(AnalogyAvailable == "No", 0, 1))
  

str(analogy_prompting_all)

analogy_prompting <- analogy_prompting_all %>%
  filter(!(Adjective %in% c("orange", "stone", "toy", "knitted")))

str(analogy_prompting)

analogy_prompting_bigrams <- analogy_prompting %>% distinct(Bigram) %>% pull(Bigram)

## Preprocess analogy text ----

### Strip articles and remove obvious not-analogies ----

analogy_prompting %>%
  mutate(OriginalAnalogyText = AnalogyText) %>%
  mutate(AnalogyText = trimws(tolower(AnalogyText))) %>%
  mutate(AnalogyText = str_replace(AnalogyText, "( |^)(yes|an|a|the) ", "\\1")) %>% 
  mutate(AnalogyText = str_replace(AnalogyText, "sign post", "signpost")) %>%
  mutate(AnalogyText = str_replace(AnalogyText, "knock off", "knockoff")) %>%
  mutate(AnalogyText = ifelse(AnalogyText %in% c("yes", "true"), NA, AnalogyText),) %>%
  mutate(AnalogyText = ifelse(grepl("( |^)(still|just) ", AnalogyText) | 
                                grepl("( |^)(is|are|was|may|need) ", AnalogyText) | 
                                grepl(" (and|or) ", AnalogyText), 
                              NA, AnalogyText)) %>%
  mutate(AnalogyText = ifelse(AnalogyText == Bigram, NA, AnalogyText)) %>%
  mutate(AnalogyAvailable = ifelse(is.na(AnalogyText), 0, AnalogyAvailable)) ->
  analogy_prompting_cleaned

# analogy_prompting %>%
#   filter(grepl("still", AnalogyText)) %>%
#   View()
# 
# analogy_prompting %>%
#   filter(grepl(" is ", AnalogyText) | grepl(" are ", AnalogyText)) %>%
#   View()

### Parse out adjective and noun ----

split_analogy_text <- function(analogy_text, original_noun) {
  one.word <- FALSE
  if (is.na(analogy_text)) {
    modifier <- NA
    noun <- NA
    one.word <- NA
  } else {
    words <- str_split_1(analogy_text, boundary("word"))
    if (length(words) == 2) {
      modifier <- words[[1]]
      noun <- words[[2]]
    } else if (length(words) == 1) {
      one.word <- TRUE
      if (words[[1]] %in% c("authentic", "unuseful", "large")) {
        modifier <- words[[1]]
        noun <- NA
      } else {
        modifier <- NA
        noun <- words[[1]]
      }
    } else if (length(words) == 3) {
      if (words[[3]] == original_noun) {
        modifier <- paste(words[[1]], words[[2]])
        noun <- words[[3]]
      } else {
        if (words[[2]] == "for" | words[[2]] == "of") {
          # e.g. "bed for mouse", "statement of words" - convert to noun-noun compound
          modifier <- words[[3]]
          noun <- words[[1]]
        } else {
          # Most 3-word items involve a noun-noun compound, use that as heuristic
          modifier <- words[[1]]
          noun <- paste(words[[2]], words[[3]])
        }
      }
    } else {
      # Shouldn't occur as input field was limited to 1-3 words
      modifier <- NA
      noun <- NA
    }
  }
  mini_df <- as.data.frame(list(AnalogyMod=modifier, AnalogyNoun=noun, OneWordAnalogy=one.word))
  return(mini_df)
}

map2(analogy_prompting_cleaned$AnalogyText, analogy_prompting_cleaned$Noun, split_analogy_text) %>% 
  list_rbind() %>% 
  bind_cols(analogy_prompting_cleaned, .) %>% 
  mutate(AnalogyAvailable = ifelse(AnalogyAvailable & AnalogyNoun == Noun & is.na(AnalogyMod), 0, AnalogyAvailable),
         AnalogyNoun = ifelse(AnalogyNoun == Noun & is.na(AnalogyMod), NA, AnalogyNoun)) %>%
  mutate(SameAdjectiveAnalogy = (AnalogyMod == Adjective),
         SameNounAnalogy = (AnalogyNoun == Noun)
         ) -> analogy_prompting_cleaned

## Add frequencies ----

noun_difficulties <- read.csv("../Adjectives-PythonCode/output/analogy/bigrams_with_4way_analogy_difficulty_4neighbours.csv")
noun_difficulties %>%
  select(Bigram, Mean, SD, Convergent, DivergentSimilarBigrams, ConflictingConvergentSimilarBigrams, 
         HighFNeighbourCount,  HighFNeighbourQuartile, WordnetDepth, BNCNounCount, BNCNounQuartile) %>%
  mutate(across(c("Bigram", "HighFNeighbourQuartile", "BNCNounQuartile"), as.factor)) %>%
  mutate(across(c("Convergent", "ConflictingConvergentSimilarBigrams"), (function(text) text == "True"))) %>%
  rename(DivergentSimilarBigramCount = DivergentSimilarBigrams) %>%
  mutate(DivergentSimilarBigrams = DivergentSimilarBigramCount > 0) ->
  noun_difficulties

str(noun_difficulties)

analogy_prompting_cleaned %>%
  add_frequency() %>%
  merge(noun_difficulties, by = 'Bigram', all.x=TRUE) ->
  analogy_prompting_cleaned

# 3 rows have a NA rating, all have analogy not available (all different user IDs)
analogy_prompting_cleaned %>%
  filter(!is.na(NumRating)) ->
  analogy_prompting_cleaned

str(analogy_prompting_cleaned)

write.csv(analogy_prompting_cleaned, file = "analogy_prompting_responses_cleaned.csv", row.names = FALSE)

## Check fillers / attention checks ----

# Run this before setting threshold on attention check exclusion above

analogy_prompting_all %>%
  filter(Bigram %in% (c("orange fish", "stone strawberry", 
                        "toy mammoth", "knitted pizza"))) %>%
  ggplot(aes(x=Rating)) +
  geom_bar(stat="count") +
  facet_wrap(~ Bigram) 

## Check missing data for Latin square ----

anp_first_list_bigrams = c("false gentleman", "knockoff bus", "fake reef", 
                       "knockoff painting", "artificial rumor", "counterfeit signpost", 
                       "artificial lifestyle", "counterfeit jacket", "fake signpost", 
                       "former allegation", "false money", "former reason")
analogy_prompting %>%
  # First bigram of each list
  filter(Bigram %in% anp_first_list_bigrams) %>%
  mutate(Bigram = fct_relevel(Bigram, anp_first_list_bigrams)) %>%
  group_by(Bigram) %>%
  summarize(n()) %>%   
  arrange(Bigram)

144 - (nrow(analogy_prompting %>% distinct(UserId)))


# Plots of analogy availability ----

## Overall ----

table(analogy_prompting_cleaned[, c("AnalogyAvailable")])

table(analogy_prompting_cleaned[, c("AdjectiveClass", "AnalogyAvailable")])

analogy_prompting_cleaned %>%
  group_by(Bigram) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), CoarseFrequency = first(CoarseFrequency),
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  filter(AnalogyPercent < 0.1)

analogy_prompting_cleaned %>%
  group_by(Bigram) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), CoarseFrequency = first(CoarseFrequency),
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  ggplot(aes(x=Noun, y=AnalogyPercent, fill=CoarseFrequency)) +
  facet_wrap(~ Adjective, scales = "free_x", ncol = 6) +
  geom_col() +
  guides(x = guide_axis(angle = 45)) +
  theme_minimal() + 
  labs(y="Analogy Available (Percentage)", fill="Bigram Frequency")
ggsave("plots/analogy_prompting_analogy_percent_by_bigram_with_bigramfreq.png",
       width=10, height=5, units='in')

analogy_prompting_cleaned %>%
  group_by(Bigram) %>%
  mutate(ZeroFrequency = ifelse(CoarseFrequency == "Zero", "Zero frequency", "High frequency")) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), ZeroFrequency = first(ZeroFrequency),
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  ggplot(aes(x=Noun, y=AnalogyPercent, fill=ZeroFrequency)) +
  facet_wrap(~ Adjective, scales = "free_x", ncol = 6) +
  geom_col() +
  guides(x = guide_axis(angle = 45)) +
  theme_minimal() + 
  labs(y="Analogy Available (Percentage)", fill="Bigram Frequency")
ggsave("plots/analogy_prompting_analogy_percent_by_bigram_with_zerofreq.png",
       width=10, height=5, units='in')

analogy_prompting_cleaned %>%
  group_by(Bigram) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), BNCNounQuartile = first(BNCNounQuartile),
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  ggplot(aes(x=Noun, y=AnalogyPercent, fill=BNCNounQuartile)) +
  facet_wrap(~ Adjective, scales = "free_x") +
  geom_col() +
  guides(x = guide_axis(angle = 45)) +
  theme_minimal()

analogy_prompting_cleaned %>%
  group_by(Bigram) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), 
            DivergentSimilarBigrams = first(DivergentSimilarBigrams),
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  ggplot(aes(x=Noun, y=AnalogyPercent, fill=DivergentSimilarBigrams)) +
  facet_wrap(~ Adjective, scales = "free_x", ncol=6) +
  geom_col() +
  guides(x = guide_axis(angle = 45)) +
  theme_minimal()
ggsave("plots/analogy_prompting_analogy_percent_by_bigram_with_div.png",
       width=10, height=5, units='in')

analogy_prompting_cleaned %>%
  group_by(Bigram) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), 
            WordnetDepth = first(WordnetDepth),
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  ggplot(aes(x=Noun, y=AnalogyPercent, fill=WordnetDepth)) +
  facet_wrap(~ Adjective, scales = "free_x", ncol=6) +
  geom_col() +
  guides(x = guide_axis(angle = 45)) +
  theme_minimal() +
  labs(y="Analogy Available (Percentage)", fill="Specificity of Noun\n(Depth in WordNet)")
ggsave("plots/analogy_prompting_analogy_percent_by_bigram_with_wordnetdepth.png",
       width=10, height=5, units='in')

analogy_prompting_cleaned %>%
  mutate(AnalogyPossiblyHard = if_else(is.na(WordnetDepth), "No", if_else(HighFNeighbourQuartile %in% c('1st quartile', '2nd quartile') | 
           ConflictingConvergentSimilarBigrams | DivergentSimilarBigrams, "Yes", "No"))) %>%
  group_by(Bigram) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), AnalogyPossiblyHard = first(AnalogyPossiblyHard),
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  ggplot(aes(x=Noun, y=AnalogyPercent, fill=AnalogyPossiblyHard)) +
  facet_wrap(~ Adjective, scales = "free_x", ncol = 6) +
  geom_col() +
  guides(x = guide_axis(angle = 45)) +
  theme_minimal() + 
  labs(y="Analogy Available (Percentage)", fill="Analogy Possibly Difficult?") +
  theme(legend.position = 'bottom') +
  scale_fill_manual(values=c("No"=light_blue_color, "Yes"=magenta_color)) ->
  ap_by_pct_w_diff_plot 
ap_by_pct_w_diff_plot
ggsave("plots/analogy_prompting_analogy_percent_by_bigram_with_difficulty.png",
       width=10, height=5, units='in')

ap_by_pct_w_diff_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  guides(x = guide_axis(angle = 90)) 
ggsave("plots/analogy_prompting_analogy_percent_by_bigram_with_difficulty_diss.png",
       width=6.5, height=4.25, units='in')

## Participants who never found analogies (in fillers or target) ----

analogy_prompting_all %>%
  group_by(UserId) %>%
  summarize(AnalogyCount = sum(AnalogyAvailable)) %>%
  group_by(AnalogyCount == 0) %>%
  summarise(n())

## Analogy availability for zero-frequency bigrams ----

analogy_prompting_cleaned %>%
  filter(CoarseFrequency == "Zero") %>%
  group_by(Bigram) %>%
  summarize(Adjective = first(Adjective), Noun = first(Noun), 
            AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  ggplot(aes(x=Noun, y=AnalogyPercent)) +
  facet_wrap(~ Adjective, scales = "free_x", ncol=6) +
  geom_col() +
  guides(x = guide_axis(angle = 45)) +
  theme_minimal() +
  labs(y="Analogy Available (Percentage)", title="Analogy availability for zero frequency bigrams")

analogy_prompting_cleaned %>%
  filter(CoarseFrequency == "Zero") %>%
  distinct(Bigram) %>%
  nrow()
# 35 zero-frequency bigrams in the dataset

analogy_prompting_cleaned %>%
  filter(CoarseFrequency == "Zero") %>%
  group_by(Bigram) %>% 
  summarize(AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  filter(AnalogyPercent < 0.5) %>%
  arrange(AnalogyPercent) %>%
  print()
# 10 zero-frequency bigrams with less than 50% analogy availability
# 1 bigram (homemade currency) with < 25% analogy availability

analogy_prompting_cleaned %>%
  group_by(Bigram) %>% 
  summarize(AnalogyPercent = sum(AnalogyAvailable) / n()) %>%
  filter(AnalogyPercent < 0.5) %>%
  arrange(AnalogyPercent) %>%
  print(n=Inf)

# Statistics of types of analogy ----

## Counts ----

analogy_prompting_cleaned %>%
  group_by(AnalogyAvailable == TRUE) %>%
  summarize(Count = n(), Percent = Count/nrow(.))

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  group_by(OneWordAnalogy) %>%
  summarize(Count = n(), Percent = Count/nrow(.))

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE & (is.na(AnalogyMod) | is.na(AnalogyNoun))) %>%
  select(UserId, Bigram, OriginalAnalogyText, AnalogyMod, AnalogyNoun) %>%
  View()

analogy_prompting_cleaned %>%
  filter(UserId %in% (
    analogy_prompting_cleaned %>%
      filter(grepl(" is ", OriginalAnalogyText)) %>%
      distinct(UserId) %>% pull(UserId)
  )) %>%
  select(UserId, Bigram, OriginalAnalogyText, AnalogyMod, AnalogyNoun) %>%
  arrange(UserId) %>%
  View()


analogy_prompting_cleaned %>%
  filter(UserId %in% c(87)) %>%
  select(UserId, Bigram, OriginalAnalogyText, AnalogyMod, AnalogyNoun) %>%
  View()

analogy_prompting_cleaned %>%
  filter(AnalogyMod == "still") %>%
  select(UserId, Bigram, OriginalAnalogyText, AnalogyMod, AnalogyNoun) %>%
  View()

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  group_by(SameAdjectiveAnalogy) %>%
  summarize(Count = n(), Percent = Count/nrow(.))

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  group_by(SameNounAnalogy) %>%
  summarize(Count = n(), Percent = Count/nrow(.))

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  group_by(Adjective, SameNounAnalogy) %>%
  summarize(Count = n(), Percent = Count/nrow(.)) %>%
  filter(SameNounAnalogy) %>%
  arrange(desc(Percent))

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE & SameNounAnalogy) %>%
  group_by(AnalogyMod) %>%
  summarize(Count = n(), Percent = Count/nrow(.)) %>%
  arrange(desc(Percent)) %>%
  slice_head(n=10)

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  group_by(AnalogyText) %>%
  summarize(Count = n(), Percent = Count/nrow(.)) %>%
  filter(Count >= 3) %>%
  arrange(desc(Count)) %>% 
  View()

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  filter(AnalogyMod == "fake" & AnalogyNoun == "news") %>%
  summarize(Count = n(), Percent = Count/nrow(.)) 

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  filter(AnalogyMod == "homemade" & AnalogyNoun == "cookies") %>%
  View()

## Classification ----

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  mutate(AnalogyAvailable = AnalogyAvailable == TRUE,
         AnalogyType = fct_relevel(factor(case_when(SameAdjectiveAnalogy == TRUE & SameNounAnalogy == FALSE ~ "Same adjective, different noun",
                                        SameNounAnalogy == TRUE & SameAdjectiveAnalogy == FALSE ~ "Same noun, different adjective", # adjective/modifier
                                        is.na(SameAdjectiveAnalogy) & SameNounAnalogy == TRUE ~ "Same noun, no adjective",
                                        is.na(SameAdjectiveAnalogy) & SameNounAnalogy == FALSE ~ "No adjective, different noun",
                                        is.na(SameNounAnalogy) & SameAdjectiveAnalogy == FALSE ~ "No noun, different adjective",
                                        SameAdjectiveAnalogy == FALSE & SameNounAnalogy == FALSE ~ "Different adjective and noun",
                                          .default = "Other"),
                                       ), "Other", after = Inf)) -> analogy_prompting_types

analogy_prompting_types %>% filter(AnalogyType == "Other") %>%
  select(UserId, Bigram, OriginalAnalogyText, AnalogyText, AnalogyMod, AnalogyNoun)

analogy_prompting_types %>%
  filter(AnalogyType == "No adjective, different noun") %>%
  select(UserId, Bigram, AnalogyMod, AnalogyNoun) %>%
  View()


analogy_prompting_types %>%
  filter(AnalogyType == "Different adjective/modifier and noun") %>%
  select(UserId, Bigram, AnalogyMod, AnalogyNoun) %>%
  sample_n(25) %>%
  View()

analogy_prompting_types %>%
  filter(AnalogyType == "Same adjective, different noun") %>%
  select(UserId, Bigram, AnalogyMod, AnalogyNoun) %>%
  sample_n(25) %>%
  View()


analogy_prompting_types%>%
  group_by(AnalogyType) %>%
  summarize(Count = n(), Percent = Count/nrow(.), 
            PercentText = paste0(round(Percent, 3) * 100, "%"),
            AnalogyAvailable=first(AnalogyAvailable)) ->
  analogy_prompting_types_pct 
analogy_prompting_types_pct %>% select(AnalogyType, PercentText)

analogy_prompting_types_pct %>%
  mutate(AnalogyType = fct_relevel(AnalogyType, "No adjective, different noun", "Different adjective and noun", "Same noun, different adjective",  "Same adjective, different noun", )) %>%
  ggplot(aes(x=AnalogyAvailable, fill=AnalogyType, y=Percent)) +
  geom_bar(position="fill", stat="identity") +
  # geom_text(aes(label=PercentText), position="fill", vjust=1) + 
  labs(fill = "Analogy Type", x = "Analogy Provided", y = "Percentage") +
  theme_minimal() +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_fill_manual(values = paper_color_scale) ->
  ap_pct_used

ap_pct_used
ggsave("plots/analogy_prompting_percent_adjectives_used.png", width=3.75, height=1.75)

ap_pct_used +
  theme(text=element_text(size=30, color="#2C365E"))
ggsave("plots/analogy_prompting_percent_adjectives_used_poster.png", width=21, height=10, units="cm")


ap_pct_used +
  theme(text = element_text(size=12, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10),
        legend.key.size = unit(0.75, "lines")
  )
ggsave("plots/analogy_prompting_percent_adjectives_used_diss.png", width=3.75, height=1.5, units="in")


## Popular bigrams ----

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  mutate(AnalogyBigram = paste(AnalogyMod, AnalogyNoun)) %>%
  group_by(AnalogyBigram) %>%
  summarise(Count = n(), UserCount = n_distinct(UserId), Dupl = UserCount < Count) %>%
  filter(Dupl)

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  mutate(AnalogyBigram = paste(AnalogyMod, AnalogyNoun)) %>%
  group_by(AnalogyBigram) %>%
  summarize(Count = n()) %>%
  filter(Count > 1) ->
  popular_analogy_bigrams

popular_analogy_bigrams %>% nrow()

popular_analogy_bigrams %>% 
  filter(Count >= 3) %>% 
  arrange(desc(Count)) %>%
  print(n = Inf)
# knockoff purse, counterfeit money, knockoff handbag very popular! Also tiny house 

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  mutate(AnalogyBigram = paste(AnalogyMod, AnalogyNoun)) %>%
  filter(AnalogyBigram %in% (popular_analogy_bigrams %>% 
                               filter(Count >= 6) %>% 
                               pull(AnalogyBigram))) %>%
  select(UserId, Bigram, AnalogyBigram)

# Distribution comparison ----

## Basis for analogy distribution (how many people used an analogy) ----

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  group_by(Bigram) %>%
  summarize(AnalogyCount = n()) ->
  analogy_prompting_available_counts

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == FALSE) %>%
  group_by(Bigram) %>%
  summarize(AnalogyCount = n()) ->
  analogy_prompting_unavailable_counts


## Are the distributions the same when analogy is used ----

### JS divergence ----

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == TRUE) %>%
  build_human_dist() %>%
  mutate(HumanOrLM = "HumanAnalogyPrompting") ->
  analogy_prompting_dist

calculate_distribution_js(human_dist, analogy_prompting_dist) ->
  analogy_prompting_divergences

analogy_prompting_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

analogy_prompting_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

 analogy_prompting_divergences %>%
  group_by(Adjective) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

analogy_prompting_divergences %>%
  group_by(CoarseFrequency) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

analogy_prompting_divergences %>%
  arrange(desc(JSDivergence)) %>%
  head(n=12)

## Are the distributions the same when analogy is not used ----
# (and asked to answer quickly)

### JS divergence ----

table(analogy_prompting_cleaned[, c("AnalogyAvailable")])

analogy_prompting_cleaned %>%
  filter(AnalogyAvailable == FALSE) %>%
  build_human_dist() %>%
  mutate(HumanOrLM = "HumanAnalogyPrompting") ->
  analogy_prompting_unavail_dist

calculate_distribution_js(human_dist, analogy_prompting_unavail_dist) ->
  analogy_prompting_unavail_divergences

analogy_prompting_unavail_divergences %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

analogy_prompting_unavail_divergences %>%
  group_by(AdjectiveClass) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

analogy_prompting_unavail_divergences %>%
  group_by(Adjective) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

analogy_prompting_unavail_divergences %>%
  group_by(CoarseFrequency) %>%
  summarize(Mean=mean(JSDivergence), SD=sd(JSDivergence))

## Plot the distributions ----

split_bar_plot(list(human_dist, analogy_prompting_dist),
               names_to_colors(list("Human out-of-the-blue", "Human analogy prompting")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = analogy_prompting_bigrams,
               facet_by = "Bigram",
               facet_wrap_width=16) +
  geom_text(mapping = aes(x = 1, y = 0.7, label = AnalogyCount),
            color = light_blue_color, size = 3,
            data = analogy_prompting_available_counts %>%
              mutate(AnalogyCount = paste0("n=", AnalogyCount))) +
  geom_text(mapping = aes(x = 1, y = -0.7, label = Count),
            color = magenta_color, size = 3,
            data = isa_data_12_combined %>%
              filter(Bigram %in% analogy_prompting_bigrams) %>%
              group_by(Bigram) %>%
              summarize(Count = n()) %>%
              mutate(Count = paste0("n=", Count)))
ggsave("plots/analogy_prompting_all_bigrams_splitbar.png", height=8, width=22, units = 'in')

# Homemade only
split_bar_plot(list(human_dist, analogy_prompting_dist),
               names_to_colors(list("Human out-of-the-blue", "Human analogy prompting")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = analogy_prompting_cleaned %>% filter(Adjective == "homemade") %>% distinct(Bigram) %>% pull(Bigram),
               facet_by = "Bigram",
               facet_wrap_width=4) +
  geom_text(mapping = aes(x = 1, y = 0.7, label = AnalogyCount),
            color = light_blue_color, size = 3,
            data = analogy_prompting_available_counts %>% 
              filter(grepl("homemade", Bigram)) %>%
              mutate(AnalogyCount = paste0("n=", AnalogyCount))) +
  geom_text(mapping = aes(x = 1, y = -0.7, label = Count),
            color = magenta_color, size = 3,
            data = isa_data_12_combined %>%
              filter(Bigram %in% analogy_prompting_bigrams) %>% 
              filter(Adjective == "homemade") %>%
              group_by(Bigram) %>%
              summarize(Count = n()) %>%
              mutate(Count = paste0("n=", Count))) +
  theme(legend.position = 'bottom')
ggsave("plots/analogy_prompting_homemade_splitbar.png", width=6, height=4)




## KS test ----

ks.test(NumRating ~ Method, data = bind_rows(analogy_prompting_cleaned %>%
                                            filter(AnalogyAvailable == TRUE) %>%
                                            select(UserId, Bigram, NumRating) %>%
                                            mutate(Method = "AnalogyPrompting"),
                                          isa_data_combined %>%
                                            filter(Bigram %in% analogy_prompting_bigrams) %>%
                                            select(UserId, Bigram, NumRating) %>%
                                            mutate(Method = "OOTB")
                                             )
        ) 
# Kolmogorov-Smirnoff test suggests that data is from different distributions overall, because p < 0.05

ks_test_by_bigram(isa_data_combined %>%
                    mutate(HumanOrLM = "Human"), 
                  analogy_prompting_cleaned %>%
                    filter(AnalogyAvailable == TRUE) %>%
                    mutate(HumanOrLM = "AnalogyPrompting"),
                  group_variable = "Bigram",
                  adjust=FALSE) -> ks_p_values
View(ks_p_values)

# None are significant when I adjust with Holm-Bonferroni (way too many tests)

ks_p_values %>%
  filter(KS_pvalue < 0.05) %>%
  select(Bigram, KS_pvalue) ->
  signif_ks_bigrams_with_ps

signif_ks_bigrams_with_ps

signif_ks_bigrams_with_ps %>%
  ungroup() %>%
  pull(Bigram) -> signif_ks_bigrams

# We don't have enough ratings in each distribution for the KS test to yield helpful p-values per bigram for most of the bigrams :(
# But we do get p < 0.05 for homemade money, homemade currency, false friend
# and also knockoff picture, fake concert, fake reef, and tiny proposal

# Use top 6 JS divergences instead
signif_ks_bigrams <- c("homemade money", "homemade currency", "fake concert", "false friend", "homemade cat", "false gentleman")

# Significant bigrams only

split_bar_plot(list(human_dist, analogy_prompting_dist),
               names_to_colors(list("Human out-of-the-blue", "Human analogy prompting")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = signif_ks_bigrams,
               facet_by = "Bigram",
               facet_wrap_width=3) +
  geom_text(mapping = aes(x = 3, y = 0.7, label = AnalogyCount),
            color = light_blue_color, size = 3,
            data = analogy_prompting_available_counts %>% 
              filter(Bigram %in% signif_ks_bigrams) %>%
              mutate(AnalogyCount = paste0("n=", AnalogyCount))) +
  geom_text(mapping = aes(x = 3, y = -0.7, label = Count),
            color = magenta_color, size = 3,
            data = isa_data_12_combined %>%
              filter(Bigram %in% signif_ks_bigrams) %>% 
              group_by(Bigram) %>%
              summarize(Count = n()) %>%
              mutate(Count = paste0("n=", Count))) +
  theme(legend.position = 'bottom')
ggsave("plots/analogy_prompting_signif_splitbar.png", width=4, height=4)

split_bar_plot(list(human_dist, analogy_prompting_dist),
               names_to_colors(list("Human out-of-the-blue", "Human analogy prompting")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = signif_ks_bigrams,
               facet_by = "Bigram",
               facet_wrap_width=6) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) ->
  ap_signif_splitbar_w

ap_signif_splitbar_w +
  geom_text(mapping = aes(x = 3, y = 0.7, label = AnalogyCount),
            color = light_blue_color, size = 3,
            data = analogy_prompting_available_counts %>% 
              filter(Bigram %in% signif_ks_bigrams) %>%
              mutate(AnalogyCount = paste0("n=", AnalogyCount))) +
  geom_text(mapping = aes(x = 3, y = -0.7, label = Count),
            color = magenta_color, size = 3,
            data = isa_data_12_combined %>%
              filter(Bigram %in% signif_ks_bigrams) %>% 
              group_by(Bigram) %>%
              summarize(Count = n()) %>%
              mutate(Count = paste0("n=", Count)))
ggsave("plots/analogy_prompting_signif_splitbar_wide.png", width=7.5, height=2)

ap_signif_splitbar_w +
  geom_text(mapping = aes(x = 3, y = 0.7, label = AnalogyCount),
            color = light_blue_color, size = 8,
            data = analogy_prompting_available_counts %>% 
              filter(Bigram %in% signif_ks_bigrams) %>%
              mutate(AnalogyCount = paste0("n=", AnalogyCount))) +
  geom_text(mapping = aes(x = 3, y = -0.7, label = Count),
            color = magenta_color, size = 8,
            data = isa_data_12_combined %>%
              filter(Bigram %in% signif_ks_bigrams) %>% 
              group_by(Bigram) %>%
              summarize(Count = n()) %>%
              mutate(Count = paste0("n=", Count))) +
  theme(text=element_text(size=24, color="#2C365E"))
ggsave("plots/analogy_prompting_signif_splitbar_wide_poster.png", width=45.5, height=11, units="cm")

split_bar_plot(list(human_dist, analogy_prompting_dist),
               names_to_colors(list("Human out-of-the-blue", "Human analogy prompting")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = signif_ks_bigrams,
               facet_by = "Bigram",
               facet_wrap_width=6,
               bigram_wrap_length = 12) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) + 
  geom_text(mapping = aes(x = 3, y = 0.75, label = AnalogyCount),
            color = light_blue_color, size = 3,
            family = "Palatino Linotype",
            data = analogy_prompting_available_counts %>% 
              filter(Bigram %in% signif_ks_bigrams) %>%
              mutate(AnalogyCount = paste0("n=", AnalogyCount),
                     Bigram = insert_linebreaks(Bigram, width=12))) +
  geom_text(mapping = aes(x = 3, y = -0.75, label = Count),
            color = magenta_color, size = 3,
            family = "Palatino Linotype",
            data = isa_data_12_combined %>%
              filter(Bigram %in% signif_ks_bigrams) %>% 
              group_by(Bigram) %>%
              summarize(Count = n()) %>%
              mutate(Count = paste0("n=", Count),
                     Bigram = insert_linebreaks(Bigram, width=12))) +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) 
ggsave("plots/analogy_prompting_signif_splitbar_wide_diss.png", width=6.5, height=2.25)



ks_test_by_bigram(isa_data_combined %>%
                    mutate(HumanOrLM = "Human"), 
                  analogy_prompting_cleaned %>%
                    filter(AnalogyAvailable == TRUE) %>%
                    mutate(HumanOrLM = "AnalogyPrompting"),
                  group_variable = "Adjective",
                  adjust=TRUE) -> ks_adjective_p_values
View(ks_adjective_p_values)
# homemade is significantly different overall even after adjusting

analogy_prompting_cleaned %>%
  filter(Bigram %in% signif_ks_bigrams & AnalogyAvailable) %>%
  select(UserId, Bigram, AnalogyMod, AnalogyNoun) %>%
  View()

# Regressions over human data ----

analogy_prompting_cleaned %>%
  mutate(across(c("DivergentSimilarBigrams", "ConflictingConvergentSimilarBigrams", "Convergent"), as.factor)) %>%
  mutate(AdjectiveClass = fct_relevel(AdjectiveClass, "Subsective"))  ->
  analogy_prompting_lm_ready

str(analogy_prompting_lm_ready)

analogy_prompting_lm_ready %>%
  group_by(Bigram) %>%
  summarize(AnalogyAvailablePercent = mean(AnalogyAvailable),
            Adjective = first(Adjective), Noun = first(Noun), 
            across(AdjectiveClass:DivergentSimilarBigrams, .fns = ~unique(.x))) %>% 
  merge(analogy_prompting_divergences %>% select(Bigram, JSDivergence), by = "Bigram") ->
  analogy_prompting_by_bigram 

str(analogy_prompting_by_bigram)

## When is analogy hard ----

# Doesn't converge if we use a scalar for divergent similar bigrams
# Doesn't converge if we have adjective as a fixed effect
# Doesn't converge if we include both coarse bigram frequency and WordNet depth as fixed effect, since they're too correlated
analogy_hard_lm <- glmer(AnalogyAvailable ~ AdjectiveClass + DivergentSimilarBigrams + ConflictingConvergentSimilarBigrams + HighFNeighbourQuartile + WordnetDepth + (1 | UserId) + (1 | Adjective),
                         data = analogy_prompting_lm_ready, family = "binomial")
summary(analogy_hard_lm)

# WordNet depth and divergent similar bigrams are significant, but only 6 bigrams have divergent similar bigrams
# so this may be overfit...
# Update: WordNet depth no longer significant once we have a few more ratings (12 for each, was missing 16 participants)
plot(predict_response(analogy_hard_lm, terms = c("DivergentSimilarBigrams")))

# We see a trend but not a significant effect of 90-99th percentile bigram frequency on whether an analogy is available (p=0.08),
# and no effect of any other frequencies
analogy_freq_lm <- glmer(AnalogyAvailable ~ DivergentSimilarBigrams + CoarseFrequency + HighFNeighbourQuartile + (1 | UserId) + (1 | Adjective),
                         data = analogy_prompting_lm_ready, family = "binomial")
summary(analogy_freq_lm)

cor.test(x=analogy_prompting_lm_ready$WordnetDepth, y=as.integer(analogy_prompting_lm_ready$CoarseFrequency), method = "spearman")

## Does analogy availability correlate with JS divergence?

# Linear model isn't great because it allows JS divergences below 0
# Yes, strong negative correlation -> more analogy availability = better fit to OOTB distribution
analogy_divergence_lm1 <- lm(JSDivergence ~ AnalogyAvailablePercent, data = analogy_prompting_by_bigram)
summary(analogy_divergence_lm1)

plot(predict_response(analogy_divergence_lm1, terms = c("AnalogyAvailablePercent")))

analogy_prompting_by_bigram %>%
  filter(JSDivergence %in% c(0, 1))
analogy_divergence_lm2 <- glmmTMB(JSDivergence ~ AnalogyAvailablePercent, 
                                  data = analogy_prompting_by_bigram %>%
                                    mutate(JSDivergence = if_else(JSDivergence == 0, 1e-5, if_else(JSDivergence == 1, 0.99999, JSDivergence))), 
                                  family = beta_family)
summary(analogy_divergence_lm2)

plot(predict_response(analogy_divergence_lm2, terms = c("AnalogyAvailablePercent"))) +
  scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1), limits = c(0, 1.01))

# Singular fit
analogy_divergence_lm3 <- lmer(JSDivergence ~ AnalogyAvailablePercent + (1 | Adjective) + (1 | Noun), data = analogy_prompting_by_bigram)
summary(analogy_divergence_lm3)

# Comparisons between analogy model and humans/LLMs ----

## Load data ----

analogy_model_divergences <- read.csv('../Adjectives-PythonCode/output/analogy/analogy_model_glove_2a_100n_5b_weighted-sim.csv')
analogy_model_divergences_memo <- read.csv('../Adjectives-PythonCode/output/analogy/analogy_model_glove_2a_100n_5b_weighted-sim_memo.csv')

analogy_model_llama_divergences <- read.csv('../Adjectives-PythonCode/output/analogy/analogy_model_llama-final_1a_100n_5b_weighted-sim.csv')
analogy_model_llama_divergences_memo <- read.csv('../Adjectives-PythonCode/output/analogy/analogy_model_llama-final_1a_100n_5b_weighted-sim_memo.csv')


analogy_model_divergences %>% 
  # Fix some floating point errors in distributions
  mutate(across(Rating1:Rating5, ~ . / (Rating1 + Rating2 + Rating3 + Rating4 + Rating5))) ->
  analogy_model_divergences

analogy_model_divergences_memo %>% 
  # Fix some floating point errors in distributions
  mutate(across(Rating1:Rating5, ~ . / (Rating1 + Rating2 + Rating3 + Rating4 + Rating5))) ->
  analogy_model_divergences_memo

merge(analogy_prompting_divergences %>%
        rename(AnalogyPromptingJSDivergence = JSDivergence), 
      analogy_model_divergences %>% 
        select(Bigram, JSDivergence) %>%
        rename(AnalogyModelJSDivergence = JSDivergence), 
      by="Bigram") %>%
  merge(analogy_prompting_cleaned %>% 
          group_by(Bigram) %>%
          summarize(AnalogyAvailablePercent = mean(AnalogyAvailable)),
        by="Bigram"
        )-> human_am_divergences

merge(analogy_model_llama_divergences %>% 
        select(Bigram, JSDivergence) %>%
        rename(AnalogyModelJSDivergence = JSDivergence), 
      llama3i_human_lm_divergences %>% 
        select(Bigram, JSDivergence) %>%
        rename(LLMJSDivergence = JSDivergence), 
      by="Bigram") -> llm_am_divergences

merge(analogy_model_llama_divergences_memo %>% 
        select(Bigram, JSDivergence) %>%
        rename(AnalogyModelJSDivergence = JSDivergence), 
      llama3i_human_lm_divergences %>% 
        select(Bigram, JSDivergence) %>%
        rename(LLMJSDivergence = JSDivergence), 
      by="Bigram") -> llm_am_divergences_memo

merge(analogy_prompting_divergences %>%
        rename(AnalogyPromptingJSDivergence = JSDivergence), 
      llama3i_human_lm_divergences %>% 
        select(Bigram, JSDivergence) %>%
        rename(LLMJSDivergence = JSDivergence), 
      by="Bigram") %>%
  merge(analogy_prompting_cleaned %>% 
          group_by(Bigram) %>%
          summarize(AnalogyAvailablePercent = mean(AnalogyAvailable)),
        by="Bigram"
  )-> human_prompting_llm_divergences

## KS tests ----

# In order to do a KS test we need to sample from the distribution predicted by the analogy model


ks_sampling_n <- 100

set.seed(42)
analogy_model_divergences %>%
  rowwise() %>%
  mutate(NumRating = list(r(DiscreteDistribution(
    supp = c(1, 2, 3, 4, 5), 
    prob = c(Rating1,
             Rating2,
             Rating3,
             Rating4,
             Rating5))
  )(ks_sampling_n))) %>%
  unnest(cols = c(NumRating)) %>%
  mutate(Rating = case_when(
    NumRating == 5 ~ "Definitely yes",
    NumRating == 4 ~ "Probably yes",
    NumRating == 3 ~ "Unsure",
    NumRating == 2 ~ "Probably not",
    NumRating == 1 ~ "Definitely not"
  )) %>%
  mutate(Rating = as.factor(Rating)) %>%
  select(Bigram, Rating, NumRating) ->
  analogy_model_sampled

set.seed(42)
analogy_model_divergences_memo %>%
  rowwise() %>%
  mutate(NumRating = list(r(DiscreteDistribution(
    supp = c(1, 2, 3, 4, 5), 
    prob = c(Rating1,
             Rating2,
             Rating3,
             Rating4,
             Rating5))
  )(ks_sampling_n))) %>%
  unnest(cols = c(NumRating)) %>%
  mutate(Rating = case_when(
    NumRating == 5 ~ "Definitely yes",
    NumRating == 4 ~ "Probably yes",
    NumRating == 3 ~ "Unsure",
    NumRating == 2 ~ "Probably not",
    NumRating == 1 ~ "Definitely not"
  )) %>%
  mutate(Rating = as.factor(Rating)) %>%
  select(Bigram, Rating, NumRating) ->
  analogy_model_memo_sampled
  
ks_test_by_bigram(isa_data_combined %>%
                    mutate(HumanOrLM = "Human"), 
                  analogy_model_sampled %>%
                    mutate(HumanOrLM = "AnalogyModel"),
                  group_variable = "Bigram",
                  adjust=TRUE) -> am_ks_p_values
am_ks_p_values %>%
  filter(KS_pvalue < 0.05) %>%
  select(Bigram, KS_pvalue) ->
  am_signif_ks_bigrams_with_ps

# 116 bigrams are significantly different between the analogy model and humans,
# according to the KS test, based on sampling from the predicted distribution with n=12
# (If we sample with n=100, it goes up to 225 different bigrams)
# Only 3 (false friend, former couple, homemade money) are significantly different after adjusting with Holm-Bonferroni when sampling at n=12
# But 20 if we sample with n=100
nrow(am_signif_ks_bigrams_with_ps)

nrow(am_signif_ks_bigrams_with_ps) / nrow(am_ks_p_values)

am_signif_ks_bigrams_with_ps %>%
  add_frequency() %>%
  group_by(AdjectiveClass) %>%
  summarize(n())

am_signif_ks_bigrams_with_ps %>%
  add_frequency() %>%
  group_by(Adjective) %>%
  summarize(n())

am_signif_ks_bigrams_with_ps %>%
  add_frequency() %>%
  group_by(CoarseFrequency) %>%
  summarize(n())

116 / nrow(analogy_model_divergences)

ks_test_by_bigram(isa_data_combined %>%
                    mutate(HumanOrLM = "Human"), 
                  analogy_model_memo_sampled %>%
                    mutate(HumanOrLM = "AnalogyModel"),
                  group_variable = "Bigram",
                  adjust=TRUE) -> am_memo_ks_p_values
am_memo_ks_p_values %>%
  filter(KS_pvalue < 0.05) %>%
  select(Bigram, KS_pvalue) ->
  am_memo_signif_ks_bigrams_with_ps

# 73 bigrams are significantly different between the memorizing analogy model and humans,
# according to the KS test, based on sampling from the predicted distribution with n=12
# But 1 of these is 90-99th percentile so this is just a sampling fluke
# Call it 72.
# After adjusting p-values (Holm-Bonferroni) only 1 is significant with n=12, namely homemade money
# 10 are significant with n=100
nrow(am_memo_signif_ks_bigrams_with_ps)

nrow(am_memo_signif_ks_bigrams_with_ps) / nrow(am_memo_ks_p_values)

nrow(am_memo_signif_ks_bigrams_with_ps) / nrow(analogy_model_divergences)

am_memo_signif_ks_bigrams_with_ps %>%
  add_frequency() %>%
  group_by(AdjectiveClass) %>%
  summarize(n())

am_memo_signif_ks_bigrams_with_ps %>%
  add_frequency() %>%
  group_by(Adjective) %>%
  summarize(n())

am_memo_signif_ks_bigrams_with_ps %>%
  add_frequency() %>%
  group_by(CoarseFrequency) %>%
  summarize(n())

### Plot mismatching bigrams ----
  

split_bar_plot(list(human_dist, analogy_model_divergences_memo %>%
                      rename("Definitely not"=Rating1,
                             "Probably not"=Rating2,
                             "Unsure"=Rating3,
                             "Probably yes"=Rating4,
                             "Definitely yes"=Rating5)),
               names_to_colors(list("Human out-of-the-blue", "Analogy model")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = am_memo_signif_ks_bigrams_with_ps %>% ungroup() %>% filter(Bigram %in% (noun_difficulties %>% filter(Convergent) %>% pull(Bigram))) %>% pull(Bigram),
               facet_by = "Bigram",
               facet_wrap_width=10)

am_memo_signif_bigrams_to_plot <- c("homemade money", "artificial gold", "tiny abundance", "multicolored gold", "tiny handbag", "unimportant instructions")
split_bar_plot(list(human_dist, analogy_model_divergences_memo %>%
                      rename("Definitely not"=Rating1,
                             "Probably not"=Rating2,
                             "Unsure"=Rating3,
                             "Probably yes"=Rating4,
                             "Definitely yes"=Rating5)),
               names_to_colors(list("Human out-of-the-blue", "Analogy model")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = am_memo_signif_bigrams_to_plot,
               facet_by = "Bigram",
               facet_wrap_width=6) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) ->
  am_signif_splitbar_w
am_signif_splitbar_w
ggsave("plots/analogy_model_signif_splitbar_wide.png", width=8.5, height=2.25)
  
am_signif_splitbar_w +
  theme(text=element_text(size=24, color="#2C365E"))
ggsave("plots/analogy_model_signif_splitbar_wide_poster.png", width=45.5, height=11, units="cm")

split_bar_plot(list(human_dist, analogy_model_divergences_memo %>%
                      rename("Definitely not"=Rating1,
                             "Probably not"=Rating2,
                             "Unsure"=Rating3,
                             "Probably yes"=Rating4,
                             "Definitely yes"=Rating5)),
               names_to_colors(list("Human out-of-the-blue", "Analogy model")),
               human_name = "Human out-of-the-blue",
               vertical = TRUE,
               bigrams = am_memo_signif_bigrams_to_plot,
               facet_by = "Bigram",
               facet_wrap_width=6,
               bigram_wrap_length = 12) +
  theme(legend.position = "bottom",
        legend.key.size = unit(0.4, 'cm'),
        legend.spacing.y = unit(0.1, 'cm'),
        legend.box.spacing = unit(0, 'pt')) +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  )
ggsave("plots/analogy_model_signif_splitbar_wide_diss.png", width=6.5, height=2.25, units="in")


## Regressions ----

# R-squared of 0.395
human_analogy_model_divergence_lm <- lm(AnalogyPromptingJSDivergence ~ AnalogyModelJSDivergence,
                                          data = human_am_divergences)
summary(human_analogy_model_divergence_lm)

# R-squared of just 0.17, less predictive
human_analogy_model_available_lm <- lm(AnalogyModelJSDivergence ~ AnalogyAvailablePercent,
                                        data = human_am_divergences)
summary(human_analogy_model_available_lm)

# R-squared of just 0.117 - analogy model not super predictive of what LLM finds hard
llm_analogy_model_divergence_lm <- lm(LLMJSDivergence ~ AnalogyModelJSDivergence,
                                        data = llm_am_divergences)
summary(llm_analogy_model_divergence_lm)
llm_analogy_model_memo_divergence_lm <- lm(LLMJSDivergence ~ AnalogyModelJSDivergence,
                                      data = llm_am_divergences_memo)
summary(llm_analogy_model_memo_divergence_lm)

# R-squared of just 0.05 with human JS divergence, terrible fit
# R-squared of 0.05 with analogy available percent, too
human_prompting_llm_divergence_lm <- lm(LLMJSDivergence ~ AnalogyAvailablePercent,
                                        data = human_prompting_llm_divergences)
summary(human_prompting_llm_divergence_lm)

## Analogy model error analysis ----

analogy_model_divergences %>%
  merge(isa_variance_12_combined %>% select(Bigram, Mean, SD), by="Bigram") %>%
  slice_max(JSDivergence, n=25) %>%
  View()

analogy_model_divergences %>%
  merge(isa_variance_12_combined %>% select(Bigram, Adjective, Mean, SD), by="Bigram") %>%
  filter(Adjective %in% c("knockoff", "counterfeit") & Mean < 4) %>%
  View()

analogy_model_fit_lm1 <- lm(JSDivergence ~ AdjectiveClass * Mean + SD,
                           data = analogy_model_divergences %>%
                             merge(isa_variance_12_combined %>%
                                     select(Bigram, Mean, SD, CoarseFrequency, AdjectiveClass), 
                                   by="Bigram") %>%
                             mutate(AdjectiveClass = fct_relevel(AdjectiveClass, "Subsective"))
                           )
summary(analogy_model_fit_lm1)

analogy_model_fit_lm2 <- lm(JSDivergence ~ Adjective * Mean + SD,
                            data = analogy_model_divergences %>%
                              merge(isa_variance_12_combined %>%
                                      select(Bigram, Mean, SD, CoarseFrequency, Adjective, AdjectiveClass), 
                                    by="Bigram") %>%
                              mutate(Adjective = fct_relevel(Adjective, "multicolored"))
)
summary(analogy_model_fit_lm2)

# Analogy model plots ----

am_performance <- read.csv("am_performance.csv") %>%
  pivot_longer(starts_with("JS"), names_to = "EvalSplit", names_prefix="JS", values_to="JS") %>%
  mutate(across(c("AdjCount", "BigramCount", "EmbeddingType", "TrainingData", "EvalSplit"), as.factor)) %>%
  mutate(EvalSplit = fct_recode(EvalSplit, "Total (with mem enabled)"="TotalMem", "Zero-frequency bigrams"="ZeroFrequency"),
         Baseline = !(EmbeddingType %in% c("GloVe embeddings", "WordNet", "Llama 3 70B embeddings")),
         NegJS = 1 - JS) %>%
  unite(ModelName, c("AdjCount", "BigramCount"), sep="\n", remove=FALSE) %>%
  mutate(ModelName = fct_relevel(factor(ModelName), "N only\nBest bigram", "N only\nTop-k bigrams", "Multi-A\nBest bigram", "Multi-A\nTop-k bigrams"),
         AdjCount = factor(AdjCount, levels=c("N only", "Multi-A")),
         BigramCount = factor(BigramCount, levels=c("Best bigram", "Top-k bigrams")))

am_performance %>% 
  filter(!Baseline & EvalSplit != "Total" & TrainingData == "Top quartile" & JS != 1.0) %>%
  ggplot(aes(x=ModelName, y=NegJS, color=EmbeddingType, shape=EvalSplit, linetype=EvalSplit, group=interaction(EmbeddingType, EvalSplit))) +
  geom_point() +
  geom_line() +
  geom_hline(data = am_performance %>% filter(ModelName == "Human baseline (resampled)" & EvalSplit == "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dotted") +
  geom_hline(data = am_performance %>% filter(ModelName == "Llama 3 70B Instruct" & EvalSplit == "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dotted") +
  geom_hline(data = am_performance %>% filter(ModelName == "Uniform distr. baseline" & EvalSplit == "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dotted") +
  theme_minimal() +
  labs(y="1 - JS Divergence", x="Model Complexity", color="Similarity Metric", shape="Evaluation Set", linetype="Evaluation Set") +
  scale_color_manual(values = names_to_colors(c("GloVe embeddings", "WordNet", "Llama 3 70B embeddings", "Human baseline (resampled)", "Llama 3 70B Instruct", "Uniform distr. baseline"))) +
  scale_linetype_manual(values = c("solid", "dashed"))
ggsave("plots/analogy_model_jsdivergence_lineplot.png", width=8, height=3.25, units="in")


am_performance %>%
  filter(!Baseline & EvalSplit != "Total" & TrainingData == "Top quartile") %>%
  ggplot(aes(x=ModelName, y=NegJS, fill=EmbeddingType)) +
  geom_col(position = "dodge") +
  geom_hline(data = am_performance %>% filter(EmbeddingType == "Human baseline (resampled)" & EvalSplit != "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dashed") +
  geom_hline(data = am_performance %>% filter(EmbeddingType == "Llama 3 70B Instruct" & EvalSplit != "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dashed") +
  geom_hline(data = am_performance %>% filter(EmbeddingType == "Uniform distr. baseline" & EvalSplit != "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dashed") +
  facet_wrap(~ EvalSplit) +
  theme_minimal() +
  labs(y="1 - JS Divergence", x="Model Complexity", fill="Similarity Metric", color="Baseline") +
  scale_fill_manual(values = names_to_colors(c("GloVe embeddings", "WordNet", "Llama 3 70B embeddings", "Human baseline (resampled)", "Llama 3 70B Instruct", "Uniform distr. baseline"))) +
  scale_color_manual(values = names_to_colors(c("GloVe embeddings", "WordNet", "Llama 3 70B embeddings", "Human baseline (resampled)", "Llama 3 70B Instruct", "Uniform distr. baseline")))
ggsave("plots/analogy_model_jsdivergence_barplot.png", width=10, height=3.25, units="in")


am_performance %>%
  filter(!Baseline & EvalSplit != "Total" & TrainingData == "Top quartile" & BigramCount == "Top-k bigrams") %>%
  mutate(AdjCount = fct_recode(AdjCount, "Noun + Adjective"="Multi-A", "Noun only"="N only")) %>%
  ggplot(aes(x=AdjCount, y=NegJS, fill=EmbeddingType)) +
  geom_col(position = "dodge") +
  geom_hline(data = am_performance %>% filter(EmbeddingType == "Human baseline (resampled)" & EvalSplit != "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dashed") +
  geom_hline(data = am_performance %>% filter(EmbeddingType == "Llama 3 70B Instruct" & EvalSplit != "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dashed") +
  geom_hline(data = am_performance %>% filter(EmbeddingType == "Uniform distr. baseline" & EvalSplit != "Total"), 
             aes(yintercept=NegJS, color=EmbeddingType),
             linetype="dashed") +
  facet_wrap(~ EvalSplit) +
  theme_minimal() +
  labs(y="1 - JS Divergence", x="Analogy Type", fill="Similarity Metric", color="Baseline") +
  scale_fill_manual(values = names_to_colors(c("GloVe embeddings", "WordNet", "Llama 3 70B embeddings", "Human baseline (resampled)", "Llama 3 70B Instruct", "Uniform distr. baseline"))) +
  scale_color_manual(values = names_to_colors(c("GloVe embeddings", "WordNet", "Llama 3 70B embeddings", "Human baseline (resampled)", "Llama 3 70B Instruct", "Uniform distr. baseline"))) -> 
  am_reduced_plot
#  theme(legend.position = "bottom", legend.box="vertical", legend.margin=margin())

am_reduced_plot
ggsave("plots/analogy_model_jsdivergence_barplot_reduced.png", width=8, height=2.75, units="in")

am_reduced_plot +
  theme(text=element_text(size=32, color="#2C365E"))
ggsave("plots/analogy_model_jsdivergence_barplot_reduced_poster.png", width=45, height=15, units="cm")

am_reduced_plot +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  )
ggsave("plots/analogy_model_jsdivergence_barplot_reduced_diss.png", width=6.75, height=2.25, units="in")
