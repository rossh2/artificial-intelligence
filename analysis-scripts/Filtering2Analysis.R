# Packages ----

library(tidyverse)
library(tidytext)
library(paletteer)

# Import data ----

filtering1_wide <- read.csv("results/human/exp1_filtering/filtering_responses_analogy1_raw.csv",
                           fileEncoding = "UTF-8-BOM")
filtering2_wide <- read.csv("results/human/exp1_filtering/filtering_responses_analogy2_raw.csv",
                            fileEncoding = "UTF-8-BOM")



# Remove the first two rows, which contain the header again and the full question text
filtering1_wide <- filtering1_wide[-c(1,2),]
rownames(filtering1_wide) <- NULL

filtering2_wide <- filtering2_wide[-c(1,2),]
rownames(filtering2_wide) <- NULL

# Select only question columns; drop all user information & comments except demographic questionnaire
filtering1_wide <- subset(filtering1_wide, 
                         select = grep("^(Q.|EnglishBefore5|Dialect|OtherLanguages|Comments)", 
                                       names(filtering1_wide)))
filtering2_wide <- subset(filtering2_wide, 
                         select = grep("^(Q.|EnglishBefore5|Dialect|OtherLanguages|Comments)", 
                                       names(filtering2_wide)))

# Add fake UserID (don't want to use actual Prolific ID)
filtering1_wide$UserId <- 1:nrow(filtering1_wide)
filtering2_wide$UserId <- (nrow(filtering1_wide)+1):(nrow(filtering1_wide)+nrow(filtering2_wide))

# Exclusion criteria ----

## Demographic ----

dem_excluded_ids1 <- filtering1_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & Dialect_2_TEXT != "American English")) %>% 
  pull(UserId)

dem_excluded_ids1

dem_excluded_ids2 <- filtering2_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & Dialect_2_TEXT != "American English")) %>% 
  pull(UserId)

dem_excluded_ids2

## Attention checks ----

filtering1_wide %>%
  select(sort(tidyselect::peek_vars())) %>%  # Sort columns alphabetically
  unite(Q.red.circle, Q.red.circle:Q.red.circle.9.2, sep="", remove = TRUE) %>%
  unite(Q.yellow.certainty, Q.yellow.certainty:Q.yellow.certainty.9.2, sep="", remove = TRUE) ->
  filtering1_wide

filtering2_wide %>%
  select(sort(tidyselect::peek_vars())) %>%  # Sort columns alphabetically
  unite(Q.red.circle, Q.red.circle:Q.red.circle.9.2, sep="", remove = TRUE) %>%
  unite(Q.yellow.certainty, Q.yellow.certainty:Q.yellow.certainty.9.2, sep="", remove = TRUE) ->
  filtering2_wide

filtering1_wide <- filtering1_wide %>%
  mutate(AttnFailed = ifelse(Q.red.circle %in% c("Very hard", "Somewhat hard"), 1, 0) + 
           ifelse(Q.yellow.certainty %in% c("Very easy"), 1, 0)
  )
filtering2_wide <- filtering2_wide %>%
  mutate(AttnFailed = ifelse(Q.red.circle %in% c("Very hard", "Somewhat hard"), 1, 0) + 
           ifelse(Q.yellow.certainty %in% c("Very easy"), 1, 0)
  )

attn_excluded_ids1 <- filtering1_wide %>%
  filter(AttnFailed >= 1) %>%
  pull(UserId)

attn_excluded_ids1

attn_excluded_ids2 <- filtering2_wide %>%
  filter(AttnFailed >= 1) %>%
  pull(UserId)

attn_excluded_ids2

filtering1_wide_excl <- filtering1_wide %>%
  filter(!UserId %in% dem_excluded_ids1 & !UserId %in% attn_excluded_ids1)

filtering2_wide_excl <- filtering2_wide %>%
  filter(!UserId %in% dem_excluded_ids2 & !UserId %in% attn_excluded_ids2)

## Select target questions and pivot ----

an_filtering1 <- filtering1_wide_excl %>%
  dplyr::select(Q.alleged.pianist:UserId) %>% 
  dplyr::select(!Q1) %>% # Drop Prolific ID 
  dplyr::select(!c("Q.alleged.pianist", "Q.stone.lion", "Q.married.weapon.", "Q.red.circle", "Q.yellow.certainty")) %>%
  pivot_longer(
    cols = Q.artificial.allegation:Q.useful.truck,
    names_to = c("Adjective","Noun"),
    names_pattern = "Q.([a-z]+).([a-z]+)",
    values_to = "Rating"
  ) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate_at(c("UserId", "Adjective", "Noun", "Bigram"), factor)

an_filtering2 <- filtering2_wide_excl %>%
  dplyr::select(Q.alleged.pianist:UserId) %>% 
  dplyr::select(!Q1) %>% # Drop Prolific ID 
  dplyr::select(!c("Q.alleged.pianist", "Q.stone.lion", "Q.married.weapon", "Q.red.circle", "Q.yellow.certainty")) %>%
  pivot_longer(
    cols = Q.artificial.abundance:Q.useful.window,
    names_to = c("Adjective","Noun"),
    names_pattern = "Q.([a-z]+).([a-z]+)",
    values_to = "Rating"
  ) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate_at(c("UserId", "Adjective", "Noun", "Bigram"), factor)

an_filtering_analogy <- bind_rows(an_filtering1, an_filtering2)

str(an_filtering_analogy)

# Drop rows with no data (no rating)
an_filtering_analogy <- an_filtering_analogy %>%
  filter(Rating!="") %>%
  mutate(Rating = factor(Rating, levels = c("Very hard", "Somewhat hard", "Somewhat easy", "Very easy"))) %>%
  mutate(NumRating = as.integer(Rating))

# Label privative adjectives
privative_as = c(
  "fake",
  "counterfeit",
  "false",
  "artificial",
  "knockoff",
  "former"
  )
intersective_as = c(
  "useful",
  "tiny",
  "illegal",
  "homemade",
  "unimportant",
  "multicolored"
  )

an_filtering_analogy <- an_filtering_analogy %>%
  mutate(AdjectiveClass = ifelse(Adjective %in% privative_as, "Privative", 
                                 ifelse(Adjective %in% intersective_as, "Intersective", "AttentionCheck"))) %>%
  mutate_at(c("AdjectiveClass"), factor)

str(an_filtering_analogy)

## Balance ----

an_filtering_analogy %>%
  distinct(Adjective, Bigram) %>%
  group_by(Adjective) %>%
  summarize(n())

an_filtering_analogy %>% 
  group_by(Bigram) %>%
  summarise(RatingCount = n(), Adjective = first(Adjective)) %>%
  filter(RatingCount < 3) #%>%
#  filter(Adjective == "fake")

# After exclusion, blocks 
# 9 (fake jewelry) - missing 1
# 16 (fake audience) - missing 1
# 20 (fake chair) - missing 2
# 26 (fake bed) - missing 1
# 47 (fake flower) - missing 1
# 55 (fake watch) - missing 1
# 56 (fake celebration) - missing 1
# have too few ratings
# So first half needs 5 more, second half needs 3 more
# Further, blocks
# 2 (fake ocean) - 1 extra
# 14 (fake effort) - 2 extra
# have too many ratings

# Filter bigrams ----

bigrams_by_rating_analogy <- an_filtering_analogy %>%
  mutate(Rating = fct_recode(Rating,
                             VeryEasy = "Very easy",
                             SomewhatEasy = "Somewhat easy", 
                             SomewhatHard = "Somewhat hard", 
                             VeryHard = "Very hard")) %>%
  group_by(Bigram, Rating) %>%
  summarise(RatingCount = length(Rating), .groups = "drop") %>%
  pivot_wider(
    names_from = Rating,
    values_from = RatingCount
  ) %>%
  replace(is.na(.), 0) %>%
  separate(Bigram, into = c("Adjective", "Noun"), remove = FALSE) %>%
  mutate(AdjectiveClass = ifelse(Adjective %in% privative_as, "Privative", 
                                 ifelse(Adjective %in% intersective_as, "Intersective", "AttentionCheck"))) %>%
  mutate_at(c("AdjectiveClass", "Adjective", "Noun"), factor)

bigrams_by_rating_analogy <- bigrams_by_rating_analogy %>%
  mutate(SenseRating = factor(ifelse(SomewhatEasy + VeryEasy >= 2, "Makes sense", "Nonsensical")))

table(bigrams_by_rating_analogy[, c("SenseRating")])

table(bigrams_by_rating_analogy[, c("Adjective", "SenseRating")])

sensical_bigrams_analogy <- bigrams_by_rating_analogy %>%
  filter(SenseRating == "Makes sense") %>%
  dplyr::select(Bigram, Adjective, Noun, AdjectiveClass)

write.table(sensical_bigrams_analogy %>% dplyr::select(Bigram), "analogy_makessense_bigrams.txt", row.names=FALSE,  col.names = FALSE, quote = FALSE)

nonsensical_bigrams_analogy <- bigrams_by_rating_analogy %>%
  filter(SenseRating == "Nonsensical") %>%
  dplyr::select(Bigram, Adjective, Noun, AdjectiveClass)

write.table(nonsensical_bigrams_analogy %>% dplyr::select(Bigram), "analogy_nonsensical_bigrams.txt", row.names=FALSE,  col.names = FALSE, quote = FALSE)

# Frequencies ----

## Add frequencies ----

c4_bigrams_freqs_analogy <- read.csv("../Adjectives-PythonCode/output/filtering_data/analogy60x12_bigrams_with_frequencies.csv", header = TRUE)

c4_unique_bigrams_freqs_analogy <- c4_bigrams_freqs_analogy %>%
  select(!MakesSense) %>%  # Always says "not known"
  mutate_at(c("Adjective", "Noun", "Bigram", "AdjectiveClass", "Frequency"), factor) %>%
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
  rename(C4Frequency = Frequency)

an_filtering_analogy <- merge(an_filtering_analogy, c4_unique_bigrams_freqs_analogy %>% dplyr::select("Bigram", "Count", "C4Frequency"), by=c("Bigram")) 

str(an_filtering_analogy)

bigrams_by_rating_analogy <- merge(bigrams_by_rating_analogy, 
                           c4_unique_bigrams_freqs_analogy %>% dplyr::select("Bigram", "C4Frequency", "Count"), 
                           all.x=TRUE,
                           by=c("Bigram"))

str(bigrams_by_rating_analogy)

bigrams_by_rating_analogy <- bigrams_by_rating_analogy %>%
  mutate(QuartileFrequency = factor(
    ifelse(C4Frequency == "Zero", "Zero",
           ifelse(C4Frequency %in% c("Near-Zero (1-3)", "Below 10th percentile", "10th-25th percentile"), "Bottom quartile", 
                  ifelse(C4Frequency == "25th-50th percentile", "1st-2nd quartile",
                         ifelse(C4Frequency == "50th-75th percentile", "2nd-3rd quartile",
                                "Top quartile")
                  ))), levels = c("Zero", "Bottom quartile", "1st-2nd quartile", "2nd-3rd quartile", "Top quartile")))

## Explore frequencies ----

bigrams_by_adj_freq_analogy <- bigrams_by_rating_analogy %>%
  group_by(Adjective, QuartileFrequency, SenseRating) %>%
  summarize(Count=n())

bigram_count_by_adj_analogy <- bigrams_by_rating_analogy %>%
  group_by(Adjective, SenseRating) %>%
  summarize(Count=n())

bigram_count_by_freq_analogy <- bigrams_by_rating_analogy %>%
  group_by(QuartileFrequency, SenseRating) %>%
  summarize(Count=n())

table(bigrams_by_rating_analogy[, c("QuartileFrequency", "SenseRating")])

ggplot(an_filtering_analogy, aes(x=Rating)) +
  geom_bar() +
  ylab("Number of ratings at each value") +
  ggtitle("Makes sense rating vs. corpus frequency") +
  facet_wrap(~C4Frequency)

ggplot(bigram_count_by_freq_analogy %>%
         mutate(SenseRating = fct_relevel(SenseRating, "Nonsensical")),
       aes(x=QuartileFrequency, y=Count)) +
  geom_col(position="fill", aes(fill=SenseRating)) +
  xlab("Frequency") +
  ylab("Proportion of unique bigrams") +
  labs(fill='Rating category') + 
  guides(x = guide_axis(angle = 90)) + 
  scale_x_discrete(drop=FALSE)
ggsave("plots/analogy_filtering_frequency_sense_proportions.png", units='in', width=4, height=4, dpi=300)

ggplot(bigram_count_by_adj_analogy %>%
         mutate(SenseRating = fct_relevel(SenseRating, "Nonsensical")),
       aes(x=Adjective, y=Count)) +
  geom_col(position="fill", aes(fill=SenseRating)) +
  xlab("Adjective") +
  ylab("Proportion of unique bigrams") +
  labs(fill='Rating category') + 
  guides(x = guide_axis(angle = 90)) + 
  scale_x_discrete(drop=FALSE)
ggsave("plots/analogy_filtering_adjective_sense_proportions.png", units='in', width=4, height=4, dpi=300)

ggplot(bigrams_by_adj_freq_analogy,
       aes(x=QuartileFrequency,y=Count, fill=Adjective)) +
  geom_col(position="stack") +
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ SenseRating)  +
  xlab("Frequency") +
  ylab("Unique Bigram Count") +
  scale_x_discrete(drop=FALSE) +
  scale_fill_paletteer_d("rcartocolor::Prism")  # pals::stepped, pals::tol
ggsave("plots/analogy_filtering_frequency_sense_by_adjective.png", units='in', width=6, height=4)

ggplot(bigrams_by_adj_freq_analogy  %>%
         mutate(QuartileFrequency = fct_rev(QuartileFrequency)),
       aes(x=Adjective,y=Count, fill=QuartileFrequency)) +
  geom_col(position="stack") +
  guides(x = guide_axis(angle = 90)) +
  xlab("Adjective") +
  ylab("Unique Bigram Count") +
  labs(fill='Frequency') +
  scale_fill_paletteer_d("rcartocolor::Prism", drop=FALSE)  # pals::stepped, pals::tol
ggsave("plots/analogy_filtering_frequency_by_adjective.png", units='in', width=6, height=4)

# Combine with original bigrams ----

## Combine ----

bigrams_by_rating_combined <- bind_rows(bigrams_by_rating,
                                        bigrams_by_rating_analogy)
str(bigrams_by_rating_combined)

bigrams_by_adj_freq_combined <- bigrams_by_rating_combined %>%
  filter(Adjective %in% intersective_as | Adjective %in% privative_as) %>%
  group_by(Adjective, QuartileFrequency, SenseRating) %>%
  summarize(Count=n())

bigram_count_by_adj_combined <- bigrams_by_rating_combined %>%
  filter(Adjective %in% intersective_as | Adjective %in% privative_as) %>%
  group_by(Adjective, SenseRating) %>%
  summarize(Count=n())

bigram_count_by_freq_combined <- bigrams_by_rating_combined %>%
  filter(Adjective %in% intersective_as | Adjective %in% privative_as) %>%
  group_by(QuartileFrequency, SenseRating) %>%
  summarize(Count=n())

write.csv(bigrams_by_rating_combined %>% 
            select(Bigram, Adjective, Noun, VeryHard, SomewhatHard, SomewhatEasy, VeryEasy,
                   AdjectiveClass, SenseRating, Count, C4Frequency, QuartileFrequency), 
          "filtering_bigrams_with_ratings_combined.csv", row.names = FALSE)

## Plot ----

ggplot(bigram_count_by_freq_combined %>%
         filter(SenseRating != "Excluded") %>%
         filter(!is.na(QuartileFrequency)) %>%
         mutate(SenseRating = fct_relevel(SenseRating, "Nonsensical")),
       aes(x=QuartileFrequency, y=Count)) +
  geom_col(position="fill", aes(fill=SenseRating)) +
  xlab("Frequency") +
  ylab("Proportion of unique bigrams") +
  labs(fill='Rating category') + 
  guides(x = guide_axis(angle = 90)) + 
  scale_x_discrete(drop=FALSE)
ggsave("plots/filtering_combined_frequency_sense_proportions.png", units='in', width=4, height=4, dpi=300)

ggplot(bigrams_by_adj_freq_combined  %>%
         filter(SenseRating != "Excluded"),
       aes(x=QuartileFrequency,y=Count, fill=Adjective)) +
  geom_col(position="stack") +
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ SenseRating)  +
  xlab("Frequency") +
  ylab("Unique Bigram Count") +
  scale_x_discrete(drop=FALSE) +
  scale_fill_paletteer_d("rcartocolor::Prism")  # pals::stepped, pals::tol
ggsave("plots/filtering_combined_frequency_sense_by_adjective.png", units='in', width=6, height=4)

