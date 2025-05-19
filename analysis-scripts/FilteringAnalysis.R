# Packages ----

library(ordinal)
library(MASS)
library(effects)
library(tidyverse)
library(tidytext)
library(paletteer)

# Import data ----

filtering_wide <- read.csv("results/human/exp1_filtering/filtering_responses_raw.csv",
                           fileEncoding = "UTF-8-BOM")


# Remove the first two rows, which contain the header again and the full question text
filtering_wide <- filtering_wide[-c(1,2),]
rownames(filtering_wide) <- NULL

ncol(filtering_wide)
nrow(filtering_wide)

# Remove the person who returned their survey and always clicked "Somewhat easy"
filtering_wide <- filtering_wide %>% filter(ProlificID != "60ddc7abdf6661b9c294691f")

# Select only question columns; drop all user information & comments except demographic questionnaire
filtering_wide <- subset(filtering_wide, 
                         select = grep("^(Q.|Q1|Q2|Q3|Q4)", 
                                       names(filtering_wide)))

# Add fake UserID (don't want to use actual Prolific ID)
filtering_wide$UserId <- 1:nrow(filtering_wide)

# Rename demographic questions
filtering_wide <- filtering_wide %>%
  rename(EnglishBefore5 = Q2, Dialect = Q3, OtherEnglish = Q3_2_TEXT, OtherLanguages = Q4, Comments = Q5)

# Exclusion criteria ----

dem_excluded_ids <- filtering_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & OtherEnglish != "American English")) %>% 
  pull(UserId)

dem_excluded_ids

demographics <- filtering_wide %>%
  dplyr::select(UserId, EnglishBefore5, Dialect, OtherEnglish, OtherLanguages) %>%
  filter(UserId %in% dem_excluded_ids)

filtering_wide <- filtering_wide %>%
  mutate(AttnFailed = ifelse(Q.red.circle %in% c("Very hard", "Somewhat hard"), 1, 0)
  )

attn_excluded_ids <- filtering_wide %>%
  filter(AttnFailed >= 1) %>%
  pull(UserId)

attn_excluded_ids

filtering_wide_excl <- filtering_wide %>%
  filter(!UserId %in% dem_excluded_ids & !UserId %in% attn_excluded_ids)

nrow(filtering_wide_excl)

## Select target questions and pivot ----

names(filtering_wide_excl)[1:10]
names(filtering_wide_excl)[(ncol(filtering_wide_excl)-10):(ncol(filtering_wide_excl)-1)]

# Append "MS" to all non-follow-up questions so that we can pivot
#filtering_wide_excl <- filtering_wide_excl %>%
#  rename_at(vars( starts_with("Q") & !ends_with("FUP") ), list( ~paste0(., ".MS") ) )

filtered_filtering_wide <- filtering_wide_excl %>%
  dplyr::select(!names(filtering_wide_excl)[1:8]) %>%  # Exclude training
  dplyr::select(!c("AttnFailed")) %>%
  dplyr::select(!EnglishBefore5:Comments) %>% # Exclude demographics
  dplyr::select(!ends_with("FUP"))

names(filtered_filtering_wide)[1:10]
names(filtered_filtering_wide)[(ncol(filtered_filtering_wide)-10):(ncol(filtered_filtering_wide)-1)]

an_filtering <- filtering_wide_excl %>%
  dplyr::select(!names(filtering_wide_excl)[1:8]) %>%  # Exclude training
  dplyr::select(!c("AttnFailed")) %>%
  dplyr::select(!EnglishBefore5:Comments) %>% # Exclude demographics
  dplyr::select(!ends_with("FUP")) %>%
  pivot_longer(
    cols = Q.fake.couple:Q.multicolored.fact,
    names_to = c("Adjective","Noun"),
    names_pattern = "Q.([a-z]+).([a-z]+)",
    values_to = "Rating"
  ) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate_at(c("UserId", "Adjective", "Noun", "Bigram"), factor)

an_homophony <- filtering_wide_excl %>%
  dplyr::select(!names(filtering_wide_excl)[1:8]) %>%  # Exclude training
  dplyr::select(!c("AttnFailed")) %>%
  dplyr::select(!EnglishBefore5:Comments) %>% # Exclude demographics
  dplyr::select(ends_with("FUP") | ends_with("UserId")) %>%
  pivot_longer(
    cols = Q.fake.couple.FUP:Q.multicolored.fact.FUP,
    names_to = c("Adjective","Noun"),
    names_pattern = "Q.([a-z]+).([a-z]+)",
    values_to = "HomophonyRating"
  ) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate_at(c("UserId", "Adjective", "Noun", "Bigram"), factor)

str(an_filtering)
str(an_homophony)

#an_filtering <- an_filtering %>%
#  mutate(HomophonyRating = an_homophony$HomophonyRating)

an_filtering <- merge(an_filtering, an_homophony, by=c("UserId", "Bigram", "Adjective", "Noun")) %>%
  arrange(UserId)  

str(an_filtering)

# Drop rows with no data (no rating)
an_filtering <- an_filtering %>%
  filter(Rating!="" & HomophonyRating !="") %>%
  mutate(Rating = factor(Rating, levels = c("Very hard", "Somewhat hard", "Somewhat easy", "Very easy"))) %>%
  mutate(HomophonyRating = factor(HomophonyRating, levels = c("No", "Yes"))) %>%
  mutate(NumRating = as.integer(Rating)) %>%
  mutate(NumHomophonyRating = as.integer(HomophonyRating))

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
attn_check_intersective_as = c(
  "red",
  "purple"
)

an_filtering <- an_filtering %>%
  mutate(AdjectiveClass = ifelse(Adjective %in% privative_as, "Privative", 
                                 ifelse(Adjective %in% intersective_as, "Intersective", "AttentionCheck"))) %>%
  mutate_at(c("AdjectiveClass"), factor)

str(an_filtering)

# Plot ----

## Balance ----

an_filtering %>% 
  filter(AdjectiveClass != "AttentionCheck") %>%
  group_by(Bigram) %>%
  summarise(RatingCount = n()) %>%
  filter(RatingCount != 3) %>%
  print(n = 50)

# Blocks 43 and 46 have too many ratings; it's fine

## Broad picture of makes sense ratings ----

ggplot(an_filtering, aes(x=Rating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Bigram 'makes sense' ratings for all bigrams")

ggplot(an_filtering %>% filter(AdjectiveClass == "Intersective"), aes(x=Rating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Bigram 'makes sense' ratings for intersective bigrams")

ggplot(an_filtering %>% filter(AdjectiveClass == "Privative"), aes(x=Rating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Bigram 'makes sense' ratings for privative bigrams")

ggplot(an_filtering %>% filter(AdjectiveClass == "AttentionCheck"), aes(x=Rating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Bigram 'makes sense' ratings for attention check bigrams") + 
  facet_wrap(~ Bigram)

ggplot(an_filtering %>% filter(AdjectiveClass == "Privative"), aes(x=Rating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Bigram 'makes sense' ratings for privative bigrams") + 
  facet_wrap(~ Adjective)

ggplot(an_filtering %>% filter(AdjectiveClass == "Intersective"), aes(x=Rating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Bigram 'makes sense' ratings for intersective bigrams") + 
  facet_wrap(~ Adjective)

## Specific adjectives ----

ggplot(an_filtering %>% filter(AdjectiveClass == "Privative"), 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating,color=Noun)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for whether AN makes sense") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "filtering_plot_privative_bigrams_by_adjective.png", width = 2000, height = 800)
dev.off()

ggplot(an_filtering %>% filter(AdjectiveClass == "Intersective" & Adjective != "red"), 
       aes(x=reorder_within(x=Noun,by=NumRating,within=Adjective),
           y=NumRating,color=Noun)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for whether AN makes sense") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "filtering_plot_intersective_bigrams_by_adjective.png", width = 2000, height = 800)
dev.off()

## Homophone nouns ----

ggplot(an_filtering %>% filter(AdjectiveClass != "AttentionCheck"), aes(x=Noun,y=NumHomophonyRating)) +
  geom_bar(stat="identity") +
  ggtitle("Homophone ratings for nouns") +
  guides(x =  guide_axis(angle = 90)) 

# Conclusion: the homophone question was not very useful, too much noise

# Filter bigrams ----

an_filtering %>%
  distinct(Bigram) %>%
  summarize(n())

excluded_nouns = read.table("../Adjectives-PythonCode/data/excluded_nouns.txt", col.names = c("Noun"))

bigrams_by_rating <- an_filtering %>%
  mutate(Rating = fct_recode(Rating,
                             VeryEasy = "Very easy",
                             SomewhatEasy = "Somewhat easy", 
                             SomewhatHard = "Somewhat hard", 
                             VeryHard = "Very hard")) %>%
  group_by(Bigram, Rating) %>%
  summarise(RatingCount = length(Rating)) %>%
  pivot_wider(
    names_from = Rating,
    values_from = RatingCount
  ) %>%
  replace(is.na(.), 0) %>%
  separate(Bigram, into = c("Adjective", "Noun"), remove = FALSE) %>%
  mutate(AdjectiveClass = ifelse(Adjective %in% privative_as, "Privative", 
                                 ifelse(Adjective %in% intersective_as, "Intersective", "AttentionCheck"))) %>%
  mutate_at(c("AdjectiveClass", "Adjective", "Noun"), factor)

bigrams_by_rating <- bigrams_by_rating %>%
  mutate(SenseRating = factor(ifelse(Noun %in% excluded_nouns$Noun, "Excluded", 
                              ifelse(SomewhatEasy + VeryEasy >= 2, "Makes sense", "Nonsensical"))))

sensical_bigrams <- bigrams_by_rating %>%
  filter(!Noun %in% excluded_nouns$Noun) %>%
  filter(SomewhatEasy + VeryEasy >= 2) %>%
  dplyr::select(Bigram, Adjective, Noun, AdjectiveClass)

nrow(bigrams_by_rating %>%
       filter(SenseRating == "Makes sense"))

write.table(sensical_bigrams %>% dplyr::select(Bigram), "sensical_bigrams.txt", row.names=FALSE,  col.names = FALSE, quote = FALSE)

nonsensical_bigrams <- bigrams_by_rating %>%
  filter(SomewhatEasy + VeryEasy < 2) %>%
  dplyr::select(Bigram, Adjective, Noun, AdjectiveClass)

nrow(nonsensical_bigrams)

nrow(bigrams_by_rating %>%
       filter(SenseRating == "Nonsensical"))

write.table(nonsensical_bigrams %>% dplyr::select(Bigram), "nonsensical_bigrams.txt", row.names=FALSE,  col.names = FALSE, quote = FALSE)

easy_bigrams <- bigrams_by_rating %>%
  filter(!Noun %in% excluded_nouns$Noun) %>%
  filter(VeryEasy >= 2 & SomewhatHard == 0 & VeryHard == 0) %>%
  dplyr::select(Bigram, Adjective, Noun, AdjectiveClass)

easy_bigrams

write.table(easy_bigrams %>% dplyr::select(Bigram), "easy_bigrams.txt", row.names=FALSE,  col.names = FALSE, quote = FALSE)

excluded_sensical_bigrams <- bigrams_by_rating %>%
  filter(Noun %in% excluded_nouns$Noun) %>%
  filter(SomewhatEasy + VeryEasy >= 2) %>%
  dplyr::select(Bigram, Adjective, Noun, AdjectiveClass)

excluded_sensical_bigrams

write.table(excluded_sensical_bigrams %>% dplyr::select(Bigram), "excluded_sensical_bigrams.txt", row.names=FALSE,  col.names = FALSE, quote = FALSE)

# Frequencies ----

## Import original frequencies ----

bigrams_freqs <- read.csv("../Adjectives-PythonCode/output/filtering_data/filtered_bigrams_with_frequencies.csv", header = TRUE)

unique_bigrams_freqs <- bigrams_freqs %>%
  filter(MakesSense != "Easy (majority very easy; no somewhat/very hard)") %>%  # Easy is a subset of makes sense
  mutate_at(c("Adjective", "Noun", "Bigram", "MakesSense", "AdjectiveClass", "Frequency"), factor) %>%
  mutate(Frequency = factor(Frequency, levels = c("High", "Medium", "Zero")))

## Import full 3336 bigrams over C4 frequencies

c4_bigrams_freqs <- read.csv("../Adjectives-PythonCode/output/filtering_data/all_c4_3912_bigrams_with_frequencies.csv", header = TRUE)

c4_unique_bigrams_freqs <- c4_bigrams_freqs %>%
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
  rename(C4Frequency = Frequency)

str(c4_unique_bigrams_freqs)

# Merge counts with observations

an_filtering <- merge(an_filtering, unique_bigrams_freqs %>% dplyr::select("Bigram", "Count", "Frequency"), by=c("Bigram")) 

an_filtering <- merge(an_filtering, c4_unique_bigrams_freqs %>% dplyr::select("Bigram", "Count", "C4Frequency"), by=c("Bigram")) 

str(an_filtering)

write.csv(an_filtering %>% dplyr::select(-one_of(c('NumRating', 'NumHomophonyRating'))), 
          "filtering_experiment_processed_data_with_corpus_counts.csv", 
          row.names = FALSE,
          fileEncoding = "UTF-8")

bigrams_by_rating <- merge(bigrams_by_rating, 
                           c4_unique_bigrams_freqs %>% dplyr::select("Bigram", "C4Frequency", "Count"), 
                           all.x=TRUE,
                           by=c("Bigram"))

str(bigrams_by_rating)

bigrams_by_rating <- bigrams_by_rating %>%
  mutate(QuartileFrequency = factor(
    ifelse(C4Frequency == "Zero", "Zero",
    ifelse(C4Frequency %in% c("Near-Zero (1-3)", "Below 10th percentile", "10th-25th percentile"), "Bottom quartile", 
    ifelse(C4Frequency == "25th-50th percentile", "1st-2nd quartile",
    ifelse(C4Frequency == "50th-75th percentile", "2nd-3rd quartile",
    "Top quartile")
    ))), levels = c("Zero", "Bottom quartile", "1st-2nd quartile", "2nd-3rd quartile", "Top quartile")))

## Explore frequencies ----

table(bigrams_by_rating[, c("QuartileFrequency")])

table(bigrams_by_rating[, c("QuartileFrequency", "SenseRating")])

bigrams_by_rating %>%
  filter(QuartileFrequency == "Top quartile" & SenseRating == "Nonsensical") %>%
  select(Bigram, C4Frequency)

## Plot frequency / counts ----

ggplot(an_filtering, aes(y=Count, x=Rating)) +
  geom_boxplot() +
  ggtitle("Makes sense rating vs. corpus frequency")

# Can't plot zero values when using a log scale
ggplot(an_filtering %>% filter(Frequency != "Zero"), aes(y=Count, x=Rating)) +
  geom_boxplot() +
  scale_y_continuous(trans="log10") +
  ggtitle("Makes sense rating vs. corpus frequency")

ggplot(an_filtering, aes(x=Rating)) +
  geom_bar() +
  ylab("Number of ratings at each value") +
  ggtitle("Makes sense rating vs. corpus frequency") +
  facet_wrap(~C4Frequency)
dev.copy(png, "rating_frequency_histogram.png", width = 900, height = 400)
dev.off()

## Plot inclusion/exclusion/sense rating by frequency

# TODO plot proportion of adjectives filtered out for not making sense intelligently as a function of adjective and frequency

bigrams_by_rating %>%
  filter(is.na(C4Frequency)) %>%
  select(Bigram)

bigrams_by_rating %>%
  filter(Adjective != "red") %>%
  filter(!(Noun %in% excluded_nouns$Noun)) %>%
  distinct(Noun)

bigrams_by_adj_freq <- bigrams_by_rating %>%
  filter(Adjective != "red") %>%
  filter(!(Noun %in% excluded_nouns$Noun)) %>%
  group_by(Adjective, QuartileFrequency, SenseRating) %>%
  summarize(Count=n())

bigram_count_by_adj <- bigrams_by_rating %>%
  filter(Adjective != "red") %>%
  filter(!(Noun %in% excluded_nouns$Noun)) %>%
  group_by(Adjective, SenseRating) %>%
  summarize(Count=n())

bigram_count_by_freq <- bigrams_by_rating %>%
  group_by(QuartileFrequency, SenseRating) %>%
  summarize(Count=n())

# TODO update with counts for all items
ggplot(bigram_count_by_freq %>%
         filter(SenseRating != "Excluded") %>%
         mutate(SenseRating = fct_relevel(SenseRating, "Nonsensical")), 
       aes(x=C4Frequency, y=Count)) +
  geom_col(position="stack", aes(fill=SenseRating)) +
  xlab("Frequency") +
  ylab("Unique Bigram Count")

ggplot(bigram_count_by_freq %>%
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
ggsave("plots/filtering_frequency_sense_proportions.png", units='in', width=4, height=4, dpi=300)

ggplot(bigram_count_by_adj %>%
         mutate(SenseRating = fct_relevel(SenseRating, "Nonsensical")), 
       aes(x=Adjective, y=Count, fill=SenseRating)) +
  geom_col(position="stack") +
  ylab("Unique Bigram Count")

ggplot(bigram_count_by_adj %>%
         mutate(SenseRating = fct_relevel(SenseRating, "Nonsensical")), 
       aes(x=Adjective, y=Count, fill=SenseRating)) +
  geom_col(position="stack") +
  ylab("Unique Bigram Count")

ggplot(bigrams_by_rating %>%
         mutate(SenseRating = fct_relevel(SenseRating, "Nonsensical")) %>%
         filter(SenseRating != "Excluded"), 
       aes(x=Adjective,y=SenseRating,col=C4Frequency)) +
  geom_jitter()

ggplot(bigrams_by_adj_freq  %>%
         filter(SenseRating != "Excluded"),
       aes(x=QuartileFrequency,y=Count, fill=Adjective)) +
  geom_col(position="stack") +
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ SenseRating)  +
  xlab("Frequency") +
  ylab("Unique Bigram Count") +
  scale_x_discrete(drop=FALSE) +
  scale_fill_paletteer_d("rcartocolor::Prism")  # pals::stepped, pals::tol
ggsave("plots/filtering_frequency_sense_by_adjective.png", units='in', width=6, height=4)

# https://github.com/EmilHvitfeldt/r-color-palettes

ggplot(bigrams_by_adj_freq  %>%
         filter(SenseRating != "Excluded") %>%
         mutate(QuartileFrequency = fct_rev(QuartileFrequency)),
         #filter(!is.na(C4Frequency)) %>%
         # mutate(C4Frequency = fct_recode(C4Frequency,
         #                                 "Below 10th pct." = "Below 10th percentile",
         #                                 "10th-25th pct." = "10th-25th percentile",
         #                                 "25th-50th pct." = "25th-50th percentile",
         #                                 "50th-75th pct." = "50th-75th percentile",
         #                                 "75th-90th pct." = "75th-90th percentile",
         #                                 "90th-95th pct." = "90th-95th percentile",
         #                                 "95th-99th pct." = "95th-99th percentile",
         #                                 "99th pct." = "99th percentile"
         # )), 
       aes(x=Adjective,y=Count, fill=QuartileFrequency)) +
  geom_col(position="stack") +
  guides(x = guide_axis(angle = 90)) +
  xlab("Adjective") +
  ylab("Unique Bigram Count") +
  labs(fill='Frequency') +
  scale_fill_paletteer_d("rcartocolor::Prism", drop=FALSE)  # pals::stepped, pals::tol
ggsave("plots/filtering_frequency_by_adjective.png", units='in', width=6, height=4)

# Fit frequency/count to ease ----

rating_freq_lm <- clmm(Rating ~ Frequency + (1 | UserId), data = an_filtering, link = "logit")

# Unsurprisingly there's a correlation between zero and not making sense, as phrases that don't make sense won't appear in the corpus
summary(rating_freq_lm)

plot(allEffects(rating_freq_lm), style="stacked", colors=rev(hcl.colors(4, palette="TealRose")))
dev.copy(png, "rating_frequency.png", width = 500, height = 500)
dev.off()

# This doesn't fit :(
rating_freq_lm_inv <- clmm(Frequency ~ Rating + (1 | UserId), data = an_filtering, link = "logit")
