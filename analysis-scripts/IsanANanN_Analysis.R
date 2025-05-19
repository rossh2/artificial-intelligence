library(lmerTest)
library(effects)
library(tidyverse)
library(tidytext)
library(yarrr)
#library(extrafont)
#font_import()
#loadfonts(device = "win")

source("Analysis_Utils.R")

# Load and prepare data ----

## Parse responses ----

isa_csv <- read.csv("results/human/exp2_isa/Pilot_Run1_Run2_Run3_combined_raw_results.csv")

isa_csv <- isa_csv %>%
  rename(Group = group, Item = item, Question = question) %>%
  mutate(ParticipantId = PROLIFIC) %>%
  # Turn Participant ID into numbers from 1 to n instead of Prolific IDs
  mutate_at(c("ParticipantId"), factor) %>%
  mutate_at(c("ParticipantId"), as.integer) %>%  
  mutate_at(c("ParticipantId"), factor) %>%
  dplyr::select(-PROLIFIC)


isa_data_all <- isa_csv %>% filter(Group != "DEM" & Group != "T") %>%
  mutate_at(c("Group", "Item"), as.numeric) %>%
  mutate_at(c("Group", "Item", "Question"), factor) %>%
  mutate(Value = fct_relevel(Value, "Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes")) %>%
  rename(Bigram = Question, IsaRating = Value) %>%
  mutate(NumIsaRating = as.integer(IsaRating)) %>%
  separate(Bigram, into = c("Adjective", "Noun"), remove = FALSE) %>%
  mutate_at(c("Adjective", "Noun"), factor)

## Merge additional data ----

isa_data_all <- add_frequency(isa_data_all)

str(isa_data_all)

## Filter participants ----

dem_excluded_ids <- isa_csv %>%
  filter(
    (Question == "native-speaker" & Value != "Yes") 
    | (Question == "dialect" & Value != "Yes%2C I speak American English")
  ) %>% 
  pull(ParticipantId)

length(dem_excluded_ids)

fillers = c("wooden pear", "orange fish", "knitted pizza", "temporary breakage")

# Use filler question to see which group the participant got assigned to
isa_csv %>% filter(ParticipantId %in% dem_excluded_ids & (Group == "DEM" | Question == "wooden pear"))

strict_attn_excluded_ids <- isa_csv %>%
  filter(
    (Question == "wooden pear" & !(Value %in% c("Definitely not", "Probably not", "Unsure")))
    | (Question == "orange fish" & !(Value %in% c("Definitely yes", "Probably yes")))
  ) %>% 
  pull(ParticipantId)

attn_excluded_ids <- isa_csv %>%
  filter(
    (Question == "wooden pear" & !(Value %in% c("Definitely not", "Probably not", "Unsure")))
  ) %>% 
  pull(ParticipantId)

length(strict_attn_excluded_ids)
length(attn_excluded_ids)

ggplot(isa_data_all %>% filter(Bigram == "wooden pear"), aes(x=IsaRating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Responses for 'Is a wooden pear still edible?'")

ggplot(isa_data_all %>% filter(Bigram == "orange fish"), aes(x=IsaRating)) +
  geom_bar() +
  ylab("Count") +
  ggtitle("Responses for 'Is an orange fish still an animal?'")

isa_csv %>% filter(ParticipantId %in% strict_attn_excluded_ids & (Bigram %in% c("wooden pear", "orange fish")))

# Remove filtered participants
isa_data <- isa_data_all %>% 
  filter(!(ParticipantId %in% dem_excluded_ids) & !(ParticipantId %in% attn_excluded_ids))

str(isa_data)

### Participant / Latin square statistics ----

isa_data_all %>%
  filter(Bigram == "wooden pear") %>%
  summarise(ResponseCount = n(), ParticipantCount = n_distinct(ParticipantId))

isa_data %>%
  filter(Bigram == "wooden pear") %>%
  summarise(FilteredResponseCount = n(), FilteredParticipantCount = n_distinct(ParticipantId))

min_participants_per_group = 10
max_participants_per_group = 12

group_participant_counts <- isa_data %>%
  group_by(Group) %>%
  summarise(ParticipantCount = n_distinct(ParticipantId))

group_participant_counts %>%
  filter(ParticipantCount < min_participants_per_group) %>%
  print(n = Inf)

group_participant_counts %>%
  filter(ParticipantCount > max_participants_per_group) %>%
  print(n = Inf)

table(group_participant_counts[, c('ParticipantCount')])

ggplot(group_participant_counts, aes(x=Group, y=ParticipantCount)) +
  geom_bar(stat = "identity") + 
  ylab("Count") +
  ggtitle("Participants per Latin square group")
dev.copy(png, "isa_plot_participants_per_group_allruns.png", width = 1200, height = 400, res=180)
dev.off()

## Remove fillers ----

isa_data <- isa_data %>%
  filter(!(Bigram %in% fillers)) %>%
  mutate(AdjectiveClass = droplevels(AdjectiveClass), Adjective = droplevels(Adjective), Noun = droplevels(Noun),
         Bigram = droplevels(Bigram), 
         Frequency = droplevels(Frequency), CoarseFrequency = droplevels(CoarseFrequency))

str(isa_data)

# Calculate variance / SD ----

set.seed(42)
isa_data_capped <- isa_data %>%
  group_by(Bigram) %>%
  slice_sample(n = max_participants_per_group) %>%
  ungroup()

isa_variance <- isa_data %>%
  group_by(Bigram) %>%
  summarise(Variance = var(NumIsaRating), Mean = mean(NumIsaRating), SD = sd(NumIsaRating),
            Adjective = unique(Adjective), Noun = unique(Noun), AdjectiveClass = unique(AdjectiveClass),
            Frequency = unique(Frequency), Count = unique(Count), CoarseFrequency = unique(CoarseFrequency))

isa_variance_capped <- isa_data_capped %>%
  group_by(Bigram) %>%
  summarise(Variance = var(NumIsaRating), Mean = mean(NumIsaRating), SD = sd(NumIsaRating),
            Adjective = unique(Adjective), Noun = unique(Noun), AdjectiveClass = unique(AdjectiveClass),
            Frequency = unique(Frequency), Count = unique(Count), CoarseFrequency = unique(CoarseFrequency))

str(isa_variance)

# Make version of variance with zero frequencies set to 1, for log scale plotting

isa_variance_log <- isa_variance %>%
  mutate(Count = if_else(Count == 0, 1, Count))

isa_variance_capped_log <- isa_variance_capped %>%
  mutate(Count = if_else(Count == 0, 1, Count))

# Exclude adjectives outside 12 experiment adjectives ----

exp_adjectives <- read.csv('../Adjectives-PythonCode/data/adjectives.txt', header = FALSE)

isa_data_12 <- isa_data %>%
  filter(Adjective %in% exp_adjectives$V1)

isa_data_12_capped <- isa_data_capped %>%
  filter(Adjective %in% exp_adjectives$V1)

isa_variance_12 <- isa_variance %>%
  filter(Adjective %in% exp_adjectives$V1)

isa_variance_12_capped <- isa_variance_capped %>%
  filter(Adjective %in% exp_adjectives$V1)

isa_variance_12_capped_log <- isa_variance_capped_log %>%
  filter(Adjective %in% exp_adjectives$V1)

write.csv(isa_data_capped %>% select(ParticipantId, Bigram, Adjective, Noun, IsaRating, NumIsaRating, AdjectiveClass, CoarseFrequency, Count), 
          file = 'isa_data_capped.csv', row.names = FALSE)

write.csv(isa_data_12_capped %>% select(ParticipantId, Bigram, Adjective, Noun, IsaRating, NumIsaRating, AdjectiveClass, CoarseFrequency, Count), 
          file = 'isa_data_12_capped.csv', row.names = FALSE)

write.csv(isa_variance_12_capped %>% select(Bigram, Adjective, Noun, Variance, Mean, SD, AdjectiveClass, CoarseFrequency, Count), 
          file = 'isa_variance_12_capped.csv', row.names = FALSE)

extra_isa_data <- isa_data %>%
  filter(!(Adjective %in% exp_adjectives$V1))


# Tables ----

# Use isa_variance since it has one row per bigram

table(isa_variance[, c("Frequency")])

table(isa_variance[, c("Frequency", "AdjectiveClass")])

nrow(isa_variance_12)

# Values ----

context_bigrams = c(
  "counterfeit diamond", "counterfeit dollar", 
  "fake reef", "fake fire", 
  "fake scarf", "fake drug",
  "fake glance", "fake plan",
  "false concert", "false war",
  "former accusation", "former house"
)

isa_variance_context <- isa_variance_12_capped %>% filter(Bigram %in% context_bigrams) %>%
  dplyr::select(Bigram, Mean, Variance)

isa_variance_context

write.csv(isa_variance_context, "isa_variance_context_bigrams.csv", row.names = FALSE)

# Plots ----

## Histogram of frequencies ----

ggplot(isa_variance, aes(x=Frequency)) +
  geom_bar() +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Bigrams by frequency (percentiles)")

ggplot(isa_variance, aes(x=Frequency)) +
  geom_bar(aes(fill=AdjectiveClass)) +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Bigrams by frequency (percentiles)") +
  facet_wrap(~AdjectiveClass)

ggplot(isa_variance, aes(x=Frequency)) +
  geom_bar(aes(fill=AdjectiveClass)) +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Bigrams by frequency (percentiles)") +
  facet_wrap(~Adjective)

ggplot(isa_variance, aes(x=Frequency)) +
  geom_bar() +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Bigrams by frequency (percentiles)") +
  facet_wrap(~Noun)
  
## Rating (likelihood of being an N) vs. adjective / adjective class ----

ggplot(isa_data, aes(x=AdjectiveClass, y=NumIsaRating)) +
  geom_boxplot(aes(color=AdjectiveClass)) +
  ggtitle("Ratings for 'Is an AN an N?' by adjective class")

ggplot(isa_data, aes(x=Adjective, y=NumIsaRating)) +
  geom_boxplot(aes(color=AdjectiveClass)) +
  ggtitle("Ratings for 'Is an AN an N?' by adjective")

ggplot(isa_data, aes(x=IsaRating)) +
  geom_bar(aes(fill=AdjectiveClass)) + 
  ggtitle("Ratings for 'Is an AN an N?' by adjective") +
  facet_wrap(~ Adjective)

### With capped data ----

ggplot(isa_data_capped, aes(x=AdjectiveClass, y=NumIsaRating)) +
  geom_boxplot(aes(color=AdjectiveClass)) +
  ggtitle(sprintf("Ratings for 'Is an AN an N?' by adjective class (max. %s ratings/bigram)", max_participants_per_group))

ggplot(isa_data_12_capped, aes(x=Adjective, y=NumIsaRating)) +
  geom_boxplot(aes(color=AdjectiveClass)) +
  ggtitle(sprintf("Ratings for 'Is an AN an N?' by adjective (max. %s ratings/bigram)", max_participants_per_group))
dev.copy(png, "isa_plot_rating_adjective_box.png", width = 600, height = 400)
dev.off()

ggplot(isa_data_12_capped, aes(x=IsaRating)) +
  geom_bar(aes(fill=AdjectiveClass)) + 
  ggtitle(sprintf("Ratings for 'Is an AN an N?' by adjective (max. %s ratings/bigram)", max_participants_per_group)) +
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ Adjective)
dev.copy(png, "isa_plot_rating_adjective_histogram.png", width = 600, height = 400)
dev.off()

ggplot(isa_data_capped, aes(x=IsaRating)) +
  geom_bar(aes(fill=AdjectiveClass)) + 
  ggtitle(sprintf("Ratings for 'Is an AN an N?' by adjective & frequency (max. %s ratings/bigram)", max_participants_per_group)) +
  ylab("Rating") +
  xlab("Count") +
  guides(x = guide_axis(angle = 90)) +
  facet_grid(rows=vars(Frequency),cols=vars(Adjective)) 
dev.copy(png, "isa_plot_rating_adjective_freq_histogram.png", width = 1200, height = 800)
dev.off()

## Ratings vs. adjective and noun ----

### Josh-style scatter plots ----

ggplot(isa_data_12_capped, 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating,color=Noun)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle(sprintf("Ratings for 'Is an AN an N?' (max. %s ratings/bigram)", max_participants_per_group)) + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "isa_plot_all_bigrams_by_adjective.png", width = 2000, height = 800)
dev.off()

ggplot(isa_data_12_capped, 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle(sprintf("Ratings for 'Is an AN an N?' (max. %s ratings/bigram)", max_participants_per_group)) + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "isa_plot_all_bigrams_by_adjective_with_frequency.png", width = 6000, height = 2500, res=300)
dev.off()

ggplot(isa_data_12_capped %>%
         mutate(Frequency = fct_recode(Frequency,
                                       "Below 10th pct." = "Below 10th percentile",
                                       "10th-25th pct." = "10th-25th percentile",
                                       "25th-50th pct." = "25th-50th percentile",
                                       "50th-75th pct." = "50th-75th percentile",
                                       "75th-90th pct." = "75th-90th percentile",
                                       "90th-95th pct." = "90th-95th percentile",
                                       "95th-99th pct." = "95th-99th percentile",
                                       "99th pct." = "99th percentile"
         )) %>%
         mutate(Adjective = fct_relevel(Adjective, "artificial", "counterfeit", "fake", "false", "former", "knockoff")), 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2, size=0.5) + 
  stat_summary(fun=mean, geom="point", size=1.5, shape=15) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x", ncol=3) +
  ggtitle("Ratings for 'Is an AN an N?'") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  theme(legend.position="bottom") + 
  theme(axis.text.x = element_text(size = 6))
# Needs to fit on full letter page (8.5"x11") minus 1" margins minus about 0.75" for the caption
ggsave(filename="plots/isa_plot_all_bigrams_by_adjective_with_frequency_tall.png", units="in", width = 6.5, height = 8, dpi=300)

ggplot(isa_data_12_capped %>% filter(AdjectiveClass == "Privative"), 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle(sprintf("Ratings for 'Is an AN an N?' (max. %s ratings/bigram)", max_participants_per_group)) + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "isa_plot_privative_bigrams_by_adjective_with_frequency.png", width = 4000, height = 1600, res=300)
dev.off()

ggplot(isa_data_12_capped %>% filter(AdjectiveClass == "Subsective"), 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle(sprintf("Ratings for 'Is an AN an N?' (max. %s ratings/bigram)", max_participants_per_group)) + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "isa_plot_subsective_bigrams_by_adjective_with_frequency.png", width = 4000, height = 1600, res=300)
dev.off()

ggplot(isa_data_capped %>% filter(AdjectiveClass %in% c("Intersective", "Subsective")), 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle(sprintf("Ratings for 'Is an AN an N?' (max. %s ratings/bigram)", max_participants_per_group)) + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))

adj <- "counterfeit"
adj <- "knockoff"
adj <- "fake"
ggplot(isa_data_capped %>% filter(Adjective == adj) %>%
         mutate(Frequency = fct_recode(Frequency,
                                       "Below 10th pct." = "Below 10th percentile",
                                       "10th-25th pct." = "10th-25th percentile",
                                       "25th-50th pct." = "25th-50th percentile",
                                       "50th-75th pct." = "50th-75th percentile",
                                       "75th-90th pct." = "75th-90th percentile",
                                       "90th-95th pct." = "90th-95th percentile",
                                       "95th-99th pct." = "95th-99th percentile",
                                       "99th pct." = "99th percentile"
         )),
       aes(x=reorder(Noun,NumIsaRating),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2, size=0.5) + 
  stat_summary(fun=mean, geom="point", size=2, shape=15) +
  stat_summary(fun.data = mean_se, geom = "errorbar") +
  ggtitle(sprintf("Ratings for 'Is a %s N still an N?'", adj)) + 
  xlab("Noun") +
  ylab("Rating") + 
  guides(x = guide_axis(angle = 90))
  # theme(legend.position="bottom")
dev.copy(png, "isa_plot_fake_bigrams_with_frequency_abstract.png", width = 525, height = 250, res=96)
dev.off()

ggplot(isa_data_capped %>% filter(Adjective == adj) %>%
         mutate(Frequency = fct_recode(Frequency,
                                       "Below 10th pct." = "Below 10th percentile",
                                       "10th-25th pct." = "10th-25th percentile",
                                       "25th-50th pct." = "25th-50th percentile",
                                       "50th-75th pct." = "50th-75th percentile",
                                       "75th-90th pct." = "75th-90th percentile",
                                       "90th-95th pct." = "90th-95th percentile",
                                       "95th-99th pct." = "95th-99th percentile",
                                       "99th pct." = "99th percentile"
         )),
       aes(x=reorder(Noun,NumIsaRating),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2, size=1) +
  geom_boxplot(outlier.shape = NA) +
#  stat_summary(fun=mean, geom="point", size=3) +
#  stat_summary(fun.data = mean_se, geom = "errorbar") +
  ggtitle(sprintf("Ratings for 'Is a %s N still an N?'", adj)) + 
  xlab("Noun") +
  ylab("Rating") + 
  guides(x = guide_axis(angle = 90))
# theme(legend.position="bottom")



### Scatter plots with error bars ----

merge(isa_data_12_capped, 
      isa_variance_12_capped %>% select(Bigram, Mean, SD), 
      by = "Bigram") %>%
  group_by(Bigram) %>%
  mutate(SE = SD / sqrt(n())) %>% 
  ungroup() ->
  isa_data_12_capped

ggplot(isa_data_12_capped, 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating, color=Frequency)) + 
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



# isa_data_12_capped %>% distinct(Noun) %>% pull(Noun)
plot_single_adjective_poster("fake", "a", 
                             c("fact", "dollar", "concert", "reef", "fire", 
                               "gun", "door", "scarf", "crowd", "form",
                               "plan", "jacket", "report", "laugh", "image"))
ggsave("plots/isa_plot_fake_bigrams_poster.png", dpi=300, units="in",
       width=12, height=5, bg="transparent")

plot_single_adjective_poster("useful", "a")
ggsave("plots/isa_plot_useful_bigrams_with_frequency_poster.svg", dpi=300, units="in",
       width=18, height=9, bg="transparent")
ggsave("plots/isa_plot_useful_bigrams_with_frequency_poster.png", dpi=300, units="in",
       width=18, height=9, bg="transparent")

plot_single_adjective_poster("illegal", "an")
ggsave("plots/isa_plot_illegal_bigrams_with_frequency_poster.svg", dpi=300, units="in",
       width=18, height=9, bg="transparent")
ggsave("plots/isa_plot_illegal_bigrams_with_frequency_poster.png", dpi=300, units="in",
       width=18, height=9, bg="transparent")

### Box plots ----

ggplot(isa_data_12_capped, 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective, fun=median),
           y=NumIsaRating,color=AdjectiveClass)) + 
  geom_boxplot(outlier.size = 0.75) +
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle(sprintf("Ratings for 'Is an AN still an N?' (max. %s ratings/bigram)", max_participants_per_group)) + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "isa_plot_all_bigrams_by_adjective_boxplot.png", width = 2000, height = 900, res=180)
dev.off()

ggplot(isa_data_12_capped %>% filter(Adjective == "fake"), 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective, fun=median),
           y=NumIsaRating,color=AdjectiveClass)) + 
  geom_boxplot(outlier.size = 0.75) +
  ggtitle(sprintf("Ratings for 'Is a fake N still an N?' (max. %s ratings/bigram)", max_participants_per_group)) + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "isa_plot_fake_boxplot.png", width = 2000, height = 900, res=180)
dev.off()

### Violin plots over nouns ----

ggplot(isa_data_12_capped, aes(x=Adjective, y=NumIsaRating, color=AdjectiveClass)) +
  geom_violin() +
  facet_wrap(~ AdjectiveClass, scales = "free_x") +
  ggtitle("Ratings for 'Is an A N an N?'")

ggplot(isa_variance_12_capped, aes(x=reorder_within(x=Adjective,by=Mean,within=AdjectiveClass), 
             y=Mean, color=AdjectiveClass)) +
  geom_violin() +
  facet_wrap(~ AdjectiveClass, scales = "free_x") +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  xlab("Adjective") +
  ylab("Mean rating (one point per bigram)") +
  ggtitle("Mean bigram ratings for 'Is an A N an N?'") + 
  scale_color_manual(values=c('Subsective'='#F8766D', 'Privative'='#00BFC4')) +
  theme(legend.position='none') 

ggplot(isa_variance_12_capped %>%
         mutate(CoarseFrequency = fct_recode(CoarseFrequency,
                                       "1-25th pct." = "Below 25th percentile",
                                       "25th-50th pct." = "25th-50th percentile",
                                       "50th-75th pct." = "50th-75th percentile",
                                       "75th-90th pct." = "75th-90th percentile",
                                       "90th-99th pct." = "90th-99th percentile"
         )),
       aes(x=reorder_within(x=Adjective,by=Mean,within=AdjectiveClass), y=Mean)) +
  geom_violin(data=isa_variance_12_capped, scale="area", linewidth=0.75) +
  geom_jitter(height = 0, width = 0.075, size = 0.5, aes(col=CoarseFrequency)) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  xlab("Adjective") +
  ylab("Mean bigram rating") +
  labs(col='Bigram frequency') +
  ggtitle("Mean bigram ratings for 'Is an A N an N?'") 
ggsave(filename="plots/isa_plot_mean_ratings_violin.png", units="px", width = 1800, height = 400, dpi=180)

ggplot(isa_variance_12_capped,
       aes(x=reorder(Adjective, Mean), y=Mean)) +
  geom_violin(data=isa_variance_12_capped, scale="area", linewidth=0.5) +
  geom_jitter(height = 0, width = 0.05, size = 0.25, col="#00BFC4", alpha=0.4) +
  # geom_boxplot(aes(alpha=0.01), outlier.shape = NA) +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  xlab("Adjective") +
  ylab("Mean rating") +
  labs(col='Bigram frequency') +
  ggtitle("Mean bigram ratings for 'Is an A N an N?'") +
  theme(legend.position='none') 
ggsave(filename="plots/isa_plot_mean_ratings_violin_no_frequency.png", units="px", width = 1800, height = 400, dpi=200)

yarrr::pirateplot(formula = Mean ~ reorder(Adjective, Mean), 
                  data = isa_variance_12_capped,
                  main = "Mean bigram ratings for 'Is an A N an N?'",
                  xlab = "Adjective",
                  ylab = "Mean rating")


ggplot(extra_isa_data, 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=Adjective),
           y=NumIsaRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ Adjective, scales = "free_x") +
  ggtitle("Ratings for 'Is an AN an N?'") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
ggsave(filename="plots/isa_plot_extra_bigrams.png")


## Ratings vs. noun and adjective class ----

ggplot(isa_data_capped, 
       aes(x=reorder_within(x=Noun,by=NumIsaRating,within=AdjectiveClass),
           y=NumIsaRating, color=Noun)) +
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  stat_summary(fun.data = mean_se, geom = "errorbar") +
  ggtitle("Ratings for 'Is an AN an N?' (max 6. ratings/bigram)") + 
  xlab("Noun") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ AdjectiveClass, scales = "free_x")


## Variance vs. count/frequency ----

ggplot(isa_variance, aes(x=Count, y=Variance)) + 
  geom_point() + 
  ggtitle("Variance by corpus count")

ggplot(isa_variance_log, 
       aes(x=Count, y=Variance)) + 
  geom_point() + 
  ggtitle("Variance by corpus count") +
  scale_x_continuous(trans='log10')

ggplot(isa_variance, aes(x=Count, y=Variance)) + 
  geom_point() + 
  ggtitle("Variance by corpus count, by adjective type") +
  facet_wrap(~ AdjectiveClass) 

ggplot(isa_variance, aes(x=Count, y=Variance)) + 
  geom_point(aes(color=AdjectiveClass)) + 
  ggtitle("Variance by corpus count, by adjective") +
  facet_wrap(~ Adjective) 

ggplot(isa_variance, aes(x=Frequency, y=Variance)) +
  geom_point(aes(color=AdjectiveClass)) +
  ggtitle("Variance by corpus frequency") + 
  facet_wrap(~ AdjectiveClass)

ggplot(isa_variance, aes(x=Frequency, y=Variance)) + 
  geom_boxplot() + 
  ggtitle("Variance by corpus frequency")

ggplot(isa_variance, aes(x=Frequency, y=Variance)) + 
  geom_boxplot(outlier.shape = NA) + 
  geom_jitter(aes(color=Adjective),width=0.1, height=0.1) +
  ggtitle("Variance by corpus frequency, by adjective type") +
  facet_wrap(~ AdjectiveClass) 

ggplot(isa_variance, aes(x=Frequency, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) + 
  ggtitle("Variance by corpus frequency, by adjective") +
  facet_wrap(~ Adjective) 

### With capped variance ----

ggplot(isa_variance_capped_log, 
       aes(x=Count, y=Variance)) + 
  geom_point(aes(color=Frequency)) + 
  ggtitle(sprintf("Variance by corpus count (max. %s ratings/bigram)", max_participants_per_group)) +
  scale_x_continuous(trans='log10')

ggplot(isa_variance_capped_log, aes(x=Count, y=Variance)) + 
  geom_point(aes(color=Frequency)) + 
  ggtitle(sprintf("Variance by corpus count, by adjective type (max. %s ratings/bigram)", max_participants_per_group)) +
  scale_x_continuous(trans='log10') +
  xlab("Count (log scale)") +
  facet_wrap(~ AdjectiveClass) 
dev.copy(png, "isa_plot_count_variance_scatter_with_freqs.png", width = 600, height = 400)
dev.off()

ggplot(isa_variance_capped_log, aes(x=Count, y=Variance)) + 
  geom_point(aes(color=Frequency)) + 
  geom_smooth(method='lm', formula = y ~ x) +
  ggtitle(sprintf("Variance by corpus count, by adjective type (max. %s ratings/bigram)", max_participants_per_group)) +
  scale_x_continuous(trans='log10') +
  xlab("Count (log scale)") +
  facet_wrap(~ AdjectiveClass)
dev.copy(png, "isa_plot_count_variance_scatter_with_freqs_fitted.png", width = 1800, height = 1200, res=300)
dev.off()

r_squared_both_text <- data.frame(
  label = c(as.character(as.expression(sprintf('paste(italic(R)^2~"="~%.3f, ", adj."~italic(R)^2~"="~%.3f)', 
                                               int_variance_count_rsquared, int_variance_count_adj_rsquared))), 
            as.character(as.expression(sprintf('paste(italic(R)^2~"="~%.3f, ", adj."~italic(R)^2~"="~%.3f)', 
                                               priv_variance_count_rsquared, priv_variance_count_adj_rsquared)))
            ),
  AdjectiveClass   = c("Subsective", "Privative")
)

r_squared_text <- data.frame(
  label = c(as.character(as.expression(sprintf('italic(R)^2~"="~%.3f', 
                                               int_variance_count_rsquared))), 
            as.character(as.expression(sprintf('italic(R)^2~"="~%.3f', 
                                               priv_variance_count_rsquared)))
  ),
  AdjectiveClass   = c("Subsective", "Privative")
)

ggplot(isa_variance_capped_log, aes(x=Count, y=Variance)) + 
  geom_point(aes(color=AdjectiveClass)) + 
  geom_smooth(method='lm', formula = y ~ x) +
  ggtitle("Effect of frequency on variance") +
  scale_x_continuous(trans='log10') +
  xlab("Bigram Count (log scale)") +
  ylab("Rating Variance") +
  geom_text(data = r_squared_text,
            mapping = aes(x = 100, y = 3, label = label),
            color="blue", size=4,
            parse=TRUE) +
  facet_wrap(~ AdjectiveClass) +
  theme(legend.position='none') 
ggsave(filename="plots/isa_plot_count_variance_scatter_with_freqs_fitted_abstract.png", units="px", width = 450, height = 300, dpi=120)
# dev.copy(png, "isa_plot_count_variance_scatter_with_freqs_fitted_abstract.png", width = 450, height = 300, res=120)
# dev.off()

ggplot(isa_variance_capped_log, aes(x=Count, y=Variance)) + 
  geom_point(aes(color=AdjectiveClass), size=4, alpha=0.9) + 
  geom_smooth(method='lm', formula = y ~ x) +
  ggtitle("Effect of frequency on variance") +
  scale_x_continuous(trans='log10') +
  xlab("Bigram Count (log scale)") +
  ylab("Rating Variance") +
  geom_text(data = r_squared_text,
            mapping = aes(x = 100, y = 2.75, label = label),
            color="blue", size=12,
            parse=TRUE) +
  facet_wrap(~ AdjectiveClass) +
  theme_minimal() + 
  theme(text=element_text(size=36), 
        legend.position='none')
ggsave(filename="plots/isa_plot_count_variance_scatter_with_freqs_fitted_poster.png", units="in", width = 18, height = 15, dpi=300)


isa_variance_capped_log %>%
  filter(Bigram=="knockoff image") %>%
  select(Bigram, Mean, Variance) %>%
  as.data.frame

ggplot(isa_variance_capped_log, aes(x=Count, y=Variance)) + 
  geom_point(aes(color=Adjective)) + 
  ggtitle(sprintf("Variance by corpus count, by adjective type (max. %s ratings/bigram)", max_participants_per_group)) +
  scale_x_continuous(trans='log10') +
  xlab("Count (log scale)") +
  facet_wrap(~ AdjectiveClass) 
dev.copy(png, "isa_plot_count_variance_scatter_with_adjs.png", width = 600, height = 400)
dev.off()

ggplot(isa_variance_capped, aes(x=Frequency, y=Variance)) + 
  geom_boxplot() + 
  ggtitle("Variance by corpus frequency (max. 6 ratings/bigram)")

ggplot(isa_variance_capped, aes(x=Frequency, y=Variance)) + 
  geom_boxplot(outlier.shape = NA) + 
  geom_point(aes(color=Adjective)) +
  ggtitle(sprintf("Variance by corpus frequency, by adjective type (max. %s ratings/bigram)", max_participants_per_group)) +
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ AdjectiveClass) 
dev.copy(png, "isa_plot_frequency_variance_box.png", width = 600, height = 400)
dev.off()

ggplot(isa_variance_12_capped %>%
         mutate(Frequency = fct_recode(Frequency,
           "Below 10th" = "Below 10th percentile",
           "10th-25th" = "10th-25th percentile",
           "25th-50th" = "25th-50th percentile",
           "50th-75th" = "50th-75th percentile",
           "75th-90th" = "75th-90th percentile",
           "90th-95th" = "90th-95th percentile",
           "95th-99th" = "95th-99th percentile",
           "99th" = "99th percentile"
         )), 
       aes(x=Frequency, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) + 
  ggtitle(sprintf("Variance by corpus frequency, by adjective (max. %s ratings/bigram)", max_participants_per_group)) +
  guides(x = guide_axis(angle = 90)) +
  facet_wrap(~ Adjective)
dev.copy(png, "isa_plot_frequency_variance_box_with_class.png", width = 600, height = 400)
dev.off()

## Variance by adjective class ----

ggplot(isa_variance, aes(x=Adjective, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) +
  ggtitle("Variance by adjective")

ggplot(isa_variance, aes(x=Adjective, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) +
  ggtitle("Variance by adjective") +
  facet_wrap(~ Frequency)

ggplot(isa_variance, aes(x=AdjectiveClass, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) + 
  ggtitle("Variance by adjective class")


### With capped variance ----

ggplot(isa_variance_12_capped, aes(x=Adjective, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) +
  ggtitle(sprintf("Variance by adjective (max. %s ratings/bigram)", max_participants_per_group))

ggplot(isa_variance_12_capped, aes(x=Adjective, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) +
  guides(x = guide_axis(angle = 90)) +
  ggtitle(sprintf("Variance by adjective (max. %s ratings/bigram)", max_participants_per_group)) +
  facet_wrap(~ Frequency)

ggplot(isa_variance_12_capped, aes(x=AdjectiveClass, y=Variance)) + 
  geom_boxplot(aes(color=AdjectiveClass)) + 
  ggtitle(sprintf("Variance by adjective class (max. %s ratings/bigram)", max_participants_per_group))
dev.copy(png, "isa_plot_var_adjclass_box.png", width = 1400, height = 900, res=300)
dev.off()

# Models ----

## Variance vs. count/frequency ----

### Log count ----

variance_count_log_lm <- lm(Variance ~ log(Count), data = isa_variance_capped_log)

summary(variance_count_log_lm)

int_variance_count_log_lm <- lm(Variance ~ log(Count), data = isa_variance_capped_log %>% filter(AdjectiveClass == "Intersective"))

summary(int_variance_count_log_lm)

int_variance_count_log_rsquared <- summary(int_variance_count_log_lm)$adj.r.squared

plot(allEffects(int_variance_count_log_lm))

priv_variance_count_log_lm <- lm(Variance ~ log(Count), data = isa_variance_capped_log %>% filter(AdjectiveClass == "Privative"))

summary(priv_variance_count_log_lm)

priv_variance_count_log_rsquared <- summary(priv_variance_count_log_lm)$adj.r.squared

### Log count with adjective class

variance_count_log_lm2 <- lm(Variance ~ log(Count) * AdjectiveClass, data = isa_variance_capped_log)

summary(variance_count_log_lm2)

### With adjective as random effect

variance_count_log_lmm <- lmer(Variance ~ log(Count) + (1 | Adjective), data = isa_variance_capped_log)

summary(variance_count_log_lmm)

### Raw count (worse) ----

variance_count_lm <- lm(Variance ~ Count + AdjectiveClass, data = isa_variance_capped)
summary(variance_count_lm)

int_variance_count_lm <- lm(Variance ~ Count, data = isa_variance_capped %>% filter(AdjectiveClass == "Subsective"))

summary(int_variance_count_lm)

int_variance_count_rsquared <- summary(int_variance_count_lm)$r.squared
int_variance_count_adj_rsquared <- summary(int_variance_count_lm)$adj.r.squared

priv_variance_count_lm <- lm(Variance ~ Count, data = isa_variance_capped %>% filter(AdjectiveClass == "Privative"))

summary(priv_variance_count_lm)

priv_variance_count_rsquared <- summary(priv_variance_count_lm)$r.squared
priv_variance_count_adj_rsquared <- summary(priv_variance_count_lm)$adj.r.squared

### Frequency category ----

int_variance_freq_lm <- lm(Variance ~ Frequency, data = isa_variance_capped %>% filter(AdjectiveClass == "Intersective"))

summary(int_variance_freq_lm)

plot(allEffects(int_variance_freq_lm))

## Variance vs. adjective & count ----

variance_adjc_count_lm <- lmer(Variance ~ log(Count) * AdjectiveClass + (1 | Noun), data = isa_variance_capped_log)

summary(variance_adjc_count_lm)

# -> Count is not really significant

variance_adj_count_lm <- lmer(Variance ~ log(Count) * Adjective + (1 | Noun), data = isa_variance_capped_log)

summary(variance_adj_count_lm)

# -> Count is not significant

# Frequency

variance_adj_freq_lm <- lmer(Variance ~ Frequency * Adjective + (1 | Noun), data = isa_variance_capped)

summary(variance_adj_freq_lm)

variance_adjc_freq_lm <- lmer(Variance ~ Frequency * AdjectiveClass + (1 | Noun), data = isa_variance_capped)

summary(variance_adjc_freq_lm)

int_variance_adjc_freq_lm <- lmer(Variance ~ Frequency + (1 | Adjective) + (1 | Noun), data = isa_variance_capped %>% filter(AdjectiveClass == "Intersective"))

summary(int_variance_adjc_freq_lm)

# Nothing is significant for intersective adjectives, not even 99th percentile
