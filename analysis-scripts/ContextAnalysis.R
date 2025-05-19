# Packages ----

library(ordinal)
library(MASS)
library(effects)
library(tidyverse)
library(tidytext)
library(lmerTest)

# Import data ----

data_wide <- read.csv("results/human/exp3_context/context_responses_raw.csv",
                           fileEncoding = "UTF-8-BOM")

# Remove the first two rows, which contain the header again and the full question text
data_wide <- data_wide[-c(1,2),]
rownames(data_wide) <- NULL

colnames(data_wide)

# Fix column names
data_wide <- data_wide %>%
  rename("cf.dollar.i" = "cf.dollar.p") %>%
  rename("cf.dollar.p" = "cf.dollar.p.1")

# Select only question columns; drop all user information & comments except demographic questionnaire
data_wide <- subset(data_wide, 
                         select = grep("^([a-z]+.[a-z]+.[tip]|Q2|Q3|Q4|Q5)", 
                                       names(data_wide)))

# Add fake UserID (don't want to use actual Prolific ID)
data_wide$UserId <- 1:nrow(data_wide)

# Rename demographic questions
data_wide <- data_wide %>%
  rename(EnglishBefore5 = Q2, Dialect = Q3, OtherEnglish = Q3_2_TEXT, OtherLanguages = Q4, Comments = Q5)

# Exclusion criteria ----

dem_excluded_ids <- data_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & OtherEnglish != "American English")) %>% 
  pull(UserId)

dem_excluded_ids

data_wide <- data_wide %>%
  mutate(AttnFailed = ifelse(or.mouse.i %in% c("Definitely not", "Probably not"), 1, 0) +
           ifelse(w.pear.p %in% c("Definitely yes", "Probably yes"), 1, 0)
  )

attn_excluded_ids <- data_wide %>%
  filter(AttnFailed >= 1) %>%
  pull(UserId)

attn_excluded_ids

data_wide_excl <- data_wide %>%
  filter(!UserId %in% dem_excluded_ids & !UserId %in% attn_excluded_ids)

nrow(data_wide_excl)

# Select target questions and pivot ----

names(data_wide_excl)

an_context <- data_wide_excl %>%
  dplyr::select(!names(data_wide_excl)[1:3]) %>%  # Exclude training
  dplyr::select(!c("AttnFailed")) %>%
  dplyr::select(!EnglishBefore5:Comments) %>% # Exclude demographics
  pivot_longer(
    cols = cf.diamond.p:fk.reef.p,
    names_to = c("Adjective","Noun","ContextBias"),
    names_pattern = "([a-z]+).([a-z]+).([tip])",
    values_to = "Rating"
  ) %>%
  mutate(Adjective = fct_recode(Adjective, 
                                counterfeit = "cf",
                                fake = "fk",
                                false = "fls",
                                former = "fm",
                                knitted = "kn",
                                broken = "br",
                                miniature = "mn",
                                temporary = "tmp",
                                wooden = "w",
                                orange = "or"
                                )) %>%
  mutate(ContextBias = fct_recode(ContextBias,
                                  intersective = "i",
                                  privative = "p"
                                  )) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate_at(c("UserId", "Adjective", "Noun", "Bigram", "ContextBias"), factor) %>%
  mutate(Bigram = fct_relevel(Bigram, "counterfeit diamond", "counterfeit dollar", 
                              "fake reef", "fake fire", 
                              "fake scarf", "fake drug",
                              "fake glance", "fake plan",
                              "false concert", "false war",
                              "former accusation", "former house"
                              ))

an_context <- an_context %>%
  filter(Rating!="") %>%
  mutate(Rating = factor(Rating, levels = c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"))) %>%
  mutate(NumRating = as.integer(Rating))


str(an_context)

write.csv(an_context, file = "an_context_exp1_long.csv", row.names=FALSE)

# Add no-context ratings ----

isa_data_12_capped_copy <- as_tibble(read.csv("isa_data_12_capped.csv")) %>%
  mutate_at(c("Adjective", "Noun", "Bigram", "AdjectiveClass", "CoarseFrequency", "IsaRating", "ParticipantId"), factor)

str(isa_data_12_capped_copy)

an_context_plus <- an_context %>%
  mutate(UserId = paste0('CTXT', UserId)) %>%
  mutate(UserId = factor(UserId)) %>%
  merge(isa_data_12_capped_copy %>%
          mutate(ParticipantId = paste0('ISA', ParticipantId)) %>%
          mutate(UserId = factor(ParticipantId)) %>%
          mutate(ContextBias = "no context") %>%
          mutate(ContextBias = factor(ContextBias)) %>%
          rename(Rating = IsaRating, NumRating = NumIsaRating) %>%
          select(c('Bigram', 'Rating', 'NumRating', 'ContextBias', 'UserId', 'Adjective', 'Noun')) %>%
          filter(Bigram %in% an_context$Bigram),
        by=c("Bigram", "UserId", "ContextBias", "Rating", "NumRating", "Adjective", "Noun"),
        all=TRUE
  ) %>%
  mutate(Bigram = droplevels(factor(Bigram))) %>%
  mutate(Noun = droplevels(factor(Noun))) %>%
  mutate(ContextBias = fct_relevel(ContextBias, "intersective", "no context", "privative"
  )) %>%
  as_tibble() 


str(an_context_plus)

# Add frequency ----

bigrams_freqs <- read.csv("../Adjectives-PythonCode/output/filtering_data/all_c4_3912_bigrams_with_frequencies.csv", header = TRUE)

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
  )))

an_context_plus <- merge(an_context_plus,
                      unique_bigrams_freqs %>% dplyr::select("Bigram", "Count", "Frequency"),
                      by=c("Bigram"),
                      all.x=TRUE) %>%
  mutate(HighLowFrequency = factor(ifelse(as.integer(Frequency) <= 4, "Low/Zero", "High"))) %>%
  mutate(Frequency = droplevels(Frequency))
  

str(an_context_plus)

# Split out subsets ----

an_context_all <- an_context_plus %>%
  filter(ContextBias %in% c('intersective', 'privative')) %>%
  mutate(ContextBias = droplevels(ContextBias))

str(an_context_all)

an_context_target <- an_context_all %>%
  filter(Adjective %in% c("counterfeit", "fake", "false", "former"))

an_context_plus_target <- an_context_plus %>%
  filter(Adjective %in% c("counterfeit", "fake", "false", "former"))

an_context_filler <- an_context_all %>%
  filter(Adjective %in% c("knitted", "broken", "temporary", "miniature"))

str(an_context_target)

# Variance ----

an_context_variance <- an_context_target %>%
  group_by(Bigram,ContextBias) %>%
  summarise(Variance = var(NumRating), Mean = mean(NumRating), SD = sd(NumRating),
            Adjective = unique(Adjective), Noun = unique(Noun), Bigram = unique(Bigram), 
            HighLowFrequency = unique(HighLowFrequency), Frequency = unique(Frequency), Count = unique(Count)) %>%
  ungroup() %>%
  as_tibble()

str(an_context_variance)

isa_variance_context <- as_tibble(read.csv("isa_variance_context_bigrams.csv")) 

an_context_variance <- an_context_variance %>%
  merge(isa_variance_context %>%
          rename(IsaMean = Mean, IsaVariance = Variance),
        by=c("Bigram"),
        all.x=TRUE) %>%
  as_tibble() 

str(an_context_variance)

an_context_variance %>%
  dplyr::select(Bigram, ContextBias, Mean, IsaMean, Variance, IsaVariance) %>%
  print(n=Inf)

# Plot ----

## Histograms ----

ggplot(an_context_all, aes(x=Rating)) +
  geom_bar() +
  ggtitle("Ratings")


ggplot(an_context_all, aes(x=Rating)) +
  geom_bar() +
  ggtitle("Ratings") +
  facet_wrap(~ Adjective)

## Assorted box/jitter plots (target bigrams) ----

ggplot(an_context_target, aes(x=Bigram, y=NumRating, group=ContextBias, col=ContextBias)) +
  geom_jitter(width=0.25, height=0.1) + 
  # stat_summary(fun.data = mean_se, geom = "errorbar", alpha=0.5) +
  # stat_summary(fun=mean, geom="point", size=4) +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")

ggplot(an_context_target, aes(x=Bigram, y=NumRating, col=ContextBias)) +
  geom_jitter(width=0.25, height=0.1) + 
  stat_summary(fun=mean, geom="point", size=4, color='darkgrey') +
  stat_summary(fun.data = mean_se, geom = "errorbar", color='darkgrey') +  
  facet_wrap(~ContextBias) +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")

ggplot(an_context_target, aes(x=ContextBias, y=NumRating, col=HighLowFrequency)) +
  geom_boxplot(outlier.shape=NA) +
  geom_point(position=position_jitterdodge()) +
  facet_wrap(~Adjective) +
  xlab('Frequency') +
  ylab('Rating') +
  labs(col='Context bias') +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")
dev.copy(png, "context_boxplot_rating_by_frequency_by_adjective.png", width = 1600, height = 1000, res=180)
dev.off()

ggplot(an_context_target, aes(x=Bigram, y=NumRating, col=HighLowFrequency)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) + 
  facet_wrap(~ContextBias) +
  xlab('Frequency') +
  ylab('Rating') +
  labs(col='Context bias') +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")

ggplot(an_context_target, aes(x=Bigram, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_point(position=position_jitterdodge()) +
  facet_wrap(~HighLowFrequency, scales = "free_x") +
  xlab('Frequency') +
  ylab('Rating') +
  labs(col='Context bias') +
  guides(x = guide_axis(angle = 90)) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")
dev.copy(png, "context_boxplot_rating_by_frequency_paired.png", width = 2000, height = 1000, res=180)
dev.off()

ggplot(an_context_target, aes(x=ContextBias, y=NumRating, col=HighLowFrequency)) +
  geom_boxplot(outlier.shape=NA) +
  facet_wrap(~Adjective) +
  xlab('Frequency') +
  ylab('Rating') +
  labs(col='Context bias') +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")
dev.copy(png, "context_boxplot_rating_by_frequency_by_adjective_nodots.png", width = 1600, height = 1000, res=180)
dev.off()

ggplot(an_context_target, aes(x=ContextBias, y=NumRating, col=HighLowFrequency)) +
  geom_boxplot(outlier.shape=NA) +
  geom_point(position=position_jitterdodge()) +
  facet_wrap(~Bigram) +
  xlab('Frequency') +
  ylab('Rating') +
  labs(col='Context bias') +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")

ggplot(an_context_target, aes(x=ContextBias, y=NumRating, col=HighLowFrequency)) +
  geom_boxplot(outlier.shape=NA) +
  geom_point(position=position_jitterdodge()) +
  xlab('Frequency') +
  ylab('Rating') +
  labs(col='Context bias') +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")

ggplot(an_context_target, aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias')
dev.copy(png, "context_plot_by_bigram_by_bias.png", width = 1600, height = 1000, res=180)
dev.off()

ggplot(an_context_plus_target %>%
         mutate(ContextBias = fct_recode(ContextBias, 
                                         "none" = "no context",
                                         "subsective" = "intersective")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram, ncol=6) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias') + 
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#F8766D', 'privative'='#00BFC4')) +
  theme(legend.position='none')
dev.copy(png, "context_plot_by_bigram_by_bias_with_isa.png", width = 2000, height = 600, res=180)
dev.off()

ggplot(an_context_plus_target %>%
         mutate(ContextBias = fct_recode(ContextBias, 
                                         "subsective" = "intersective")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram, ncol=6) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  # guides(x = "none") +
  scale_x_discrete(name='Context bias', 
                   labels=c('subs.', 'none', 'priv.')) +
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#F8766D', 'privative'='#00BFC4', 'no context'='#7F7F7F'))
ggsave(filename="plots/context_plot_by_bigram_by_bias_with_isa_legend.png", units="px", width = 2000, height = 600, dpi=200)

ggplot(an_context_plus_target %>%
         mutate(ContextBias = fct_recode(ContextBias, 
                                         "subsective" = "intersective")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram, ncol=4) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  # guides(x = "none") +
  scale_x_discrete(name='Context bias') +
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#F8766D', 'privative'='#00BFC4', 'no context'='#7F7F7F'))
ggsave(filename="plots/context_plot_by_bigram_by_bias_with_isa_legend_4x4.png", units="px", width = 2800, height = 1700, dpi=300)


abstract_bigrams <- c('fake reef', 'fake fire', 'false concert', 'false war', 'counterfeit diamond', 'counterfeit dollar')
ggplot(an_context_plus_target %>% filter(Bigram %in% abstract_bigrams) %>%
         mutate(ContextBias = fct_recode(ContextBias, 
                                         "none" = "no context",
                                         "subsective" = "intersective")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram, nrow=1) +
  ggtitle("Ratings for 'In this setting, is an A N still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias') + 
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#F8766D', 'privative'='#00BFC4')) +
  theme(legend.position='none')
dev.copy(png, "context_plot_by_bigram_by_bias_abstract.png", width = 2000, height = 400, res=200)
dev.off()


ggplot(an_context_target, 
       aes(x=reorder_within(x=Bigram,by=NumRating,within=ContextBias),
           y=NumRating,color=Frequency)) + 
  geom_jitter(width=0.2, height=0.2) + 
  stat_summary(fun=mean, geom="point", size=4) +
  # Doesn't seem to play well with reorder_within + not every plot having every noun
  # stat_summary(fun.data = mean_se, geom = "errorbar") +  
  facet_wrap(~ ContextBias, scales = "free_x") +
  ggtitle("Ratings for 'In this setting, Is an AN still an N?'") + 
  xlab("Adjective-noun pair") +
  ylab("Rating") + 
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) + 
  guides(x = guide_axis(angle = 90))
dev.copy(png, "context_plot_by_bigram_matching_isa.png", width = 2000, height = 1000, res=180)
dev.off()

## Target variance ----

isa_variance_context

ggplot(an_context_variance, aes(x=HighLowFrequency, y=Variance)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.1, height=0.1) +
  xlab("Frequency")
dev.copy(png, "context_plot_variance_by_frequency.png", width = 1000, height = 1000, res=180)
dev.off()

ggplot(an_context_variance, aes(x=HighLowFrequency, y=Variance, col=ContextBias)) +
  geom_jitter(width=0.1, height=0.1) +
  facet_wrap(~Adjective) +
  xlab("Frequency")

ggplot(an_context_variance, aes(x=Bigram, y=Variance, col=ContextBias)) +
  geom_jitter(width=0.1, height=0.1) +
  facet_wrap(~HighLowFrequency) +
  xlab("Frequency") + 
  guides(x = guide_axis(angle = 90))

ggplot(an_context_variance, aes(x=Bigram)) +
  geom_point(aes(y=Mean, col=ContextBias)) +
  geom_point(aes(y=IsaMean, color="no context")) + 
  guides(x = guide_axis(angle = 90)) +
  ylab("Mean Rating") +
  xlab("Adjective-noun pair") +
  scale_color_manual(name="Context Bias",
                     breaks=c('no context', 'intersective', 'privative'),
                     values=c('no context'='black', 'intersective'='#F8766D', 'privative'='#00BFC4'))
dev.copy(png, "context_plot_mean_move_comparison.png", width = 800, height = 600, res=180)
dev.off()

ggplot(an_context_variance, aes(x=Bigram)) +
  geom_point(aes(y=Variance, col=ContextBias)) +
  geom_point(aes(y=IsaVariance, color="no context")) + 
  guides(x = guide_axis(angle = 90)) +
  ylab("Rating Variance") +
  xlab("Adjective-noun pair") +
  scale_color_manual(name="Context Bias",
                   breaks=c('no context', 'intersective', 'privative'),
                   values=c('no context'='black', 'intersective'='#F8766D', 'privative'='#00BFC4'))
dev.copy(png, "context_plot_variance_reduction_comparison.png", width = 800, height = 600, res=180)
dev.off()

# hue_pal()(2)
## Fillers ----

ggplot(an_context_filler, aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias')

# Fit models ----

## Variance ----

# Hypothesis: each high frequency bigram should have a context that "fits" and a 
# context that doesn't. So it should have
# (a) one low variance and one high variance context
# or
# (b) two low variance contexts with the same rating
# Meanwhile for low frequency bigrams, we predict that it should be easy to 
# manipulate their context, so they should have low variance and bimodal ratings.
# Any effects of them being hard to understand, because infrequent, should be
# offset by the explicit description of what they're supposed to mean.

var_freq_lm <- lm(Variance ~ HighLowFrequency, data=an_context_variance)

summary(var_freq_lm)

var_freq_ct_lm <- lm(Variance ~ HighLowFrequency + ContextBias, data=an_context_variance)

summary(var_freq_ct_lm)

## Contexts actually work ----

an_context_biased <- an_context_target %>%
  filter(ContextBias %in% c('intersective', 'privative'))

bias_works_lm <- clm(Rating ~ ContextBias, 
                     data = an_context_plus_target %>% 
                       mutate(ContextBias = fct_relevel(ContextBias, 'no context')) %>%
                       mutate(ContextBias = fct_recode(ContextBias, 'subsective' = 'intersective')) %>%
                       mutate(Rating = factor(Rating, 
                                              levels=c("Definitely not", "Probably not", 
                                                       "Unsure",
                                                       "Probably yes", "Definitely yes"))),
                     link = "logit")

summary(bias_works_lm)

bias_works_lmm <- clmm(Rating ~ ContextBias + (1 | UserId), 
                       data = an_context_plus_target %>% 
                         mutate(ContextBias = fct_relevel(ContextBias, 'no context')) %>%
                         mutate(ContextBias = fct_recode(ContextBias, 'subsective' = 'intersective')) %>%
                         mutate(Rating = factor(Rating, 
                                                levels=c("Definitely not", "Probably not", 
                                                         "Unsure",
                                                         "Probably yes", "Definitely yes"))),
                       link = "logit")

summary(bias_works_lmm)

exp(coef(bias_works_lmm)[5:6])

AIC(bias_works_lm)
AIC(bias_works_lmm)

plot(allEffects(bias_works_lmm), 
     style="stacked",
     colors=rev(hcl.colors(5, palette="TealRose")),
     main="Effect of context on rating",
     xlab="Context bias"
     )
dev.copy(png, "context_bias_effects_plot.png", width = 1200, height = 900, res=180)
dev.off()

bias_works_ip_lmm <- clmm(Rating ~ ContextBias + (1 | UserId), 
     data = an_context_biased, link = "logit")

summary(bias_works_lmm)
