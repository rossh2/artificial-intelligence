# Packages ----

library(tidyverse)
library(tidytext)
library(ordinal)

# Prepare data ----

## Import data ----

context2_wide <- read.csv("results/human/exp3_context/context2_responses_raw.csv",
                      fileEncoding = "UTF-8-BOM")

# Remove the first two rows, which contain the header again and the full question text
context2_wide <- context2_wide[-c(1,2),]
rownames(context2_wide) <- NULL

colnames(context2_wide)

# Select only question columns; drop all user information & comments except demographic questionnaire
context2_wide <- subset(context2_wide, 
                    select = grep("^([a-z]+.[a-z]+.[tsp]|Q2|Q3|Q4|Q5)", 
                                  names(context2_wide)))

# Add fake UserID (don't want to use actual Prolific ID)
context2_wide$UserId <- 1:nrow(context2_wide)

# Rename demographic questions
context2_wide <- context2_wide %>%
  rename(EnglishBefore5 = Q2, Dialect = Q3, OtherEnglish = Q3_2_TEXT, OtherLanguages = Q4, Comments = Q5)

## Exclusion criteria ----

dem_excluded_ids <- context2_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & OtherEnglish != "American English")) %>% 
  pull(UserId)

dem_excluded_ids

context2_wide <- context2_wide %>%
  mutate(AttnFailed = ifelse(or.mouse.s %in% c("Definitely not", "Probably not"), 1, 0) +
           ifelse(w.pear.p %in% c("Definitely yes", "Probably yes"), 1, 0)
  )

attn_excluded_ids <- context2_wide %>%
  filter(AttnFailed >= 1) %>%
  pull(UserId)

attn_excluded_ids

excluded_ids <- context2_wide %>%
  filter(UserId %in% dem_excluded_ids | UserId %in% attn_excluded_ids) %>%
  select(EnglishBefore5, Dialect, OtherEnglish, OtherLanguages, or.mouse.s, w.pear.p)

context2_wide_excl <- context2_wide %>%
  filter(!UserId %in% dem_excluded_ids & !UserId %in% attn_excluded_ids)

nrow(context2_wide_excl)

## Select target questions and pivot ----

names(context2_wide_excl)

an_context2 <- context2_wide_excl %>%
  dplyr::select(!names(context2_wide_excl)[1:3]) %>%  # Exclude training
  dplyr::select(!c("AttnFailed")) %>%
  dplyr::select(!EnglishBefore5:Comments) %>% # Exclude demographics
  rename("kt.pizza.s" = "kn.pizza.s", "kt.pizza.p" = "kn.pizza.p") %>%
  pivot_longer(
    cols = a.couple.p:fm.image.s,
    names_to = c("Adjective","Noun","ContextBias"),
    names_pattern = "([a-z]+).([a-z]+).([tsp])",
    values_to = "Rating"
  ) %>%
  mutate(Adjective = fct_recode(Adjective, 
                                artificial = "a",
                                counterfeit = "c",
                                fake = "fk",
                                false = "fl",
                                former = "fm",
                                knockoff = "kn",
                                knitted = "kt",
                                broken = "br",
                                miniature = "mn",
                                temporary = "tmp",
                                wooden = "w",
                                orange = "or"
  )) %>%
  mutate(ContextBias = fct_recode(ContextBias,
                                  subsective = "s",
                                  privative = "p"
  )) %>%
  unite(Bigram, c(Adjective, Noun), sep = " ", remove = FALSE) %>%
  mutate_at(c("UserId", "Adjective", "Noun", "Bigram", "ContextBias"), factor) 

an_context2 <- an_context2 %>%
  filter(Rating!="") %>%
  mutate(Rating = factor(Rating, levels = c("Definitely not", "Probably not", "Unsure", "Probably yes", "Definitely yes"))) %>%
  mutate(NumRating = as.integer(Rating))


str(an_context2)

## Add no-context ratings ----

# Adjust water / spring water

an_context2 <- an_context2 %>%
  mutate(Noun = fct_recode(Noun, "spring water" = "water"), 
         Bigram = fct_recode(Bigram, 
                             "counterfeit spring water" = "counterfeit water",
                             "knockoff spring water" = "knockoff water"))

# Combine

isa_data_combined_copy <- as_tibble(read.csv("isa_data_combined.csv")) %>%
  mutate_at(c("Adjective", "Noun", "Bigram", "AdjectiveClass", "Rating", "UserId"), factor)

isa_data_combined_copy %>% filter(grepl("water", Noun))

an_context2_plus <- an_context2 %>%
  mutate(UserId = paste0('CTXT', UserId)) %>%
  mutate(UserId = factor(UserId)) %>%
  merge(isa_data_combined_copy %>%
          mutate(UserId = paste0('ISA', UserId)) %>%
          mutate(ContextBias = "no context") %>%
          mutate(ContextBias = factor(ContextBias)) %>%
          select(c('Bigram', 'Rating', 'NumRating', 'ContextBias', 'UserId', 'Adjective', 'Noun')) %>%
          filter(Bigram %in% an_context2$Bigram),
        by=c("Bigram", "UserId", "ContextBias", "Rating", "NumRating", "Adjective", "Noun"),
        all=TRUE
  ) %>%
  mutate(Bigram = droplevels(Bigram)) %>%
  mutate(Noun = droplevels(Noun)) %>%
  mutate(ContextBias = fct_relevel(ContextBias, "subsective", "no context", "privative"
  )) %>%
  as_tibble() 



## Split out target items ----

an_context2_plus_target <- an_context2_plus %>%
  filter(Adjective %in% c("artificial", "counterfeit", "fake", "false", "former", "knockoff"))

# Plots ----

ggplot(an_context2_plus_target %>%
         mutate(ContextBias = fct_recode(ContextBias, 
                                         "none" = "no context")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram, ncol=5) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias') + 
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#F8766D', 'privative'='#00BFC4')) +
  theme(legend.position='none')
ggsave("plots/context2_plot_by_bigram_by_bias_with_isa.png", width=1500, height=1000,
       units="px", dpi=180)

ggplot(an_context2_plus_target %>%
         mutate(ContextBias = fct_recode(ContextBias, 
                                         "none" = "no context")), 
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
ggsave("plots/context2_plot_by_bigram_by_bias_with_isa_wide.png", width=1800, height=800,
       units="px", dpi=180)

# Fit models ----

bias_works2_lmm <- clmm(Rating ~ ContextBias + (1 | UserId), 
                       data = an_context2_plus_target %>% 
                         mutate(ContextBias = fct_relevel(ContextBias, 'no context')),
                       link = "logit")

summary(bias_works2_lmm)

bias_by_adjective_lmm <- clmm(Rating ~ ContextBias * Adjective + (1 | UserId), 
                        data = an_context2_plus_target %>% 
                          mutate(ContextBias = fct_relevel(ContextBias, 'no context')),
                        link = "logit")

summary(bias_by_adjective_lmm)

bias_by_adjective_2way_lmm <- clmm(Rating ~ ContextBias * Adjective + (1 | UserId), 
                              data = an_context2_plus_target %>% 
                                filter(ContextBias != 'no context') %>%
                                mutate(ContextBias = fct_drop(ContextBias)),
                              link = "logit")

summary(bias_by_adjective_2way_lmm)

# Combine with first context experiment ----

## Build dataframe ----

an_context = read.csv("an_context_exp1_long.csv")

an_context_combined = bind_rows(an_context %>%
                                  mutate(UserId = paste0('CTXT1-', UserId)) %>%
                                  # Exclude these since we re-ran them
                                  filter(!(Bigram %in% c("fake reef", "former house"))) %>%
                                  mutate(ContextBias = fct_recode(ContextBias, 
                                                                  "subsective" = "intersective")), 
                                an_context2 %>% 
                                  mutate(UserId = paste0('CTXT2-', UserId)))

an_context_combined_plus <- an_context_combined %>%
  mutate(UserId = factor(UserId)) %>%
  merge(isa_data_combined_copy %>%
          mutate(UserId = paste0('ISA', UserId)) %>%
          mutate(ContextBias = "no context") %>%
          mutate(ContextBias = factor(ContextBias)) %>%
          select(c('Bigram', 'Rating', 'NumRating', 'ContextBias', 'UserId', 'Adjective', 'Noun')) %>%
          filter(Bigram %in% an_context_combined$Bigram),
        by=c("Bigram", "UserId", "ContextBias", "Rating", "NumRating", "Adjective", "Noun"),
        all=TRUE
  ) %>%
  mutate(Bigram = droplevels(factor(Bigram))) %>%
  mutate(Noun = droplevels(factor(Noun))) %>%
  mutate(Rating = factor(Rating)) %>%
  mutate(ContextBias = fct_relevel(ContextBias, "subsective", "no context", "privative"
  )) %>%
  as_tibble() 

# Re-sort bigrams
bigram_levels <- levels(an_context_combined_plus$Bigram)
an_context_combined_plus <- an_context_combined_plus %>%
  mutate(Bigram = factor(Bigram, c(sort(bigram_levels))))

str(an_context_combined_plus)

an_context_combined_plus_target <- an_context_combined_plus %>%
  filter(Adjective %in% c("artificial", "counterfeit", "fake", "false", "former", "knockoff"))

write.csv(an_context_combined_plus_target, 'isa_context_combined_target.csv', row.names=FALSE)

## Plots ----

ggplot(an_context_combined_plus_target %>%
         mutate(ContextBias = fct_recode(ContextBias, 
                                         "none" = "no context")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram) +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias') + 
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#F8766D', 'privative'='#00BFC4')) +
  theme(legend.position='none')
ggsave("plots/context2_combined_plot_by_bigram_by_bias_with_isa.png", dpi=180)

ggplot(an_context_combined_plus_target %>%
         filter(Bigram %in% c("counterfeit diamond", "counterfeit dollar", 
                              "counterfeit spring water", "counterfeit drug",  
                              "fake concert", "fake laugh",
                              "fake reef", "fake fire", 
                              "false concert", "false war",
                              "former image", "former house")) %>%
         mutate(Bigram = fct_relevel(Bigram, 
                                     "counterfeit diamond", "counterfeit dollar", 
                                     "counterfeit spring water", "counterfeit drug",  
                                     "fake concert", "fake laugh",
                                     "fake reef", "fake fire", 
                                     "false concert", "false war",
                                     "former image", "former house"
                                     )) %>%
         mutate(ContextBias = fct_relevel(fct_recode(ContextBias, 
                                         "none" = "no context"),
                                         "subsective", "none", "privative")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram) +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias') + 
  theme_minimal() + 
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#00BFC4', 'privative'='#F8766D')) +
  theme(legend.position='none') -> context2plot_selected
context2plot_selected +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'")
ggsave("plots/context2_selected_plot_by_bigram_by_bias_with_isa.png", dpi=180,
       width = 7, height = 4, units="in")
context2plot_selected +
  theme(text=element_text(size=14),
        panel.grid.minor.y = element_blank())
ggsave("plots/context2_selected_plot_by_bigram_by_bias_with_isa_paper.png", dpi=300,
       width = 9.5, height = 3.5, units="in")

context2plot_selected +
  theme(text = element_text(size=12, family="Palatino Linotype"),
        strip.text.x = element_text(size=9.5),
        axis.text = element_text(size=9),
        panel.grid.minor.y = element_blank()) +
  scale_color_manual(name="Context bias",
                     values=c('subsective'=light_blue_color, 'privative'=magenta_color))
ggsave("plots/context2_selected_plot_by_bigram_by_bias_with_isa_diss.png", dpi=300,
       width = 6.5, height = 4, units="in")


ggplot(an_context_combined_plus_target %>%
         filter(Bigram %in% c("fake fire", "fake reef", "counterfeit painting")) %>%
         mutate(Bigram = fct_relevel(Bigram, 
                                     "fake fire", "fake reef", "counterfeit painting"
         )) %>%
         mutate(ContextBias = fct_relevel(fct_recode(ContextBias, 
                                                     "none" = "no context"),
                                          "subsective", "none", "privative")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram, nrow = 1) +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias') + 
  theme_minimal() + 
  scale_color_manual(name="Context bias",
                     values=c('subsective'=light_blue_color, 'privative'=magenta_color)) +
  theme(legend.position='none') +
  ggtitle("Ratings for 'In this setting, is an AN still an N?'") +
  theme(text=element_text(size=14))
ggsave("plots/context2selected2_plot_by_bigram_by_bias_with_isa_slides.png",
       width = 6.5, height = 2.5, units="in")

ggplot(an_context_combined_plus_target %>%
         filter(Bigram %in% c("fake fire")) %>%
         mutate(Bigram = fct_relevel(Bigram, 
                                     "fake fire"
         )) %>%
         mutate(ContextBias = fct_relevel(fct_recode(ContextBias, 
                                                     "none" = "no context"),
                                          "subsective", "none", "privative")), 
       aes(x=ContextBias, y=NumRating, col=ContextBias)) +
  geom_boxplot(outlier.shape=NA) +
  geom_jitter(width=0.25, height=0.1) +
  facet_wrap(~Bigram, nrow = 1) +
  xlab('Context bias') +
  ylab('Rating') +
  labs(col='Context bias') + 
  theme_minimal() + 
  scale_color_manual(name="Context bias",
                     values=c('subsective'='#00BFC4', 'privative'='#F8766D')) +
  theme(legend.position='none') +
  theme(text=element_text(size=14))
ggsave("plots/context2_fakefire_plot_by_bias_with_isa_slides.png",
       width = 2.75, height = 1.75, units="in")

# Models ----

## Contexts actually work ----

bias2_works_lmm <- clmm(Rating ~ ContextBias + (1 | UserId), 
                       data = an_context_combined_plus_target %>% 
                         mutate(ContextBias = fct_relevel(ContextBias, 'no context')) %>%
                         mutate(Rating = factor(Rating, 
                                                levels=c("Definitely not", "Probably not", 
                                                         "Unsure",
                                                         "Probably yes", "Definitely yes"))),
                       link = "logit")

summary(bias2_works_lmm)

exp(coef(bias2_works_lmm)[5:6])
