library(tidyverse)
library(ggeffects)
library(tidytext)
library(lmerTest)
library(MuMIn)

# Prepare data ----

## Import data ----

fake_wide <- read.csv("results/human/exp4_fake/fake_responses.raw.csv",
                          fileEncoding = "UTF-8-BOM")
real_wide <- read.csv("results/human/exp4_fake/real_responses.raw.csv")

# Remove the first two rows, which contain the header again and the full question text
fake_wide <- fake_wide[-c(1,2),]
rownames(fake_wide) <- NULL

real_wide <- real_wide[-c(1,2),]
rownames(reak_wide) <- NULL

colnames(fake_wide)

# Select only question columns; drop all user information & comments except demographic questionnaire
fake_wide <- subset(fake_wide, 
                        select = grep("^(X[0-9].[a-z]+.[a-z]+|EnglishBefore5|Dialect|OtherLanguages|BlockOrder)", 
                                      names(fake_wide)))
real_wide <- subset(real_wide, 
                    select = grep("^(X[0-9].[a-z]+.[a-z]+|EnglishBefore5|Dialect|OtherLanguages|BlockOrder)", 
                                  names(real_wide)))

# Add fake UserID (don't want to use actual Prolific ID)
fake_wide$UserId <- 1:nrow(fake_wide)
real_wide$UserId <- nrow(fake_wide)+(1:nrow(real_wide))

## Demographic exclusion criteria ----

fake_dem_excluded_ids <- fake_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & Dialect_2_TEXT != "American English")) %>% 
  pull(UserId)

fake_dem_excluded_ids

real_dem_excluded_ids <- real_wide %>%
  filter(EnglishBefore5!="Yes" | (Dialect!="Yes" & Dialect_2_TEXT != "American English")) %>% 
  pull(UserId)

real_dem_excluded_ids

## Select target questions and pivot ----

names(fake_wide)

fake_wide %>%
  rename("X3.gun.gunmetal_1" = "X3.gun.metal_1", "X3.needle.needlemetal_1" = "X3.needle.metal_1",
         "X2.bird.birdfly_1" = "X2.bird.fly_1", "X1.airplane.airplanefly_1" = "X1.airplane.fly_1") %>%
  dplyr::select(!EnglishBefore5:OtherLanguages) %>% # Exclude demographics
  pivot_longer(
    cols = X1.table.support_1:X3.painting.copy_1,
    names_to = c("QuestionBlockId","Noun","Property"),
    names_pattern = "X([0-9]).([a-z]+).([a-z]+)_1",
    values_to = "FakeRating"
  ) %>%
  mutate(across(c("UserId", "BlockOrder", "QuestionBlockId", "Noun", "Property"), factor)) %>%
  mutate(FakeRating = as.numeric(FakeRating)) ->
  an_fake_all

real_wide %>%
  rename("X3.gun.gunmetal_1" = "X3.gun.metal_1", "X3.needle.needlemetal_1" = "X3.needle.metal_1",
         "X2.bird.birdfly_1" = "X2.bird.fly_1", "X1.airplane.airplanefly_1" = "X1.airplane.fly_1") %>%
  dplyr::select(!EnglishBefore5:OtherLanguages) %>% # Exclude demographics
  pivot_longer(
    cols = X1.table.support_1:X3.painting.copy_1,
    names_to = c("QuestionBlockId","Noun","Property"),
    names_pattern = "X([0-9]).([a-z]+).([a-z]+)_1",
    values_to = "RealRating"
  ) %>%
  mutate(across(c("UserId", "BlockOrder", "QuestionBlockId", "Noun", "Property"), factor)) %>%
  mutate(RealRating = as.numeric(RealRating)) ->
  an_real_all


str(an_fake_all)
str(an_real_all)


## Exclusion criteria ----

an_fake_all %>%
  # mutate(Outlier = (UserId %in% c(4, 9, 18))) %>%
  ggplot(aes(x=FakeRating,color=UserId)) + # , linewidth=Outlier
  geom_density()

# Distribution of User 4 looks nothing like the others, 
# manual inspection also suggests it does not look like they were doing the task
# and they didn't follow the training
fake_attn_excluded_ids <- c(4)

an_real_all %>%
#  mutate(Outlier = (UserId %in% c(26, 38))) %>%
  ggplot(aes(x=RealRating,color=UserId)) + # , linewidth=Outlier
  geom_density()

real_attn_excluded_ids <- c()

an_fake_only <- an_fake_all %>%
  filter(!UserId %in% fake_dem_excluded_ids & !UserId %in% fake_attn_excluded_ids)

an_real_only <- an_real_all %>%
  filter(!UserId %in% real_dem_excluded_ids & !UserId %in% real_attn_excluded_ids)

an_fake_only %>% distinct(UserId) %>% nrow()
an_real_only %>% distinct(UserId) %>% nrow()

# Add additional data ----

## Write out properties ----


expand_properties <- function(an_fr_only) {
  an_fr_only %>%
    mutate(Property = fct_recode(Property, 
                                 "Tables support objects" = "support",
                                 "Guns are not toy guns" = "toy",
                                 "Airplanes can fly" = "airplanefly",
                                 "Eggshells are laid by animals" = "laid",
                                 "Cars have four wheels" = "wheels",
                                 "Fire trucks are red" = "red",
                                 "Lemons are sour" = "sour",
                                 "Sandpaper is brown" = "brown",
                                 "Ambulances have sirens" = "siren",
                                 "Police officers eat donuts" = "donut",
                                 "Plans contain information" = "information",
                                 "Sofas can be sat on" = "seat",
                                 "Tightrope walkers have good balance" = "balance",
                                 "Raincoats are waterproof" = "waterproo",
                                 "Tables are flat" = "flat",
                                 "Guns can shoot" = "shoot",
                                 "Airplanes have wings" = "wings",
                                 "Trampolines are bouncy" = "bouncy",
                                 "Cars have radios" = "radio",
                                 "Needles are sharp" = "sharp",
                                 "Lemons are yellow" = "yellow",
                                 "Rocks are hard" = "hard",
                                 "Ambulances have life-saving equipment" = "equipmen",
                                 "Snow is white" = "white",
                                 "Plans are carried out" = "carried",
                                 "Submarines are airtight" = "watertig",
                                 "Birds can fly" = "birdfly",
                                 "Concerts are only concerts" = "only",
                                 "Tables are made of wood" = "wooden",
                                 "Guns are made of metal" = "gunmetal",
                                 "Eggshells are fragile" = "fragile",
                                 "Trampolines are black" = "black",
                                 "Fire trucks have hoses" = "hose",
                                 "Needles are made of metal" = "needlemetal",
                                 "Sandpaper is rough" = "rough",
                                 "Rocks are made of minerals" = "mineral",
                                 "Police officers wear badges" = "badge",
                                 "Snow forms in clouds" = "clouds",
                                 "Sofas are big enough for humans to sit on" = "size",
                                 "Lifeguards can swim" = "swim",
                                 "Scissors cut" = "cut",
                                 "Paintings are painted by the expected artist" = "copy"
    )) %>%
    mutate(Noun = fct_recode(Noun,
                             "fire truck" = "firetruck",
                             "tightrope walker" = "tightrope",
                             "police officer" = "police"
    )) ->
    an_fr_only
  
  return(an_fr_only)
}

an_fake_only %>% 
  expand_properties() ->
  an_fake_only

an_real_only %>% 
  expand_properties() ->
  an_real_only

## From P&D ----

pd_ratings <- read.csv("../../Readings/Prasada_Dillingham_Appendix_Data.csv") 

pd_ratings %>%
  rename(Property = Phrase) %>%
  mutate(Property = as.factor(Property)) ->
  pd_ratings

pd_ratings$RowId <- 1:nrow(pd_ratings)

pd_ratings %>%
  mutate(PropertyType = "", PropertySource = "P&D") %>%
  rows_update(tibble(RowId = 1:45, PropertyType = "k-property"), by="RowId") %>%
  rows_update(tibble(RowId = 46:90, PropertyType = "t-property"), by="RowId") %>%
  select(!RowId) ->
    pd_ratings


str(pd_ratings)

an_fake_only %>%
  merge(pd_ratings, by = "Property", all.x = TRUE) ->
  an_fake

an_real_only %>%
  merge(pd_ratings, by = "Property", all.x = TRUE) ->
  an_real

## Add estimated property types ----

add_property_types <- function(an_fr) {
  an_fr %>%
    mutate(PropertySource = case_when(!is.na(PropertySource) ~ PropertySource,
                                      Property %in% c("Lemons are yellow", 
                                                      "Sandpaper is brown", 
                                                      "Needles are made of metal",
                                                      "Guns are made of metal"
                                                      ) ~ "Analogy to P&D",
                                      .default = "Estimate"),
           PropertyType = case_when(!is.na(PropertyType) ~ PropertyType,
                                    as.character(Property) %in% c("Airplanes can fly",
                                                    "Eggshells are laid by animals",
                                                    "Tables support objects",
                                                    "Rocks are made of minerals",
                                                    "Ambulances have life-saving equipment",
                                                    "Snow forms in clouds",
                                                    "Guns can shoot",
                                                    "Plans contain information",
                                                    "Plans are carried out",
                                                    "Concerts are only concerts",
                                                    "Sofas can be sat on"
                                                    ) ~ "k-property",
                                    as.character(Property) %in% c("Needles are made of metal",
                                                    "Lemons are yellow",
                                                    "Sandpaper is brown",
                                                    "Guns are made of metal",
                                                    "Sofas are big enough for humans to sit on"
                                                    ) ~ "t-property"
                                     ),
           FinePropertyType = case_when(as.character(Property) %in% c("Airplanes can fly",
                                                        "Eggshells are laid by animals",
                                                        "Tables support objects",
                                                        "Trampolines are bouncy",
                                                        "Submarines are airtight",
                                                        "Sandpaper is rough",
                                                        "Rocks are made of minerals",
                                                        "Ambulances have life-saving equipment",
                                                        "Guns can shoot",
                                                        "Scissors cut",
                                                        "Plans contain information",
                                                        "Sofas can be sat on"
                                                        ) ~ "essential (privative) k-property",
                                        as.character(Property) %in% c("Cars have four wheels",
                                                        "Lifeguards can swim",
                                                        "Needles are sharp",
                                                        "Lemons are sour",
                                                        "Rocks are hard",
                                                        "Birds can fly",
                                                        "Snow forms in clouds",
                                                        "Raincoats are waterproof",
                                                        "Plans are carried out",
                                                        "Concerts are only concerts"
                                                        ) ~ "important (subsective) k-property",
                                        as.character(Property) %in% c("Airplanes have wings",
                                                        "Eggshells are fragile",
                                                        "Tables are flat",
                                                        "Fire trucks have hoses",
                                                        "Tightrope walkers have good balance",
                                                        "Ambulances have sirens",
                                                        "Police officers wear badges",
                                                        "Snow is white",
                                                        "Sofas are big enough for humans to sit on"
                                                        ) ~ "borderline/statistical 'k-property'",
                                        PropertyType == "t-property" ~ "t-property"
                                        )
           ) %>%
    mutate(across(c("PropertyType", "PropertySource", "FinePropertyType"), as.factor),
           PropertyType = fct_relevel(PropertyType, "t-property"),
           FinePropertyType = fct_relevel(FinePropertyType, 
                                          "t-property",
                                          "borderline/statistical 'k-property'",
                                          "important (subsective) k-property",
                                          "essential (privative) k-property"
                                          ),
           Property = fct_relevel(Property, sort),
           Noun = fct_relevel(Noun, sort)) ->
    an_fr
  contrasts(an_fr$PropertyType) <- contr.helmert(2)
  contrasts(an_fr$FinePropertyType) <- contr.helmert(4)
  
  return(an_fr)
}

an_fake %>%
  add_property_types() ->
  an_fake

an_real %>%
  add_property_types() ->
  an_real

str(an_fake)
str(an_real)

## Block info & z-scores ----

add_block <- function(an_fr) {
  an_fr %>%
    mutate(SeenInBlock = case_when(
      substr(BlockOrder, start=1, stop=1) == QuestionBlockId ~ "1st",
      substr(BlockOrder, start=2, stop=2) == QuestionBlockId ~ "2nd",
      substr(BlockOrder, start=3, stop=3) == QuestionBlockId ~ "3rd",
    )) ->
    an_fr
  return(an_fr)
}

an_fake %>%
  add_block() %>%
  group_by(UserId) %>%
  mutate(FakeZRating = scale(FakeRating)) %>%
  ungroup() ->
  an_fake

an_real %>%
  add_block() %>%
  group_by(UserId) %>%
  mutate(RealZRating = scale(RealRating)) %>%
  ungroup() ->
  an_real

## Exclude bad contexts ----

# Exclude scissors can cut since fake can't be used felicitously for broken things

an_fake %>%
  rows_update(tibble(Property = "Scissors cut", PropertyType = NA, FinePropertyType = NA), by="Property") ->
  an_fake

an_real %>%
  rows_update(tibble(Property = "Scissors cut", PropertyType = NA, FinePropertyType = NA), by="Property") ->
  an_real

## Combine fake and real ----

merge(
  an_fake %>%
    group_by(Property) %>%
    summarize(MeanFakeRating = mean(FakeRating), MeanFakeZRating = mean(FakeZRating),
              Noun = first(Noun), 
              PropertyType = first(PropertyType), FinePropertyType = first(FinePropertyType),
              PropertySource = first(PropertySource),
              In.General = first(In.General), By.Virtue.Of = first(By.Virtue.Of), 
              Prevalence.Estimates = first(Prevalence.Estimates)),
  an_real %>%
    group_by(Property) %>%
    summarize(MeanRealRating = mean(RealRating), MeanRealZRating = mean(RealZRating)),
  by="Property") -> an_fake_real 

str(an_fake_real)

# Plots ----

## Fake ----

an_fake %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=FakeZRating, color=PropertyType)) +
  geom_density() +
  theme_minimal()

an_fake %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=PropertyType, y=FakeZRating)) +
  geom_boxplot() +
  theme_minimal() +
  labs(x="Property category according to experimenter judgments", y="Rating for 'fake' negating property (z-score)")

an_fake %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=FinePropertyType, y=FakeZRating)) +
  geom_boxplot() +
  theme_minimal() +
  labs(x="Property category according to my judgments", y="Rating for 'fake' negating property (z-score)")

an_fake %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=FinePropertyType, y=FakeZRating, color=Noun)) +
  geom_boxplot() +
  theme_minimal() +
  labs(x="Property category according to my judgments", y="Rating for 'fake' negating property (z-score)")

an_fake %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=Property, y=FakeZRating, color=Noun)) +
  geom_boxplot() +
  theme_minimal() +
  guides(x = guide_axis(angle = 90)) +
  facet_grid(~ FinePropertyType, scales = "free_x") +
  labs(x="Property category according to my judgments", y="Rating for 'fake' negating property (z-score)")

an_fake %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=reorder(Property, FakeZRating, FUN=mean), y=FakeZRating, color=PropertyType)) +
  geom_boxplot() +
  theme_minimal() +
  guides(x = guide_axis(angle = 90)) +
  labs(x="Property", y="Rating for 'fake' negating property (z-score)", color="Property type\naccording to P&D\n(extended by analogy)")


an_fake %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=reorder(Property, FakeZRating, FUN=mean), y=FakeZRating, color=FinePropertyType)) +
  geom_boxplot() +
  theme_minimal() +
  guides(x = guide_axis(angle = 45)) +
  labs(x="Property", y="Rating for 'fake' negating property (z-score)", color="Fine-grained property type\n(by author introspection)") +
  theme(plot.margin=unit(c(0.5, 0.5, 0.5, 1.5), 'cm'))
ggsave('plots/kproperties_fake_barplot_property_by_finegrained.png', width=10, height=5.5, units="in")

# Some noise from context design -> "scissors cut" should be a privative k-property really just like "guns can shoot"
# but apparently we make an exception for / fake cannot target broken things (guns or scissors) 
# and my context used old=broken scissors instead of scissors that were designed not to be able to cut, 
# which means fake can't be used to describe these non-cutting scissors

# Also it looks like people are somewhat willing to extend "fake [wooden] table" and "fake [metal] gun" with contextual support

an_fake %>%
  filter(!is.na(By.Virtue.Of) & !is.na(PropertyType)) %>%
  group_by(Property) %>%
  summarize(By.Virtue.Of = first(By.Virtue.Of), FakeMeanZRating = mean(FakeZRating)) %>%
  ggplot(aes(x=By.Virtue.Of, y=FakeMeanZRating)) +
  geom_point() +
  geom_smooth(method="lm") +
  labs(x="Bare plural + 'by virtue of' rating in P&D", y="Mean rating (z-scored) for 'fake' negating property") + 
  theme_minimal()

## Real ----

an_real %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=PropertyType, y=RealZRating)) +
  geom_boxplot() +
  theme_minimal() +
  labs(x="Property category according to experimenter judgments", y="Rating for 'not real' negating property (z-score)")

an_real %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=FinePropertyType, y=RealZRating)) +
  geom_boxplot() +
  theme_minimal() +
  labs(x="Property category according to my judgments", y="Rating for 'not real' negating property (z-score)")

an_real %>%
  filter(!is.na(PropertyType)) %>%
  ggplot(aes(x=reorder(Property, RealZRating, FUN=mean), y=RealZRating, color=FinePropertyType)) +
  geom_boxplot() +
  theme_minimal() +
  guides(x = guide_axis(angle = 45)) +
  labs(x="Property", y="Rating for 'not real' negating property (z-score)", color="Fine-grained property type\n(by author introspection)") +
  theme(plot.margin=unit(c(0.5, 0.5, 0.5, 1.5), 'cm'))
ggsave('plots/kproperties_real_barplot_property_by_finegrained.png', width=10, height=5.5, units="in")

 
# Regressions ----

## Fake ----

### Block order ----

# Block seen in is not significant
rating_by_blockorder_lm <- lm(FakeZRating ~ SeenInBlock, data = an_fake)
summary(rating_by_blockorder_lm)

### By property type ----

#### Rating by type ----

# Helmert coding used throughout (see Sonderegger p.202) (unless using an ordered factor)
# so the significance is whether the level is significantly different from the combination of previous levels
# 1. Is borderline k different from t?
# 2. Is important k different from t + borderline k?
# 3. Is essential k different from t + borderline k + important k?
# Since our data isn't balanced, a weighted Helmert coding would be better but we'd have to implement that by hand (see Sonderegger)

# Including a slope by UserId causes a singular fit with or without z-scores... why?
rating_by_type_lm <- lmer(FakeZRating ~ PropertyType + (0 + PropertyType | UserId) + (1 | SeenInBlock), data = an_fake)
summary(rating_by_type_lm)

rating_by_type_lm2 <- lmer(FakeRating ~ PropertyType + (1 + PropertyType | UserId) + (1 | SeenInBlock), data = an_fake)
summary(rating_by_type_lm2)

# This model uses z-scores and no random effect of participant; it's partially addressed by the z-score
rating_by_type_lm3 <- lmer(FakeZRating ~ PropertyType + (1 | SeenInBlock), data = an_fake)
summary(rating_by_type_lm3)
r.squaredGLMM(rating_by_type_lm3)

#plot(predict_response(rating_by_type_lm3, terms=c("PropertyType")), show_data = TRUE, jitter = 0.2, dot_alpha = 0.1) +
rating_by_type_lm3_preds = predict_response(rating_by_type_lm3, terms=c("PropertyType"))
ggplot(rating_by_type_lm3_preds, aes(x=x, y=predicted, color=x)) +
  geom_jitter(data = model.frame(rating_by_type_lm3),
              aes(x=PropertyType, y= FakeZRating, color=PropertyType),
              width=0.2, alpha = 0.1, inherit.aes = FALSE) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0, linewidth=0.8) +
  labs(
    # x = "Property category (according to P&D's / author's introspection)",
    # y = "Acceptability of fake when property negated (z-scored)",
    # title = "Effects plot for acceptability of fake with P&D property categories"
    title="",
    x="Property type",
    y="Acceptability of 'fake' (z-scored)"
) +
  theme_minimal() +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10),
        legend.position = "none"
  ) +
  scale_color_manual(values=c('t-property'=light_blue_color, 'k-property'=magenta_color))
ggsave('plots/kproperties_fake_rating_by_pdcategory_colored.png', height=3.25, width=3.25)

# We can also use an ordered factor for proper comparison to the fine-grained properties below, 
# see below for notes / interpretation
rating_by_type_lm4 <- lmer(FakeZRating ~ OrdPropertyType + (1 | SeenInBlock), 
                           data = an_fake %>%
                             mutate(OrdPropertyType = as.ordered(PropertyType)))
summary(rating_by_type_lm4)

plot(predict_response(rating_by_type_lm4, terms=c("OrdPropertyType"))) +
  labs(x = "Property category (according to P&D's / author's introspection)",
       y = "Acceptability of fake when property negated (z-scored)",
       title = "Effects plot for acceptability of fake with P&D property categories")

rating_by_finetype_lm <- lmer(FakeZRating ~ FinePropertyType + (1 | SeenInBlock), 
                              data = an_fake %>%
                                mutate(FinePropertyType = fct_recode(FinePropertyType,
                                                                     "non-essential\nk-property" = "important (subsective) k-property",
                                                                     "essential\nk-property" = "essential (privative) k-property",
                                                                     "borderline\n'k-property'" = "borderline/statistical 'k-property'")))
summary(rating_by_finetype_lm)
r.squaredGLMM(rating_by_finetype_lm)

# Showing data makes it hard to see how different the categories are (none of the error bars overlap)
plot(predict_response(rating_by_finetype_lm, terms=c("FinePropertyType"))) +
  labs(x = "Fine-grained property category (according to author's introspection)",
       y = "Acceptability of fake when property negated (z-scored)",
       title = "Effects plot for acceptability of fake with fine-grained property categories")
plot(predict_response(rating_by_finetype_lm, terms=c("FinePropertyType")), show_data = TRUE, jitter = 0.2, dot_alpha = 0.1) +
  labs(
    # x = "Fine-grained property category (according to author's introspection)",
    # y = "Acceptability of fake when property negated (z-scored)",
    # title = "Effects plot for acceptability of fake with fine-grained property categories")
    x = "Fine-grained property type",
    y = "Acceptability of 'fake' (z-scored)",
    title = ""
    ) +
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) 
ggsave('plots/kproperties_fake_rating_by_finecategory.png', height=3.25, width=4.8)

# Ordered factor - see Sonderegger p. 208 (contrast coding scheme: orthogonal polynomial contrasts)
# Shows that there is a significant linear relationship (not a quadratic one, also kind of a cubic one) 
# between the levels of fine property type and the rating
rating_by_finetype_lm2 <- lmer(FakeZRating ~ OrdFinePropertyType + (1 | SeenInBlock), 
                               data = an_fake %>%
                                 mutate(OrdFinePropertyType = as.ordered(FinePropertyType)))
summary(rating_by_finetype_lm2)

plot(predict_response(rating_by_finetype_lm2, terms=c("OrdFinePropertyType")), show_data = TRUE, jitter = 0.2, dot_alpha = 0.1) +
  labs(x = "Fine-grained property category (according to author's introspection)",
       y = "Acceptability of fake when property negated (z-scored)",
       title = "Effects plot for acceptability of fake with fine-grained property categories")

#### Variance by type ----

an_fake %>%
  group_by(Property) %>% 
  summarize(FakeZRatingSD = sd(FakeZRating), FinePropertyType = first(FinePropertyType),
            PropertyType = first(PropertyType)) ->
  an_fake_variance

an_fake_variance %>%
  group_by(PropertyType) %>%
  summarize(mean(FakeZRatingSD))

varrating_by_finetype_lm <- lm(FakeZRatingSD ~ FinePropertyType, 
                               data = an_fake_variance)
summary(varrating_by_finetype_lm)

# subsective k-properties have significantly higher variance than t-properties
# but borderline k-properties and privative k-properties don't

plot(predict_response(varrating_by_finetype_lm, terms = c("FinePropertyType")), show_data = TRUE, dot_alpha = 0.2, jitter = 0.1)


### By P&D rating ----

# Significant
rating_by_pdrating_lm <- lmer(FakeZRating ~ By.Virtue.Of + (1 | SeenInBlock), 
                              data = an_fake %>%
                                filter(!is.na(PropertyType)))
summary(rating_by_pdrating_lm)

plot(predict_response(rating_by_pdrating_lm, terms=c("By.Virtue.Of")), show_data = TRUE, colors="us", jitter = 0.02) +
  labs(x = "P&D mean rating for property in bare plural generic with 'by virtue of'",
       y = "Acceptability of fake when property negated (z-scored)",
       title = "Effects plot for acceptability of fake vs. P&D ratings")
     
rating_by_pdratingcat_lm <- lmer(FakeZRating ~ PropertyType * By.Virtue.Of + (1 | SeenInBlock), 
                              data = an_fake %>%
                                filter(!is.na(PropertyType)))
summary(rating_by_pdratingcat_lm)
# Both factors and the interaction are significant! (but only once we remove "scissors cut" which was a poorly designed context)
# Effects plot is cool
r.squaredGLMM(rating_by_pdratingcat_lm)

plot(predict_response(rating_by_pdratingcat_lm, terms=c("By.Virtue.Of", "PropertyType"))) +
  labs(x = "P&D mean rating for property in bare plural generic with 'by virtue of'",
       y = "Acceptability of fake when property negated (z-scored)",
       title = "Effects plot for acceptability of fake vs. P&D ratings",
       color = "Property type\naccording to P&D\n(extended by analogy)")

plot(predict_response(rating_by_pdratingcat_lm, terms=c("By.Virtue.Of", "PropertyType")), 
     show_data = TRUE, jitter = 0.02, dot_alpha = 0.2) +
  labs(
#    x = "P&D mean rating for property in bare plural generic with 'by virtue of'",
#       y = "Acceptability of fake when property negated (z-scored)",
#       title = "Effects plot for acceptability of fake vs. P&D ratings",
#       color = "Property type\naccording to P&D\n(extended by analogy)") 
    title="",
    x="Mean `by virtue of' rating",
    y="Acceptability of 'fake' (z-scored)",
    color = "Property type"
) + 
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) +
  theme(legend.position = "top", legend.key.size = unit(0.5, "lines")) +
  guides(color = guide_legend(title.position = "top", title.hjust = 0.5)) +
  scale_color_manual(values=c('t-property'=light_blue_color, 'k-property'=magenta_color))
ggsave('plots/kproperties_fake_rating_by_pdrating.png', height=3.25, width=3.25)

rating_by_pdrating_konly_lm <- lmer(FakeZRating ~ By.Virtue.Of + (1 | SeenInBlock), 
                                    data = an_fake %>%
                                      filter(PropertyType == "k-property"))
summary(rating_by_pdrating_konly_lm)

plot(predict_response(rating_by_pdrating_konly_lm, terms=c("By.Virtue.Of")), show_data = TRUE, colors="us", jitter = 0.02) +
  labs(x = "P&D mean rating for property in bare plural generic with 'by virtue of'",
       y = "Acceptability of fake when property negated (z-scored)",
       title = "Effects plot for acceptability of fake vs. P&D ratings for properties labelled as 'k-properties'")

## Real ----

# Block seen in is not significant
r_rating_by_blockorder_lm <- lm(RealZRating ~ SeenInBlock, data = an_real)
summary(r_rating_by_blockorder_lm)

### By property type ----

#### Rating by type ----

# This model uses z-scores and no random effect of participant; it's partially addressed by the z-score
r_rating_by_type_lm <- lmer(RealZRating ~ PropertyType + (1 | SeenInBlock), data = an_real)
summary(r_rating_by_type_lm)

# Significant effect for real as well as fake
plot(predict_response(r_rating_by_type_lm, terms=c("PropertyType")), show_data = TRUE, jitter = 0.2, dot_alpha = 0.1) +
  labs(x = "Property category (according to P&D's / author's introspection)",
       y = "Acceptability of 'not real' when property negated (z-scored)",
       title = "Effects plot for acceptability of 'not real' with P&D property categories")
ggsave('plots/kproperties_real_rating_by_pdcategory.png')

r_rating_by_finetype_lm <- lmer(RealZRating ~ FinePropertyType + (1 | SeenInBlock), data = an_real)
summary(r_rating_by_finetype_lm)

# Showing data makes it hard to see how different the categories are (none of the error bars overlap)
plot(predict_response(r_rating_by_finetype_lm, terms=c("FinePropertyType"))) +
  labs(x = "Fine-grained property category (according to author's introspection)",
       y = "Acceptability of 'not real' when property negated (z-scored)",
       title = "Effects plot for acceptability of 'not real' with fine-grained property categories")
plot(predict_response(r_rating_by_finetype_lm, terms=c("FinePropertyType")), show_data = TRUE, jitter = 0.2, dot_alpha = 0.1) +
  labs(x = "Fine-grained property category (according to author's introspection)",
       y = "Acceptability of 'not real' when property negated (z-scored)",
       title = "Effects plot for acceptability of 'not real' with fine-grained property categories")
ggsave('plots/kproperties_real_rating_by_finecategory.png')

### By P&D rating ----

r_rating_by_pdratingcat_lm <- lmer(RealZRating ~ PropertyType * By.Virtue.Of + (1 | SeenInBlock), 
                                 data = an_real %>%
                                   filter(!is.na(PropertyType)))
summary(r_rating_by_pdratingcat_lm)
# Both factors and the interaction are significant! (but only once we remove "scissors cut" which was a poorly designed context)
# Effects plot is cool

plot(predict_response(r_rating_by_pdratingcat_lm, terms=c("By.Virtue.Of", "PropertyType"))) +
  labs(x = "P&D mean rating for property in bare plural generic with 'by virtue of'",
       y = "Acceptability of 'not real' when property negated (z-scored)",
       title = "Effects plot for acceptability of 'not real' vs. P&D ratings",
       color = "Property type\naccording to P&D\n(extended by analogy)")

plot(predict_response(r_rating_by_pdratingcat_lm, terms=c("By.Virtue.Of", "PropertyType")), 
     show_data = TRUE, jitter = 0.02, dot_alpha = 0.2) +
  labs(x = "P&D mean rating for property in bare plural generic with 'by virtue of'",
       y = "Acceptability of 'not real' when property negated (z-scored)",
       title = "Effects plot for acceptability of 'not real' vs. P&D ratings",
       color = "Property type\naccording to P&D\n(extended by analogy)")
ggsave('plots/kproperties_real_rating_by_pdrating.png')

## Fake vs. real ----

fake_real_raw_lm1 <- lm(MeanRealRating ~ MeanFakeRating, data = an_fake_real)
summary(fake_real_raw_lm1)
# Adjusted R-squared: 0.8363

plot(predict_response(fake_real_raw_lm1, terms=c("MeanFakeRating")), show_data = TRUE) +
  labs(x = "Mean raw acceptability of 'fake' when property negated",
       y = "Mean raw acceptability of 'not real' when property negated",
       title = "Correlation of mean acceptability (per property) of 'not real' and 'not fake'")

fake_real_z_lm1 <- lm(MeanRealZRating ~ MeanFakeZRating, data = an_fake_real)
summary(fake_real_z_lm1)
# Adjusted R-squared: 0.8213

plot(predict_response(fake_real_z_lm1, terms=c("MeanFakeZRating")), show_data = TRUE) +
  labs(x = "Mean acceptability of 'fake' (z-scored)",
       y = "Mean acceptability of 'not real' (z-scored)",
       title = "") + # Correlation of mean acceptability (per property) of 'not real' and 'not fake'
  theme(text = element_text(size=10, family="Palatino Linotype"),
        strip.text.x = element_text(size=10),
        axis.text = element_text(size=10)
  ) 
ggsave('plots/kproperties_real_zrating_vs_fake.png', height=3, width=4)


fake_real_z_lm2 <- lm(MeanRealZRating ~ MeanFakeZRating + PropertyType, data = an_fake_real)
summary(fake_real_z_lm2)
# Adjusted R-squared: 0.8227, no significant effect of property type 

# Use property type to facet even though not significant
plot(predict_response(fake_real_z_lm2, terms=c("MeanFakeZRating", "PropertyType")), show_data = TRUE, facets = TRUE) +
  labs(x = "Mean z-scored acceptability of 'fake' when property negated",
       y = "Mean z-scored acceptability of 'not real' when property negated",
       title = "Correlation of mean acceptability (per property) of 'not real' and 'not fake'")


fake_real_z_lm3 <- lm(MeanRealZRating ~ MeanFakeZRating + FinePropertyType, data = an_fake_real)
summary(fake_real_z_lm3)
# Adjusted R-squared: 0.8227, no significant effect of fine property type 

# Use property type to facet even though not significant
plot(predict_response(fake_real_z_lm3, terms=c("MeanFakeZRating", "FinePropertyType")), show_data = TRUE, facets = TRUE) +
  labs(x = "Mean z-scored acceptability of 'fake' when property negated",
       y = "Mean z-scored acceptability of 'not real' when property negated",
       title = "Correlation of mean acceptability (per property) of 'not real' and 'not fake'")


# Artist and toy gun ----

painting_property = "Paintings are painted by the expected artist"
toygun_property = "Guns are not toy guns"
metalgun_property = "Guns are made of metal"

an_fake %>% 
  filter(Property == painting_property) %>% 
  ggplot(aes(x=FakeZRating)) +
  geom_density()

an_fake %>% 
  filter(Property == painting_property) %>% 
  group_by(BinaryRating = if_else(FakeRating >= 50, "Acceptable", "Unacceptable")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == painting_property) %>% 
  group_by(BinaryRating = if_else(FakeZRating >= 0, "AcceptableZ", "UnacceptableZ")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == toygun_property) %>% 
  ggplot(aes(x=FakeZRating)) +
  geom_density()

an_fake %>% 
  filter(Property == toygun_property) %>% 
  group_by(BinaryRating = if_else(FakeRating >= 50, "Acceptable", "Unacceptable")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == toygun_property) %>% 
  group_by(BinaryRating = if_else(FakeZRating >= 0, "AcceptableZ", "UnacceptableZ")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == metalgun_property) %>% 
  ggplot(aes(x=FakeZRating)) +
  geom_density()

an_fake %>% 
  filter(Property == metalgun_property) %>% 
  group_by(BinaryRating = if_else(FakeRating >= 50, "Acceptable", "Unacceptable")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == metalgun_property) %>% 
  group_by(BinaryRating = if_else(FakeZRating >= 0, "AcceptableZ", "UnacceptableZ")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == "Lifeguards can swim") %>% 
  group_by(BinaryRating = if_else(FakeZRating >= 0, "AcceptableZ", "UnacceptableZ")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == "Eggshells are fragile") %>% 
  group_by(BinaryRating = if_else(FakeZRating >= 0, "AcceptableZ", "UnacceptableZ")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == "Trampolines are bouncy") %>% 
  group_by(BinaryRating = if_else(FakeZRating >= 0, "AcceptableZ", "UnacceptableZ")) %>%
  summarize(n())

an_fake %>% 
  filter(Property == "Trampolines are black") %>% 
  group_by(BinaryRating = if_else(FakeZRating >= 0, "AcceptableZ", "UnacceptableZ")) %>%
  summarize(n())
