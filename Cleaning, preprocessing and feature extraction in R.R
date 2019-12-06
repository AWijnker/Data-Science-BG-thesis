## loading required packages
library(tidyverse)
library(dplyr)
library(sqldf)
library(lattice)
library(ggplot2)
library(caret)
library(lubridate)
library(reshape)
library(RColorBrewer)
#install.packages("arulesSequences")
library(arulesSequences)
library(tm)
#library(plyr)

##loading the data

setwd("~/Master Data Science/AAA Thesis/source")

mood_sampling_data <- read.delim("input/mood_sampling_data.csv", stringsAsFactors = FALSE, sep = ",")
phone_use_data <- read.delim("input/phone_use_data.csv", stringsAsFactors = FALSE, sep = ",")
app_categories <- read.delim("input/app_categories.csv", stringsAsFactors = FALSE, sep = ",")
names(app_categories)[1] <- "application"


## adding the catgories to the phone use data
phone_use_data_categories <- sqldf("select * from phone_use_data 
                                   left join app_categories 
                                   on phone_use_data.application = app_categories.application",
                                   row.names = TRUE)
na_category <- as.data.frame(table(is.na(phone_use_data_categories$better_category)))

n_distinct(phone_use_cat_clean$user_id)

##General data information


##overview of the different categories
overview_categories <- as.data.frame(table(phone_use_data_categories$category))
overview_apps <- as.data.frame(table(phone_use_data_categories$application))


##analysing the apps without category (NAs)
apps_without_category <- phone_use_data_categories$application[phone_use_data_categories$application 
                                                               %!in% app_categories$application]

apps_without_category <- as.data.frame(table(apps_without_category))
apps_without_category <- apps_without_category %>% arrange(desc(Freq))
top_10_without_category <- apps_without_category[1:10,]

write.csv(top_10_without_category, "top_10_without_category.csv", row.names = FALSE)

top_10_without_category$apps_without_category <- revalue(top_10_without_category$apps_without_category, 
                                   c("com.lbe.parallel.intl.arm64"= "Parallel Space", 
                                     "com.android.systemui" = "Andriod SystemUI",
                                     "com.ethica.logger" = "Ethica",
                                     "nl.caci.osiris.student.tiu" = "Tilburg Osiris",
                                     "com.rhapsody.napster" = "Napster",
                                     "com.google.android.apps.nexuslauncher" = "Nexus Launcher",
                                     "com.deliveroo.driverapp" = "Deliveroo Rider",
                                     "mong.moptt" = "Mo PTT",
                                     "air.eu.bandainamcoent.asterixandfriends" = "Asterix and Friends",
                                     "com.ikeyboard.theme.blue.love.heart" = "IKeyboard blue love heart theme"
                                     ))

##creating barplot of top 10 most frequently used apps without label
without_category_plot <- top_10_without_category %>% ggplot(aes(x = reorder(apps_without_category, Freq), 
                                                                y = Freq)) + 
  geom_col(fill = "darkblue") +
  ggtitle("Frequency of top 10 apps without category") +
  labs(x = "App names", y = "Frequency") +
  coord_flip() + 
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
without_category_plot

mean(apps_without_category$Freq)
apps_without_10 <- tail(apps_without_category, -10)
mean(apps_without_10$Freq)

sum(top_10_without_category$Freq)

##creating new categorisation method (using Zipfian distribution as explanation)
apps_table <- as.data.frame(table(phone_use_data_categories$application))
apps_table <- apps_table[order(-apps_table$Freq),]

write.csv(apps_table, "zipfian_for_apps.csv", row.names = FALSE)

###check for zipfian distribution (ADD ROW NAMES)
table_apps <- as.data.frame(table(phone_use_data_categories$better_category))
table_apps <- table_apps[order(-table_apps$Freq),]

write.csv(table_apps, "zipfian_for_categories.csv", row.names = FALSE)

apps_plot <- ggplot(data = table_apps, aes(x = reorder(Var1, -Freq), y = Freq)) +
  geom_col(fill = "darkblue") +
  ggtitle("Frequency distribution of top 20 apps") +
  scale_y_continuous(limits = c(0, 125000)) +
  labs(x = "Apps", y = "Frequency") +
  theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))

apps_plot



##treating the NAs and creatin new categories, with 10 most used apps as theri own app 
app_categories <- app_categories[!(app_categories$application == "com.whatsapp" | 
                                     app_categories$application == "com.android.systemui"| 
                                     app_categories$application == "com.instagram.android"| 
                                     app_categories$application == "com.snapchat.android"| 
                                     app_categories$application == "com.android.chrome"| 
                                     app_categories$application == "com.facebook.katana"| 
                                     app_categories$application == "com.spotify.music"| 
                                     app_categories$application == "com.google.android.youtube" 
                                     | app_categories$application == "com.ethica.logger"),]

app_categories <- app_categories %>% add_row(application = "com.android.systemui", count = 100, name = "System UI", 
                           category = "Background Process", better_category_hybrid = "None", 
                           better_category = "Background_Process")  %>% 
  #add_row(application = "com.ethica.logger", count = 100, name = "Ethica", 
  #        category = "Tools", better_category_hybrid = "None", better_category = "Ethica") %>%
  add_row(application = "com.lbe.parallel.intl.arm64", count = 100, name = "Parallel Space", 
          category = "Tools", better_category_hybrid = "None", better_category = "Phone_Optimization") %>%
  add_row(application = "nl.caci.osiris.student.tiu", count = 100, name = "Osiris", 
          category = "Education", better_category_hybrid = "None", better_category = "Education") %>%
  add_row(application = "com.ikeyboard.theme.blue.love.heart", count = 100, 
          name = "IKeyboard blue love heart theme", 
          category = "Phone_Personalization", better_category_hybrid = "Phone_Personalization", 
          better_category = "Phone_Personalization") %>%
  add_row(application = "com.rhapsody.napster", count = 100, name = "Napster", 
          category = "Music & Audio", better_category_hybrid = "None", better_category = "Streaming_Services") %>%
  add_row(application = "com.google.android.apps.nexuslauncher", count = 100, name = "Nexus launcher", 
          category = "Personalization", better_category_hybrid = "None", better_category = "Phone_Personalization") %>%
  add_row(application = "com.deliveroo.driverapp", count = 100, name = "Deliveroo Driver App", 
          category = "Business", better_category_hybrid = "None", better_category = "Business_Management") %>%
  add_row(application = "mong.moptt", count = 100, name = "Mo PTT", 
          category = "Social", better_category_hybrid = "None", better_category = "Social_Networking") %>%
  add_row(application = "air.eu.bandainamcoent.asterixandfriends", count = 100, name = "Asterix and Friends", 
          category = "Role Playing", better_category_hybrid = "None", better_category = "Game_Multiplayer") %>%
  add_row(application = "com.whatsapp", count = 100, name = "WhatsApp", 
          category = "WhatsApp_Messenger", better_category_hybrid = "None", better_category = "Whatsapp_Messenger") %>%
  add_row(application = "com.instagram.android", count = 100, name = "Instagram", 
          category = "Instagram", better_category_hybrid = "None", better_category = "Instagram") %>%
  add_row(application = "com.snapchat.android", count = 100, name = "Snapchat", 
          category = "Snapchat", better_category_hybrid = "None", better_category = "Snapchat") %>%
  add_row(application = "com.android.chrome", count = 100, name = "Google Chrome", 
          category = "Google_Chrome", better_category_hybrid = "None", better_category = "Google_Chrome") %>%
  add_row(application = "com.facebook.katana", count = 100, name = "Facebook", 
          category = "Facebook", better_category_hybrid = "None", better_category = "Facebook") %>%
  add_row(application = "com.spotify.music", count = 100, name = "Spotify", 
          category = "Spotify", better_category_hybrid = "None", better_category = "Spotify") %>%
  add_row(application = "com.google.android.youtube", count = 100, name = "Youtube", 
          category = "Youtube", better_category_hybrid = "None", better_category = "Youtube") #%>%
  #add_row(application = "com.ethica.logger", count = 100, name = "Ethica", 
  #        category = "Ethica", better_category_hybrid = "None", better_category = "Tools") #%>%
  #add_row(application = "com.facebook.orca", count = 100, name = "Facebook_Messenger", 
  #        category = "Facebook_Messenger", better_category_hybrid = "None", better_category = "Facebook_Messenger")
  
phone_use_data_categories <- merge(phone_use_data, app_categories, 
                                   by=c("application"))
phone_use_data_categories <- distinct(phone_use_data_categories)
sum(!complete.cases(phone_use_data_categories))
count(distinct(phone_use_data_categories))
count(unique(phone_use_data_categories))

###check for zipfian distribution
table_apps <- as.data.frame(table(phone_use_data_categories$better_category))
table_apps <- table_apps[order(-table_apps$Freq),]

write.csv(table_apps, "zipfian_after_rearranging.csv", row.names = FALSE)

apps_plot <- ggplot(data = table_apps, aes(x = reorder(Var1, -Freq), y = Freq)) +
  geom_col(fill = "darkblue") +
  ggtitle("Frequency distribution of top 20 categories") +
  labs(x = "Apps", y = "Frequency") +
  scale_y_continuous(limits = c(0, 125000)) +
  theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))

apps_plot


##outlier analysis in stress
overview_stress <- as.data.frame(table(mood_sampling_data$stressed))
mood_sampling_data_new <- subset(mood_sampling_data, stressed <= 5 | is.na(stressed))
mood_sampling_data_stress <- as.data.frame(table(mood_sampling_data_new$stressed))

ggplot(data = mood_sampling_data_stress, aes(x = Var1, y = Freq)) +
  geom_bar(stat = "identity", fill = "darkblue") +
  geom_text(aes(label = Freq), vjust = 2, color = "white", size = 3) +
  xlab("Stress Level") +
  ylab("Frequency") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))

##NAs in stress
overview_stress_NA <- as.data.frame(table(is.na(mood_sampling_data$stressed)))
first_percentage_NA_stress <- (overview_stress_NA$Freq[2] / 
                                 (overview_stress_NA$Freq[1] + overview_stress_NA$Freq[2])) * 100






#---------------------------------------------APP DURATION---------------------------------------------------




##creating app duration variable
phone_use_data_categories$endTime <- ymd_hms(phone_use_data_categories$endTime)
phone_use_data_categories$startTime <- ymd_hms(phone_use_data_categories$startTime)    

phone_use_data_categories$duration <- phone_use_data_categories$endTime - phone_use_data_categories$startTime

phone_use_data_categories$duration <- as.numeric(phone_use_data_categories$duration)

subset_duration_index <- createDataPartition(phone_use_data_categories$duration, p = 0.001, list = FALSE)
subset_duration <- phone_use_data_categories[subset_duration_index,]

duration_plot <- ggplot(data = subset_duration, aes(duration)) +
  geom_area(stat = "bin") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
duration_plot

mean(phone_use_data_categories$duration)
sd(phone_use_data_categories$duration)





#write.table(phone_use_data_categories, "data_for_duration.csv", sep = ",", row.names = FALSE, quote = FALSE)

#---------------------------------------------STRESS PER DAY--------------------------------------------------
##creating stress per day variable


#first, split the time, so that days are a column on their own
mood_sampling_data_new <- separate(mood_sampling_data_new, response_time,
                                   c("response_date", "response_time", NA), sep = " ")
mood_sampling_data_new$response_date <- ymd(mood_sampling_data_new$response_date)


# second, create stress averages per day
mood_averages <- mood_sampling_data_new %>% group_by(user_id, response_date) %>%
  summarise_all(mean, na.rm = TRUE)
mood_averages <- subset(mood_averages, select = c("user_id", "response_date", "stressed"))
mood_averages$stressed <- as.factor(round(mood_averages$stressed, 0))
mood_
#write.csv(mood_averages, "mood_averages.csv", row.names = FALSE, quote = FALSE)

second_overview_stress_NaN <- as.data.frame(table(mood_averages$stressed))
second_percentage_NaN_stress <- (second_overview_stress_NaN$Freq[7] / 
                                   sum(second_overview_stress_NaN$Freq)) * 100



# third, split time and date in the phone use data
phone_use_data_categories <- separate(phone_use_data_categories, startTime, 
                                      c("response_date", "response_time"), sep = " ")
phone_use_data_categories$response_date <- ymd(phone_use_data_categories$response_date)



# fourth, merge the stress data into the phone dataset
phone_use_data_categories <- merge(phone_use_data_categories, mood_averages, 
                                   by=c("user_id","response_date"))

# overview amount of NaN in data
table_phoneuse_stress <- as.data.frame(table(phone_use_data_categories$stressed))
phone_stress_NaN_percentage <- (table_phoneuse_stress$Freq[7] / 
                                  sum(table_phoneuse_stress$Freq)) * 100



## Adding binary stress levels

## stress level 0
phone_use_data_categories$s_level_0 <- ifelse(phone_use_data_categories$stressed == 0, 1, 0)

## stress level 1
phone_use_data_categories$s_level_1 <- ifelse(phone_use_data_categories$stressed == 1, 1, 0)

## stress level 2
phone_use_data_categories$s_level_2 <- ifelse(phone_use_data_categories$stressed == 2, 1, 0)

## stress level 3
phone_use_data_categories$s_level_3 <- ifelse(phone_use_data_categories$stressed == 3, 1, 0)

## stress level 4
phone_use_data_categories$s_level_4 <- ifelse(phone_use_data_categories$stressed == 4, 1, 0)

## stress level 5
phone_use_data_categories$s_level_5 <- ifelse(phone_use_data_categories$stressed == 5, 1, 0)





#------------------------------------CLEANING APP CATEGORIES------------------------------

##  clean phone_use_categories 
phone_use_cat_clean <- phone_use_data_categories[complete.cases(phone_use_data_categories),]
phone_use_cat_clean <- phone_use_cat_clean[(phone_use_cat_clean$stressed == "0" | 
                                              phone_use_cat_clean$stressed == "1" | 
                               phone_use_cat_clean$stressed == "2" | phone_use_cat_clean$stressed == "3" | 
                               phone_use_cat_clean$stressed == "4" | phone_use_cat_clean$stressed == "5"),]


## remove background process apps (not deliberately opened by users)
phone_use_cat_clean <- phone_use_cat_clean[!(phone_use_cat_clean$better_category == "Background_Process" | 
                                               phone_use_cat_clean$category == "Background Process"),]


##based on findings in the boa model, some categories will be combined, due to low frequency rates
phone_use_cat_clean$better_category[phone_use_cat_clean$better_category == "Job_Search"] <- 
  "Social_Networking"
phone_use_cat_clean$better_category[which(phone_use_cat_clean$better_category == "Messages")] <- 
  "Messaging"
phone_use_cat_clean$better_category[which(phone_use_cat_clean$better_category == "Music_&_Audio")] <- 
  "Music_Audio"
phone_use_cat_clean <- phone_use_cat_clean[complete.cases(phone_use_cat_clean),]
#write.csv(phone_use_cat_clean, "phone_use_cat_clean.csv", sep = " ", row.names = FALSE, quote = FALSE)

#---------------------------------- creating app duration df (extra, not used eventually)--------------------

phone_use_cat_clean$duration_cat[phone_use_cat_clean$duration < 5] <- "very_short"
phone_use_cat_clean$duration_cat[phone_use_cat_clean$duration > 5 & phone_use_cat_clean$duration <= 57.5] <- 
  "short"
phone_use_cat_clean$duration_cat[phone_use_cat_clean$duration > 57.5 & phone_use_cat_clean$duration <= 300] <- 
  "middle"
phone_use_cat_clean$duration_cat[phone_use_cat_clean$duration > 300 & phone_use_cat_clean$duration <= 1200] <- 
  "long"
phone_use_cat_clean$duration_cat[phone_use_cat_clean$duration > 1200] <- "very_long"

barplot(table(phone_use_cat_clean$duration_cat))

## very low
phone_use_cat_clean$very_short <- ifelse(phone_use_cat_clean$duration_cat == "very_short", 1, 0)

## low
phone_use_cat_clean$short <- ifelse(phone_use_cat_clean$duration_cat == "short", 1, 0)

## middle
phone_use_cat_clean$middle <- ifelse(phone_use_cat_clean$duration_cat == "middle", 1, 0)

## high
phone_use_cat_clean$long <- ifelse(phone_use_cat_clean$duration_cat == "long", 1, 0)

## very high
phone_use_cat_clean$very_long <- ifelse(phone_use_cat_clean$duration_cat == "very_long", 1, 0)


duration_count <- phone_use_cat_clean %>% group_by(user_id, response_date) %>%
  summarise_at(vars(very_short, short, middle, long, very_long), funs(sum))
duration_count <- as.data.frame(duration_count)
duration_count$response_date <- as.character(duration_count$response_date)

write.csv(duration_count, "duration_count.csv", row.names = FALSE, quote = FALSE)

#----------------------MORNING AND EVENING (extra, not used eventually)-----------------------------------


## adding stress level per time of day (morning and evening)

mood_sampling_data_new$response_time <- as.numeric(gsub(":","",mood_sampling_data_new$response_time))  

## creating binary varibales to indicate morning and evening
mood_sampling_data_new$morning <- ifelse(mood_sampling_data_new$response_time < 120000 & 
                                           mood_sampling_data_new$response_time >= 060000, 1, 0) 
mood_sampling_data_new$evening <- ifelse(mood_sampling_data_new$response_time >= 180000 & 
                                           mood_sampling_data_new$response_time <= 235959, 1, 0)


## creating dfs of average per user per part of day

##morning (6am till 12pm)
stress_morning_averages <- mood_sampling_data_new %>% group_by(user_id, response_date, morning) %>%
  summarise_all(mean, na.rm = TRUE)
stress_morning_averages <- subset(stress_morning_averages, select = 
                                    c("user_id", "response_date", "morning", "stressed"))
names(stress_morning_averages)[4] <- "stressed_morning"
stress_morning_averages <- subset(stress_morning_averages, stress_morning_averages$morning == 1)
stress_morning_averages$morning <- NULL
stress_morning_averages <- stress_morning_averages[complete.cases(stress_morning_averages),]

##evening (6pm till 12am)
stress_evening_averages <- mood_sampling_data_new %>% group_by(user_id, response_date, evening) %>% 
  summarise_all(mean, na.rm = TRUE)
stress_evening_averages <- subset(stress_evening_averages, select = 
                                    c("user_id", "response_date", "evening", "stressed"))
names(stress_evening_averages)[4] <- "stressed_evening"  
stress_evening_averages <- subset(stress_evening_averages, stress_evening_averages$evening == 1) 
stress_evening_averages$evening <- NULL
stress_evening_averages <- stress_evening_averages[complete.cases(stress_evening_averages),]


## prepare phone use data categories set for subsetting into morning and evening
phone_use_cat_clean$response_time <- as.numeric(gsub(":","",phone_use_cat_clean$response_time))

phone_use_cat_clean$morning <- ifelse(phone_use_cat_clean$response_time < 120000 & 
                                        phone_use_cat_clean$response_time >= 060000, 1, 0) 
phone_use_cat_clean$evening <- ifelse(phone_use_cat_clean$response_time >= 180000 & 
                                        phone_use_cat_clean$response_time <= 235959, 1, 0)

#create new df with morning phone use + stress data
phone_use_morning_stress <- subset(phone_use_cat_clean, phone_use_cat_clean$morning == 1)
phone_use_morning_stress <- merge(phone_use_morning_stress, stress_morning_averages,
                                  by=c("user_id","response_date"))
morning_columns <- c("user_id", "response_date", "response_time", "application", "name", "category", 
                     "better_category", "stressed_morning")
phone_use_morning_stress <- phone_use_morning_stress[,morning_columns]
phone_use_morning_stress$stressed_morning <- as.factor(
  round(phone_use_morning_stress$stressed_morning , 0))

# create new df with evening phone use + stress data
phone_use_evening_stress <- subset(phone_use_cat_clean, phone_use_cat_clean$evening == 1)
phone_use_evening_stress <- merge(phone_use_evening_stress, stress_evening_averages,
                                  by=c("user_id","response_date"))
evening_columns <- c("user_id", "response_date", "response_time", "application", "name", "category", 
                     "better_category", "stressed_evening")
phone_use_evening_stress <- phone_use_evening_stress[,evening_columns]
phone_use_evening_stress$stressed_evening <- as.factor(
  round(phone_use_evening_stress$stressed_evening , 0))

#--------------------------------------#BOA MODEL TRAINING---------------------------------------


## creating bag of apps model


#install.packages("tm")

boa_sequences <- phone_use_cat_clean %>% group_by(user_id, response_date, stressed) %>%
  dplyr::summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
boa_sequences$size <- NULL

boa_apps <- boa_sequences$apps
boa_source <- VectorSource(boa_apps)
boa_corpus <- VCorpus(boa_source)

boa_document_term <- DocumentTermMatrix(boa_corpus)
boa_df <- as.data.frame(as.matrix(boa_document_term))

##adding the data together, needed for prediction model
boa_sequences$apps <- NULL
boa_sequences$counter <- seq.int(nrow(boa_sequences))
boa_df$counter <- seq.int(nrow(boa_df))
boa_final <- merge(boa_sequences, boa_df, by = "counter")
boa_final <- select(boa_final, -(counter:response_date))

sum_boa_final <- summarise_all(boa_final, sum)
sum_boa_final <- as.data.frame(t(sum_boa_final))
row_names <- rownames(sum_boa_final)
sum_boa_final$apps <- row_names
sum_boa_final <- sum_boa_final[order(-sum_boa_final$V1),] %>% head(20)

boa_features_plot <- ggplot(data = sum_boa_final, aes(x = reorder(apps, -V1), y = V1)) +
  geom_col(fill = "darkblue") +
  ggtitle("Frequency distribution of top 20 BoA features") +
  #scale_y_continuous(limits = c(0, 125000)) +
  labs(x = "Apps", y = "Frequency") +
  theme(axis.ticks.x=element_blank(), axis.text.x=element_blank(),
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))

boa_features_plot



set.seed(2019)
boa_index <- createDataPartition(y = c(boa_final$stressed), p = 0.7, list = FALSE)
boa_train <- boa_final[boa_index,]
boa_test <- boa_final[-boa_index,]




boa_train_x <- select(boa_train, -(stressed))
boa_train_y <- select(boa_train, stressed)
write.csv(boa_train_x, "method/boa_dur_train_x.csv", sep=",", row.names = FALSE)
write.csv(boa_train_y, "method/boa_dur_train_y.csv", sep=",", row.names = FALSE)


boa_test_x <- select(boa_test, -(stressed))
boa_test_y <- select(boa_test, stressed)
write.csv(boa_test_x, "method/boa_dur_test_x.csv", sep=",", row.names = FALSE)
write.csv(boa_test_y, "method/boa_dur_test_y.csv", sep=",", row.names = FALSE)


#------------------------ adding app duration to boa model----------------------
duration_count <- read.delim("duration_count.csv", sep = ",", stringsAsFactors = FALSE)

boa_final$response_date <- as.character(boa_final$response_date)
boa_finally <- merge(boa_final, duration_count, by = c("user_id", "response_date"))

boa_finally <- select(boa_finally, -(user_id:counter))

boa_finally$very_short <- as.factor(boa_finally$very_short)
boa_finally$short <- as.factor(boa_finally$short)
boa_finally$middle <- as.factor(boa_finally$middle)
boa_finally$long <- as.factor(boa_finally$long)
boa_finally$very_long <- as.factor(boa_finally$very_long)

boa_final[boa_final >0 & boa_final <= 8] <- 1
boa_final[boa_final >8 & boa_final <= 16] <- 2
boa_final[boa_final > 16] <- 3

boa_final <- na.omit(boa_final)

#------------------------ test train split -------------------------------------

boa_x <- select(boa_final, -(stressed))
boa_y <- select(boa_final, stressed)
write.csv(boa_x, "method/boa_x.csv", sep=",", row.names = FALSE)
write.csv(boa_y, "method/boa_y.csv", sep=",", row.names = FALSE)


#-----------------------------------------------------#CSPADE MODEL----------------------------
## cspade for the categorical patterns

## with all the data
cspade_input<- phone_use_cat_clean %>% group_by(user_id, response_date) %>%
  dplyr::summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_input$response_date <- gsub("-","",cspade_input$response_date)
cspade_input$size <- NULL
cspade_input$SIZE <- sapply(cspade_input$apps, function(x) length(unlist(strsplit(as.character(x), 
                                                                                  "\\W+"))))
cspade_input <- cspade_input[,c(1,2,4,3)]
names(cspade_input)[1] <- "sequenceID"
names(cspade_input)[2] <- "eventID"
cspade_input$eventID <- as.integer(cspade_input$eventID)
cspade_input$apps <- as.factor(cspade_input$apps)

write.table(cspade_input, "cspade_input.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_stress_ready <- read_baskets("cspade_input.txt", info = c("sequenceID","eventID", "SIZE"))
spade_stress_ready.df <- as(spade_stress_ready, "data.frame")

# run cspade
cspade_stress_categorical <- cspade(spade_stress_ready, parameter = list(
  support = 0.9, maxsize = 1, maxlen = 6), control = list(verbose = TRUE))
cspade_stress_categorical.df <- as(cspade_stress_categorical, "data.frame")

write.csv(cspade_stress_categorical.df, "method/spm_daily_patterns_0.9_6.csv", row.names = FALSE)


#-------------------------CSPADE FOR MORNING (extra, not used)------------------------------------
cspade_morning <- phone_use_morning_stress %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_morning$response_date <- gsub("-","",cspade_morning$response_date)
cspade_morning$size <- NULL
cspade_morning$SIZE <- sapply(cspade_morning$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_morning <- cspade_morning[,c(1,2,4,3)]
names(cspade_morning)[1] <- "sequenceID"
names(cspade_morning)[2] <- "eventID"
cspade_morning$eventID <- as.integer(cspade_morning$eventID)
cspade_morning$apps <- as.factor(cspade_morning$apps)

write.table(cspade_morning, "cspade_morning.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_morning_ready <- read_baskets("cspade_morning.txt", info = c("sequenceID","eventID", "SIZE"))


# run cspade
cspade_stress_morning <- cspade(spade_morning_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 5), control = list(verbose = TRUE))
cspade_stress_morning.df <- as(cspade_stress_morning, "data.frame")

write.csv(cspade_stress_morning.df, "spm_morning_patterns.csv", row.names = FALSE)


#---------------------------CSPADE EVENING (extra, not used)-------------------------------------
cspade_evening <- phone_use_evening_stress %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_evening$response_date <- gsub("-","",cspade_evening$response_date)
cspade_evening$size <- NULL
cspade_evening$SIZE <- sapply(cspade_evening$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_evening <- cspade_evening[,c(1,2,4,3)]
names(cspade_evening)[1] <- "sequenceID"
names(cspade_evening)[2] <- "eventID"
cspade_evening$eventID <- as.integer(cspade_evening$eventID)
cspade_evening$apps <- as.factor(cspade_evening$apps)

write.table(cspade_evening, "cspade_evening.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_evening_ready <- read_baskets("cspade_evening.txt", info = c("sequenceID","eventID", "SIZE"))

# run cspade
cspade_stress_evening <- cspade(spade_evening_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 4), control = list(verbose = TRUE))
cspade_stress_evening.df <- as(cspade_stress_evening, "data.frame")

write.csv(cspade_stress_evening.df, "spm_evening_patterns.csv", row.names = FALSE)

#----------------------------------------CSPADE FOR DAILY STRESS LEVEL 0 (extra, not used)----------------------------------
phone_use_0 <- subset(phone_use_cat_clean, phone_use_cat_clean$s_level_0 == 1)

cspade_0 <- phone_use_0 %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_0$response_date <- gsub("-","",cspade_0$response_date)
cspade_0$size <- NULL
cspade_0$SIZE <- sapply(cspade_0$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_0 <- cspade_0[,c(1,2,4,3)]
names(cspade_0)[1] <- "sequenceID"
names(cspade_0)[2] <- "eventID"
cspade_0$eventID <- as.integer(cspade_0$eventID)
cspade_0$apps <- as.factor(cspade_0$apps)

write.table(cspade_0, "cspade_0.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_0_ready <- read_baskets("cspade_0.txt", info = c("sequenceID","eventID", "SIZE"))

# run cspade
cspade_stress_0 <- cspade(spade_0_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 5), control = list(verbose = TRUE))
cspade_stress_0.df <- as(cspade_stress_0, "data.frame")

write.csv(cspade_stress_0.df, "spm_0_patterns.csv", row.names = FALSE)


#----------------------------------------CSPADE FOR DAILY STRESS LEVEL 1 (extra, not used)----------------------------------
phone_use_1 <- subset(phone_use_cat_clean, phone_use_cat_clean$s_level_1 == 1)

cspade_1 <- phone_use_1 %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_1$response_date <- gsub("-","",cspade_1$response_date)
cspade_1$size <- NULL
cspade_1$SIZE <- sapply(cspade_1$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_1 <- cspade_1[,c(1,2,4,3)]
names(cspade_1)[1] <- "sequenceID"
names(cspade_1)[2] <- "eventID"
cspade_1$eventID <- as.integer(cspade_1$eventID)
cspade_1$apps <- as.factor(cspade_1$apps)

write.table(cspade_1, "cspade_1.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_1_ready <- read_baskets("cspade_1.txt", info = c("sequenceID","eventID", "SIZE"))

# run cspade
cspade_stress_1 <- cspade(spade_1_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 5), control = list(verbose = TRUE))
cspade_stress_1.df <- as(cspade_stress_1, "data.frame")

write.csv(cspade_stress_1.df, "spm_1_patterns.csv", row.names = FALSE)


#----------------------------------------CSPADE FOR DAILY STRESS LEVEL 2 (extra, not used)----------------------------------
phone_use_2 <- subset(phone_use_cat_clean, phone_use_cat_clean$s_level_2 == 1)

cspade_2 <- phone_use_2 %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_2$response_date <- gsub("-","",cspade_2$response_date)
cspade_2$size <- NULL
cspade_2$SIZE <- sapply(cspade_2$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_2 <- cspade_2[,c(1,2,4,3)]
names(cspade_2)[1] <- "sequenceID"
names(cspade_2)[2] <- "eventID"
cspade_2$eventID <- as.integer(cspade_2$eventID)
cspade_2$apps <- as.factor(cspade_2$apps)

write.table(cspade_2, "cspade_2.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_2_ready <- read_baskets("cspade_2.txt", info = c("sequenceID","eventID", "SIZE"))

# run cspade
cspade_stress_2 <- cspade(spade_2_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 5), control = list(verbose = TRUE))
cspade_stress_2.df <- as(cspade_stress_2, "data.frame")

write.csv(cspade_stress_2.df, "spm_2_patterns.csv", row.names = FALSE)

#----------------------------------------CSPADE FOR DAILY STRESS LEVEL 3 (extra, not used)----------------------------------
phone_use_3 <- subset(phone_use_cat_clean, phone_use_cat_clean$s_level_3 == 1)

cspade_3 <- phone_use_3 %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_3$response_date <- gsub("-","",cspade_3$response_date)
cspade_3$size <- NULL
cspade_3$SIZE <- sapply(cspade_3$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_3 <- cspade_3[,c(1,2,4,3)]
names(cspade_3)[1] <- "sequenceID"
names(cspade_3)[2] <- "eventID"
cspade_3$eventID <- as.integer(cspade_3$eventID)
cspade_3$apps <- as.factor(cspade_3$apps)

write.table(cspade_3, "cspade_3.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_3_ready <- read_baskets("cspade_3.txt", info = c("sequenceID","eventID", "SIZE"))

# run cspade
cspade_stress_3 <- cspade(spade_3_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 5), control = list(verbose = TRUE))
cspade_stress_3.df <- as(cspade_stress_3, "data.frame")

write.csv(cspade_stress_3.df, "spm_3_patterns.csv", row.names = FALSE)

#----------------------------------------CSPADE FOR DAILY STRESS LEVEL 4 (extra, not used)----------------------------------
phone_use_4 <- subset(phone_use_cat_clean, phone_use_cat_clean$s_level_4 == 1)

cspade_4 <- phone_use_4 %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_4$response_date <- gsub("-","",cspade_4$response_date)
cspade_4$size <- NULL
cspade_4$SIZE <- sapply(cspade_4$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_4 <- cspade_4[,c(1,2,4,3)]
names(cspade_4)[1] <- "sequenceID"
names(cspade_4)[2] <- "eventID"
cspade_4$eventID <- as.integer(cspade_4$eventID)
cspade_4$apps <- as.factor(cspade_4$apps)

write.table(cspade_4, "cspade_4.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_4_ready <- read_baskets("cspade_4.txt", info = c("sequenceID","eventID", "SIZE"))

# run cspade
cspade_stress_4 <- cspade(spade_4_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 5), control = list(verbose = TRUE))
cspade_stress_4.df <- as(cspade_stress_4, "data.frame")

write.csv(cspade_stress_4.df, "spm_4_patterns.csv", row.names = FALSE)

#----------------------------------------CSPADE FOR DAILY STRESS LEVEL 5 (extra, not used)----------------------------------
phone_use_5 <- subset(phone_use_cat_clean, phone_use_cat_clean$s_level_5 == 1)

cspade_5 <- phone_use_5 %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
cspade_5$response_date <- gsub("-","",cspade_5$response_date)
cspade_5$size <- NULL
cspade_5$SIZE <- sapply(cspade_5$apps, function(x) length(unlist(strsplit(as.character(x), "\\W+"))))
cspade_5 <- cspade_5[,c(1,2,4,3)]
names(cspade_5)[1] <- "sequenceID"
names(cspade_5)[2] <- "eventID"
cspade_5$eventID <- as.integer(cspade_5$eventID)
cspade_5$apps <- as.factor(cspade_5$apps)

write.table(cspade_5, "cspade_5.txt", sep=" ", row.names = FALSE, col.names = FALSE)

spade_5_ready <- read_baskets("cspade_5.txt", info = c("sequenceID","eventID", "SIZE"))

# run cspade
cspade_stress_5 <- cspade(spade_5_ready, parameter = list(support = 0.3, maxsize = 1, maxlen = 5), control = list(verbose = TRUE))
cspade_stress_5.df <- as(cspade_stress_5, "data.frame")

write.csv(cspade_stress_5.df, "spm_5_patterns.csv", row.names = FALSE)


#----------------------------------------APRIORI ALGORITHM (extra, not used)-------------------------------------------------

apriori_sequences <- phone_use_cat_clean %>% group_by(user_id, response_date) %>%
  summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
#apriori_sequences <- apriori_sequences[,-c(1,2,3)]
#apriori_sequences$apps <- gsub(" ",",",apriori_sequences$apps)
#apriori_sequences <- data.frame(lapply(apriori_sequences, as.factor))

#apriori_split <- do.call("rbind", strsplit(apriori_sequences$apps, ","))
#apriori_input <- data.frame(apply(apriori_split, 2, as.factor))

write.csv(apriori_sequences,"apriori_input.csv", sep = " ", row.names = FALSE, quote = FALSE)
apriori_sequences<- read.transactions("apriori_input.csv", sep = ",", format = "basket")

apriori_rules <- apriori(apriori_sequences, parameter = list(supp = 0.3, target="rules", minlen=2))
summary(apriori_rules)



#---------------------------------CLEAN AND PREPARE RULES---------------------------


spm_daily_patterns <- read.delim("input/spm_daily_patterns_0.8_4.csv", stringsAsFactors = FALSE, 
                                 sep = ",")

spm_daily_patterns$sequence <- gsub("\\{","",spm_daily_patterns$sequence)
spm_daily_patterns$sequence <- gsub("\\}","",spm_daily_patterns$sequence)
spm_daily_patterns$sequence <- gsub(">", "", spm_daily_patterns$sequence)
spm_daily_patterns$sequence <- gsub("<", "", spm_daily_patterns$sequence)
spm_daily_patterns <- as.data.frame(sapply(spm_daily_patterns, function(x) gsub("\"", "", x)))

#removing rows with single values
spm_daily_patterns <- spm_daily_patterns[grepl(",", spm_daily_patterns$sequence),]


#spm_daily_patterns <- add_case(spm_daily_patterns, sequence = "stressed", support = "0", 
.before = TRUE)
spm_daily_patterns <- add_case(spm_daily_patterns, sequence = "apps", support = "0", 
                               .before = TRUE)
spm_daily_patterns <- add_case(spm_daily_patterns, sequence = "size", support = "0", 
                               .before = TRUE)
spm_daily_patterns <- add_case(spm_daily_patterns, sequence = "response_date", support = "0", 
                               .before = TRUE)
spm_daily_patterns <- add_case(spm_daily_patterns, sequence = "user_id", support = "0", 
                               .before = TRUE)
spm_daily_patterns$support <- NULL
spm_daily_patterns$sequence <- gsub(",", " ", spm_daily_patterns$sequence)

#flip to make patterns columns
spm_daily_flipped <- as.data.frame(t(spm_daily_patterns))

#create another row
spm_daily_flipped <- spm_daily_flipped[rep(seq_len(nrow(spm_daily_flipped)), each = 2), ]

#create column names
colnames(spm_daily_flipped) <- as.character(unlist(spm_daily_flipped[1,]))

#delete extra row
spm_daily_flipped <- spm_daily_flipped[-1, ]


spm_daily_flipped_2 <- phone_use_cat_clean %>% group_by(user_id, response_date) %>%
  dplyr::summarise(size = n(), apps = paste(as.character(better_category), collapse = " "))
spm_daily_flipped_2 <-spm_daily_flipped_2[-c(1,2,3),]

#---------------------------------COMBINE DATASETS-------------------------------

library(plyr)

spm_daily_combined <- rbind.fill(spm_daily_flipped, spm_daily_flipped_2)


write.csv(spm_daily_combined, "method/spm_daily_combined_0.8_4.csv", row.names = FALSE, 
          quote = FALSE)
