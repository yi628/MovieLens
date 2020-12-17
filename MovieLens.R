############################################################################
# Data Preparation: Create edx set, validation set (final hold-out test set)
############################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

############################################################################
# Data Cleaning
############################################################################

# Review the distribution of numeric columns and number of zeros/NAs
summary(edx)

# Make sure there is not any zeros/NAs
colSums(is.na(edx))

############################################################################
# Data Exploration
############################################################################

# Check the content of a particular row
edx %>% head(5)

# Top users who give the most ratings
edx %>%
  group_by(userId) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(5)

# Top movies which get the most number of five-stars ratings
edx %>%
  filter(rating == 5) %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  head(5)

# Obtain the distribution of ratings
table(edx$rating)

# The earliest five ratings given
edx %>%
  mutate(datetime = as.POSIXct(timestamp, origin="1970-01-01")) %>%
  select(userId, title, rating, timestamp, datetime) %>%
  arrange(datetime) %>%
  head(5)

# Top movies which get the most number of ratings
edx %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(5)

# Top genres and their number of ratings
edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  head(5)

############################################################################
# Data Visualization
############################################################################

# rating pie chart
edx %>%
  group_by(rating) %>% 
  summarise(count = n()) %>%
  mutate(prop = count / sum(count) * 100) %>%
  mutate(rating = as.character(rating)) %>%
  ggplot(aes(x = "", y = count, fill = rating)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0) +
  geom_text(aes(label = paste0(round(prop), "%")), position = position_stack(vjust = 0.5)) +
  scale_fill_brewer(palette = "Set3")+
  theme_void()

# genres bar graph
edx %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  ggplot(aes(x = genres, y = count)) +
  geom_bar(stat="identity", width=0.7, fill="steelblue") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5)) +
  scale_y_continuous(labels = scales::comma)

############################################################################
# Modeling Approach
############################################################################

set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
train_set  <- edx[-test_index,]
temp       <- edx[test_index,]

# Make sure userId and movieId in train_set are also in test_set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test_set back into train_set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# Define the RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

############################################################################
# Model 1: Movie Effects Model
############################################################################

# Calculate the overall mean
mu <- mean(train_set$rating)

# Add movie bias
avg_movie <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Model 1 Prediction
predicted <- mu + test_set %>%
  left_join(avg_movie, by = 'movieId') %>%
  pull(b_i)

# Model 1 RMSE
model_1_rmse <- RMSE(test_set$rating, predicted)
model_1_rmse

############################################################################
# Model 2: Movie + User Effects Model
############################################################################

# Add user bias
avg_user <- train_set %>%
  left_join(avg_movie, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Model 2 Prediction
predicted <- test_set %>%
  left_join(avg_movie, by = 'movieId') %>%
  left_join(avg_user,  by = 'userId')  %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Model 2 RMSE
model_2_rmse <- RMSE(test_set$rating, predicted)
model_2_rmse

############################################################################
# Model 3: Movie + User-specific Genre Effects Model
############################################################################

# Add user-specific genre bias
avg_genre <- train_set %>%
  left_join(avg_movie, by = 'movieId') %>%
  left_join(avg_user,  by = 'userId')  %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(userId, genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# Model 3 Prediction
predicted <- test_set %>%
  separate_rows(genres, sep = "\\|") %>%
  left_join(avg_genre, by = c('userId','genres')) %>%
  group_by(userId, movieId, rating) %>%
  summarize(b_g = mean(b_g, na.rm = TRUE)) %>%
  left_join(avg_movie, by = 'movieId') %>%
  left_join(avg_user,  by = 'userId')  %>%
  mutate(pred = sum(mu, b_i, b_u, b_g, na.rm = TRUE)) %>%
  pull(pred)

# Model 3 RMSE
model_3_rmse <- RMSE(test_set$rating, predicted)
model_3_rmse

############################################################################
# Result: Final Validation
############################################################################

# Update the overall mean
mu <- mean(edx$rating)

# Update avg_movie
avg_movie <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Update avg_user
avg_user <- edx %>%
  left_join(avg_movie, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Update avg_genre
avg_genre <- edx %>%
  left_join(avg_movie, by = 'movieId') %>%
  left_join(avg_user,  by = 'userId')  %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(userId, genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# Final Prediction
predicted <- validation %>%
  separate_rows(genres, sep = "\\|") %>%
  left_join(avg_genre, by = c('userId','genres')) %>%
  group_by(userId, movieId, rating) %>%
  summarize(b_g = mean(b_g, na.rm = TRUE)) %>%
  left_join(avg_movie, by = 'movieId') %>%
  left_join(avg_user,  by = 'userId')  %>%
  mutate(pred = sum(mu, b_i, b_u, b_g, na.rm = TRUE)) %>%
  pull(pred)

# Final RMSE
final_rmse <- RMSE(validation$rating, predicted)
final_rmse



