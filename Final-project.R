##############################################################
# 1.Introduction
##############################################################


##############################################################
## MovieLens Dataset Overview
##############################################################

# Load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!requireNamespace("plotly", quietly = TRUE)) install.packages("plotly")
if (!requireNamespace("webshot", quietly = TRUE)) install.packages("webshot")
if (!requireNamespace("webshot2", quietly = TRUE)) install.packages("webshot2")
if (!requireNamespace("htmlwidgets", quietly = TRUE)) install.packages("htmlwidgets")
webshot::install_phantomjs()
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(patchwork)) install.packages("patchwork", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra")

library(tidyverse)
library(caret)
library(webshot)
library(webshot2)
library(htmlwidgets)
library(plotly)
library(scales)
library(patchwork)
library(kableExtra)

# Download MovieLens dataset if necessary
options(timeout = 120)
dl <- "ml-10M100K.zip"
if(!file.exists(dl)) download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file)) unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file)) unzip(dl, movies_file)

# Read and process ratings
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
# Read and process movies
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>% mutate(movieId = as.integer(movieId))
# Merge ratings and movies
movielens <- left_join(ratings, movies, by = "movieId")

##############################################################
## MovieLens Dataset Overview 2 (Data Split)
##############################################################

# Split the data into edx and final_holdout_test
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
final_holdout_test <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##############################################################
## MovieLens Dataset Overview 3 (Quick Overview)
##############################################################

cat("Number of lines in edx:", comma(nrow(edx)), "\n")
cat("Number of lines in final_holdout_test:", comma(nrow(final_holdout_test)), "\n")
cat("Number of distinct users:", comma(n_distinct(edx$userId)), "\n")
cat("Number of distinct movies:", comma(n_distinct(edx$movieId)), "\n")
cat("Number of distinct ratings (values):", comma(n_distinct(edx$rating)), "\n")
cat("The different possible ratings (values):", sort(unique(edx$rating)), "\n")
cat("Number of distinct timestamps:", comma(n_distinct(edx$timestamp)), "\n")
cat("Number of distinct movie titles:", comma(n_distinct(edx$title)), "\n")
cat("Number of distinct genre combinations:", comma(n_distinct(edx$genres)), "\n")

##############################################################
## 2.Methods and Data analysis
##############################################################

##############################################################
## 2.1 Data Preparation
##############################################################

##############################################################
## 2.1.1 Data Cleaning
##############################################################

#Before exploring the data or building any models, it is important to make sure that the dataset is clean and usable. In order to build solid models, it is important to rely in clean data.

##############################################################
# MovieLens Checking for missing values
##############################################################

colSums(is.na(edx))

##############################################################
# MovieLens Checking for duplicate rows
##############################################################

sum(duplicated(edx))

##############################################################
# MovieLens Checking column types
##############################################################

str(edx)

##############################################################
# MovieLens Checking rating values
##############################################################

sort(unique(edx$rating))

##############################################################
## 2..1.2 Creating new variables
##############################################################

# Once the dataset has been cleaned, additional variables can be created to enhance its usefulness. These new variables facilitate a more detailed description of the data and allow for the identification of relevant patterns. They are also expected to improve the performance of subsequent predictive modeling.

##############################################################
# MovieLens The new variable: release_year
##############################################################

edx <- edx %>%
  mutate(release_year = str_extract(title, "\\(\\d{4}\\)") %>%
           str_remove_all("[()]") %>%
           as.integer())
head(select(edx, title, release_year))

##############################################################
# MovieLens The new variable: rating_year
##############################################################

edx <- edx %>%
  mutate(rating_year = as.POSIXct(timestamp, origin = "1970-01-01") %>%
           format("%Y") %>%
           as.integer())
head(select(edx, timestamp, rating_year))

##############################################################
# MovieLens The new variable: main_genre
##############################################################

edx <- edx %>%
  mutate(main_genre = str_split(genres, "\\|", simplify = TRUE)[,1])
head(select(edx, genres, main_genre))

##############################################################
# MovieLens The new variable: user_frequency
##############################################################

user_activity <- edx %>%
  group_by(userId) %>%
  summarise(
    n_ratings = n(),
    first_rating = min(rating_year, na.rm = TRUE),
    last_rating = max(rating_year, na.rm = TRUE),
    active_years = last_rating - first_rating + 1,
    user_frequency = n_ratings / active_years,
    .groups = "drop"
  )
head(user_activity)


##############################################################
## 2.2 Data Exploration and Visualization
##############################################################

#With the dataset now cleaned and complete, further exploration can be conducted to better understand user rating behaviors. This step helps identify meaningful patterns and informs the selection of appropriate modeling approaches in subsequent analyses.

##############################################################
# 2.2.1 What do the ratings look like?
##############################################################

library(knitr)

# Group ratings by type and calculate percentage (rounded to whole number)
rating_summary <- edx %>%
  mutate(rating_type = ifelse(rating %% 1 == 0, "Rounded", "Half-point")) %>%
  count(rating_type, name = "Number of Ratings") %>%
  mutate(
    Percentage = round(`Number of Ratings` / sum(`Number of Ratings`) * 100)
  )

# Display formatted table
kable(rating_summary, caption = "Distribution of Rounded vs Half-point Ratings (in %)")

#The analysis of the table reveals that 80% of users give rounded ratings rather than half point ratings such as 1.5 and similar values.

##############################################################
# MovieLens What-do-the-ratings-look-like-the-graph
##############################################################

library(ggplot2)
library(dplyr)

# Prepare data with percentage
rating_dist <- edx %>%
  count(rating) %>%
  mutate(
    percentage = n / sum(n),
    label = percent(percentage, accuracy = 1)
  )

# Plot
ggplot(rating_dist, aes(x = factor(rating), y = percentage)) +
  geom_col(fill = "plum3") +
  geom_text(aes(label = label), vjust = -0.5, size = 3.5) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, max(rating_dist$percentage) + 0.05)) +
  labs(
    title = "Rating Distribution",
    x = "Rating",
    y = "Percentage of Ratings"
  ) +
  theme_minimal()

# The data indicates that the most frequently given rating is 4. However, approximately 24% of users remain neutral, assigning a rating of 3. Only 15% of users appear to be fully convinced by the movie, awarding it the maximum rating of 5.

##############################################################
# MovieLens What-do-the-ratings-look-like-the-graph-high-low-rates
##############################################################


# Group ratings into categories and control display order
edx_grouped <- edx %>%
  mutate(rating_group = case_when(
    rating <= 2.5 ~ "Low",
    rating == 3.0 ~ "Neutral",
    rating > 3.0  ~ "High"
  )) %>%
  mutate(rating_group = factor(rating_group, levels = c("Low", "Neutral", "High")))

# Summarize ratings
rating_group_summary <- edx_grouped %>%
  count(rating_group) %>%
  mutate(
    percentage = n / sum(n),
    percentage_label = percent(percentage, accuracy = 1),
    explanation = case_when(
      rating_group == "Low" ~ "Ratings ≤ 2.5",
      rating_group == "Neutral" ~ "Ratings = 3.0",
      rating_group == "High" ~ "Ratings > 3.0"
    )
  )

# Plot with internal explanation and top percentage
ggplot(rating_group_summary, aes(x = rating_group, y = percentage, fill = rating_group)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  
  # Label inside the bar — centered vertically
  geom_text(aes(y = percentage / 2, label = explanation), color = "white", size = 4) +
  
  # Percentage above the bar
  geom_text(aes(label = percentage_label), vjust = -1.5, size = 4.2, fontface = "bold") +
  
  scale_y_continuous(labels = percent_format(accuracy = 1),
                     limits = c(0, max(rating_group_summary$percentage) + 0.1)) +
  
  scale_fill_manual(values = c("Low" = "red3", "Neutral" = "grey70", "High" = "olivedrab3")) +
  
  labs(
    title = "Distribution of Ratings by Group",
    subtitle = "Including Definition and Percentage",
    x = "Rating Group",
    y = "Percentage of Ratings"
  ) +
  
  theme_minimal()

#The ratings were divided into three groups: Low (ratings ≤ 2.5), Neutral (ratings = 3.0), and High (ratings \> 3.0). This chart demonstrates that most users (59%) give high ratings, while low and neutral ratings are much less common. The clear difference between the groups confirms a positive bias in user behavior.

##############################################################
# 2.2.2 Which movies are rated the most?
##############################################################

#An examination of which movies receive the most ratings helps determine if a few popular movies dominate the dataset.

##############################################################
# MovieLens Which movies are rated the most-the-top-10
##############################################################

edx %>%
  count(title) %>%
  top_n(10, n) %>%
  ggplot(aes(x = reorder(title, n), y = n)) +
  geom_col(fill = "plum3") +
  coord_flip() +
  labs(title = "Top 10 Most Rated Movies", x = "Movie Title", y = "Number of Ratings") +
  theme_minimal()

#Some movies, like Pulp Fiction, Forrest Gump or The Silence of the Lambs, have thousands of ratings. These popular movies might influence the model more significantly than others.

##############################################################
# MovieLens Which movies are rated the most-the-graph-of-the-number-of ratings-per-movie
##############################################################

# # Step 1: Count the number of ratings per film + their average rating
movie_stats <- edx %>%
  group_by(title) %>%
  summarise(
    n_ratings = n(),
    avg_rating = mean(rating),
    .groups = "drop"
  )

# Step 2: Create groups of 2000 ratings
breaks <- seq(0, max(movie_stats$n_ratings) + 2000, by = 2000)
labels <- paste0("≤", comma(breaks[-1]))

movie_stats <- movie_stats %>%
  mutate(
    rating_bin = cut(
      n_ratings,
      breaks = breaks,
      labels = labels,
      include.lowest = TRUE
    )
  )

# Step 3: Summary by group
rating_bin_summary <- movie_stats %>%
  group_by(rating_bin) %>%
  summarise(
    number_of_movies = n(),
    avg_rating = mean(avg_rating),
    .groups = "drop"
  ) %>%
  mutate(
    proportion = number_of_movies / sum(number_of_movies)
  )

# Step 4: Combo Chart

ggplot(rating_bin_summary, aes(x = rating_bin)) +
  geom_col(aes(y = proportion), fill = "plum3", width = 0.7) +
  geom_line(aes(y = avg_rating / 5), color = "darkred", linewidth = 1.2, group = 1) +
  geom_point(aes(y = avg_rating / 5), color = "darkred", size = 2) +
  scale_y_continuous(
    name = "Percentage of Movies in Each Group",
    labels = percent_format(accuracy = 1),
    sec.axis = sec_axis(~ . * 5, name = "Average Rating", breaks = seq(0, 5, 0.5))
  ) +
  geom_text(
    aes(y = avg_rating / 5, label = round(avg_rating, 1)),
    vjust = -1.2,
    color = "darkred",
    size = 3.5
  ) +
  labs(
    title = "Distribution of Movies by Number of Ratings",
    subtitle = "Percentage of movies per rating group and their average score",
    x = "Number of Ratings (Grouped)",
    y = "Share of Movies"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#This chart demonstrates that the vast majority of movies in the dataset received fewer than 2,000 ratings. Moving to more popular films (over 10,000 or 30,000 ratings), the number of movies in each group becomes very small.

##############################################################
# MovieLens Which movies are rated the most-the-graph-of-the-number-of-ratings-per-movie-in-log-scale
##############################################################

# 1. Compute per‐movie stats
movie_stats <- edx %>%
  group_by(movieId) %>%
  summarise(
    n_ratings  = n(),
    avg_rating = mean(rating),
    .groups     = "drop"
  )

# 2. Define 20 log‐spaced breaks
breaks <- 10^seq(
  floor(log10(1)),
  ceiling(log10(max(movie_stats$n_ratings))),
  length.out = 20
)

# 3. Subset every 3rd break for axis labels
break_labels <- breaks[seq(1, length(breaks), by = 3)]

# 4. Bin the movies for the line/points
bin_summary <- movie_stats %>%
  mutate(bin = cut(
    n_ratings,
    breaks         = breaks,
    include.lowest = TRUE
  )) %>%
  group_by(bin) %>%
  summarise(
    movies_in_bin = n(),
    mean_rating   = mean(avg_rating),
    .groups       = "drop"
  ) %>%
  # compute numeric mid‐point of each bin
  mutate(
    lower = breaks[as.integer(bin)],
    upper = breaks[as.integer(bin) + 1],
    mid   = sqrt(lower * upper)
  )

# 5. Scaling factor for the secondary axis
scale_factor <- max(bin_summary$movies_in_bin) / 5  # ratings run ~0.5–5

# 6. Plot: histogram + line + points
ggplot() +
  # A) histogram of counts
  geom_histogram(
    data   = movie_stats,
    aes(x = n_ratings, y = after_stat(count)),
    breaks = breaks,
    fill   = "plum3",
    color  = "white"
  ) +
  # B) average rating trend
  geom_line(
    data = bin_summary,
    aes(x = mid, y = mean_rating * scale_factor),
    color = "firebrick",
    size  = 1
  ) +
  geom_point(
    data = bin_summary,
    aes(x = mid, y = mean_rating * scale_factor),
    color = "firebrick",
    size  = 2
  ) +
  geom_hline(
    yintercept = mean(movie_stats$avg_rating) * scale_factor,
    linetype   = "dashed",
    color      = "black"
  ) +
  # Left axis = number of movies; Right axis = average rating
  scale_y_continuous(
    name   = "Number of Movies",
    labels = comma_format(),
    sec.axis = sec_axis(
      ~ . / scale_factor,
      name   = "Average Rating",
      breaks = seq(0.5, 5, 0.5)
    )
  ) +
  # X axis on log10, labeled in k, label only every 3rd bin
  scale_x_log10(
    name   = "Number of Ratings per Movie (log10) in thousands (k)",
    limits = range(breaks),
    expand = c(0, 0),
    breaks = break_labels,  # <--- only every 3rd tick is labeled
    labels = label_number(scale = 1/1000, suffix = "k", accuracy = 0.01)
  ) +
  labs(
    title    = "Distribution of Movies by Rating Volume (log scale)",
    subtitle = "Bars = movie counts; line = average rating per bin"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title         = element_text(face = "bold"),
    axis.title.y.right = element_text(margin = margin(l = 10))
  )

#The majority of movies receive a moderate number of ratings, while only a few movies have very high or very low numbers of ratings. The average rating for most movies is between around 3.3 and 3.7, but the most popular movies tend to have even higher ratings.


##############################################################
## 2.2.3 How many ratings does each user give?
##############################################################

# 1. Count ratings per user
user_stats <- edx %>%
  count(userId, name="n_ratings") %>%
  filter(n_ratings > 1)

# 2. Define log-spaced “main” breaks up to 2000
main_breaks <- 10^seq(
  floor(log10(min(user_stats$n_ratings))),
  log10(2000),
  length.out = 20
)

# 3. Compute an overflow break so each bin has equal log-width
ratio          <- main_breaks[2] / main_breaks[1]
max_n          <- max(user_stats$n_ratings)
n_steps        <- ceiling(log(max_n / tail(main_breaks,1)) / log(ratio))
overflow_break <- tail(main_breaks,1) * ratio^n_steps

# 4. Final breaks vector & labels
breaks <- c(main_breaks, overflow_break)
labels <- c(
  label_number(accuracy = 1, big.mark = ",")(main_breaks),
  "over 2000"
)
x_limits <- range(breaks)

# 5. Assign integer bins 1:(length(breaks)-1)
user_stats <- user_stats %>%
  mutate(bin = cut(
    n_ratings,
    breaks         = breaks,
    include.lowest = TRUE,
    labels         = FALSE
  ))

# 6a. Top summary: count per bin + numeric edges + midpoint
top_summary <- user_stats %>%
  count(bin, name="n_users") %>%
  mutate(
    lower = breaks[bin],
    upper = breaks[bin+1],
    mid   = sqrt(lower * upper)
  )

# 6b. Bottom summary: average of all ratings per bin + midpoint
bot_summary <- edx %>%
  inner_join(user_stats, by="userId") %>%
  group_by(bin) %>%
  summarise(
    avg_rating = mean(rating),
    .groups    = "drop"
  ) %>%
  inner_join(top_summary %>% select(bin, mid), by="bin")

# 7. Global average rating
global_avg <- mean(edx$rating)

# 8a. Top panel: fixed histogram with locked x-axis
p1 <- ggplot(top_summary) +
  geom_rect(aes(
    xmin = lower, xmax = upper,
    ymin = 0,     ymax = n_users
  ), fill="plum3", color="white") +
  scale_x_log10(
    name    = "Number of Ratings per User (log10)",
    limits  = x_limits,
    expand  = c(0, 0),
    breaks  = breaks,
    labels  = labels
  ) +
  scale_y_continuous(
    name   = "Number of Users",
    breaks = seq(0, max(top_summary$n_users), by = 500),
    labels = comma_format()
  ) +
  labs(
    title    = "User Activity by Rating Volume",
    subtitle = "Counts of users in each log-spaced bin (last bin = over 2000)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 8b. Bottom panel: average-rating trend with the same x-axis
p2 <- ggplot(bot_summary, aes(x = mid, y = avg_rating)) +
  geom_line(color="darkred", size=1) +
  geom_point(color="darkred", size=2) +
  geom_text(aes(label = round(avg_rating,1)),
            vjust = -1, color = "darkred", size = 3) +
  geom_hline(yintercept = global_avg,
             linetype = "dashed", color = "black") +
  scale_x_log10(
    name    = "Number of Ratings per User (log10)",
    limits  = x_limits,
    expand  = c(0, 0),
    breaks  = breaks,
    labels  = labels
  ) +
  scale_y_continuous(
    name   = "Average Rating",
    limits = c(3, 4.),
    breaks = seq(3, 4., 0.1)
  ) +
  labs(subtitle = "Average of all ratings given by users in each bin") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 9. Stack panels at 3:2 ratio and draw
combo <- p1 / p2 + plot_layout(heights = c(3, 2))

# Print the combined plot
combo

#It is notable that users who provide less than approximately 30 ratings and those who provide more than 80 ratings tend to give lower ratings than users who provide between 30 and 80 ratings.


##############################################################
# MovieLens How many ratings does each user give-the-percentage
##############################################################

# Step 1: Create of the user_stats
user_stats <- edx %>%
  count(userId, name="n_ratings") %>%
  filter(n_ratings > 1)

# Step 2: Then computate and print percentages
total_users    <- nrow(user_stats)
pct_less100    <- round(mean(user_stats$n_ratings <  100) * 100, 1)
pct_atleast100 <- round(mean(user_stats$n_ratings >= 100) * 100, 1)

cat("Users with <100 ratings:   ", pct_less100, "%\n")
cat("Users with ≥100 ratings:   ", pct_atleast100, "%\n")

#The data shows that approximately 2/3 of users give fewer than 100 ratings. A small group of users contribute hundreds or thousands of ratings. This distribution suggests that the model needs to take user activity levels into account.

##############################################################
## 2.2.4 Which genres are most common?
##############################################################

library(dplyr)
library(stringr)
library(ggplot2)
library(forcats)
library(scales)
library(knitr)

# 1. Compute top 15 genres by volume + median
genre_summary <- edx %>%
  mutate(main_genre = str_extract(genres, "^[^|]+")) %>%      # first genre
  group_by(main_genre) %>%
  summarise(
    n_ratings    = n(),
    avg_rating = mean(rating),
    .groups       = "drop"
  ) %>%
  slice_max(n_ratings, n = 15) %>%                            # top 15
  mutate(
    main_genre    = fct_reorder(main_genre, n_ratings, .desc = TRUE),
    volume_10k    = n_ratings / 10000
  )

# 2. Plot
max10k <- ceiling(max(genre_summary$volume_10k) / 5) * 5

ggplot(genre_summary, aes(x = volume_10k, y = main_genre)) +
  geom_col(fill = "plum3") +
  geom_text(
    aes(label = sprintf("%.2f", avg_rating)),
    hjust = -0.1, color = "black", size = 3
  ) +
  scale_x_continuous(
    name   = "Number of Ratings (×10 000)",
    breaks = seq(0, max10k, by = 50),
    labels = label_number(accuracy = 1),
    expand = expansion(mult = c(0, 0.15))
  ) +
  labs(
    title = "Top 15 Movie Genres by Rating Volume",
    y     = "Genre",
    caption = "Numbers are in units of 10 000; grade = average rating"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.y = element_text(face = "bold")
  )

#Action, Comedy and Drama appear to be the most frequently rated genres, while genres such as Musical or Film-Noir are considerably less common.


##############################################################
## 2.2.5 Do the number of ratings change over time?
##############################################################

# 1. Summarize once, store into df_year
df_year <- edx %>%
  group_by(rating_year) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(rating_year)

# 2. Plot using df_year
ggplot(df_year, aes(x = rating_year, y = count)) +
  geom_col(fill = "plum3") +
  scale_x_continuous(
    breaks = df_year$rating_year,
    labels = df_year$rating_year
  ) +
  scale_y_continuous(
    name   = "Number of Ratings",
    labels = label_number(scale = 1/1000, suffix = "k")
  ) +
  labs(
    title = "Number of Ratings by Year",
    x     = "Year"
  ) +
  theme_minimal()

#The annual number of ratings fluctuates rather than following a linear trend, and does not display a normal distribution. A significant increase is observed around the year 2000, followed by a decline in 2001–2002. The volume of ratings then rises again towards 2005, before stabilizing between 2006 and 2008. This saw-tooth pattern indicates that user activity occurred in waves rather than exhibiting steady growth. Except for the years 1995 and 2009, the number of ratings recorded each year appears sufficient to represent the general user base.

##############################################################
## 2.2.6 When were the movies released?
##############################################################

library(dplyr)
library(ggplot2)
library(stringr)

edx %>%
  # 1) pull out the 4-digit year from the title into a new column
  mutate(
    release_year = str_extract(title, "(?<=\\()\\d{4}(?=\\))") %>% 
      as.integer()
  ) %>%
  # 2) reduce to one row per movie
  distinct(movieId, release_year) %>%
  # 3) drop any missing/extract failures (just in case)
  filter(!is.na(release_year)) %>%
  # 4) build the histogram
  ggplot(aes(x = release_year)) +
  geom_histogram(binwidth = 1, fill = "plum3", color = "white") +
  labs(
    title = "Movie Release Year Distribution",
    x     = "Release Year",
    y     = "Number of Movies"
  ) +
  theme_minimal()

#Most movies in the dataset were released between 1980 and 2005, with a small number of older classics and very few titles released after 2010. This distribution may impact predictions if user rating behavior differs between older and newer films.

##############################################################
## 2.2.7 Do some genres get better ratings?
##############################################################

edx %>%
  # 1. Extract the first genre into main_genre
  mutate(
    main_genre = str_extract(genres, "^[^|]+")
  ) %>%
  # 2. Drop any movies where extraction failed
  filter(!is.na(main_genre)) %>%
  # 3. Reorder genres by their median rating
  mutate(
    main_genre = fct_reorder(main_genre, rating, .fun = median)
  ) %>%
  # 4. Plot
  ggplot(aes(x = main_genre, y = rating)) +
  geom_boxplot(fill = "plum3") +
  coord_flip() +
  labs(
    title = "Median Ratings by Main Genre",
    x     = "Main Genre",
    y     = "Median Rating"
  ) +
  theme_minimal()

#Certain genres, such as Documentary or War, tend to receive higher average ratings, while genres like Horror exhibit greater variability. These patterns indicate that genre may influence user rating behavior.

##############################################################
# MovieLens Do some genres get better ratings-the-graph-with-the average-rating-by-genre
##############################################################

# Calculate the average rating by genre
avg_rating_genre <- edx %>%
  filter(!is.na(main_genre)) %>%
  group_by(main_genre) %>%
  summarise(
    avg_rating = mean(rating, na.rm = TRUE),
    count = n()
  ) %>%
  arrange(desc(avg_rating))

# Plot: Average Rating by Genre (bar chart)
ggplot(avg_rating_genre, aes(x = reorder(main_genre, avg_rating), y = avg_rating, fill = avg_rating)) +
  geom_col(show.legend = FALSE, width = 0.7, fill = "plum3") +
  geom_text(aes(label = round(avg_rating, 2)), 
            hjust = -0.1, size = 4, color = "black") + # Adds value to end of bar
  coord_flip() +
  labs(
    title = "Average Rating by Genre",
    x = "Genre",
    y = "Average Rating"
  ) +
  scale_y_continuous(limits = c(0, max(avg_rating_genre$avg_rating) + 0.5), breaks = seq(1, 5, 0.5)) +
  theme_minimal(base_size = 14)

#The chart reveals substantial differences in average ratings across movie genres. Film-Noir, Crime, and Documentary genres receive the highest average ratings, with Film-Noir notably standing out at 4.15. In contrast, IMAX and Horror genres register the lowest average ratings, with IMAX rated lowest at 2.32.

##############################################################
## 2.2.8 How often do users rate movies?
##############################################################

user_activity %>%
  ggplot(aes(x = user_frequency)) +
  geom_histogram(bins = 100, fill = "plum3", color = "white") +
  labs(title = "User Rating Frequency",
       x = "Average Ratings per Year",
       y = "Number of Users") +
  theme_minimal()

#The majority of users rate fewer than 20 movies per year on average, while a small number of highly active users provide significantly more ratings. This imbalance may influence the model, as more active users have a disproportionate impact on the dataset.

##############################################################
# MovieLens How often do users rate movies-graph-with-log-x-axis
##############################################################

user_activity %>%
  ggplot(aes(x = user_frequency)) +
  geom_histogram(bins = 100, fill = "plum3", color = "white") +
  scale_x_log10(
    name = "Average Ratings per Year (log scale)",
    breaks = c(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
    labels = scales::comma_format()
  ) +
  labs(
    title = "User Rating Frequency",
    y = "Number of Users"
  ) +
  theme_minimal()

#The histogram provides a clearer view of the distribution of user rating frequency when displayed on a logarithmic scale. Most users submit between 20 and 200 ratings per year, while very few users rate fewer than 10 or more than 1,000 movies annually. The use of a log scale highlights that a small proportion of highly active users account for a large number of ratings, whereas the majority of users are less active.

##############################################################
## 2.2.9 How many ratings were given for each genre over time?
##############################################################

# 1. Calculate number of ratings per year and genre
genre_year_volume <- edx %>%
  filter(!is.na(release_year), !is.na(main_genre)) %>%
  group_by(release_year, main_genre) %>%
  summarise(count = n(), .groups = "drop")

# 2. Calculate total volume by genre
genre_totals <- genre_year_volume %>%
  group_by(main_genre) %>%
  summarise(total_count = sum(count)) %>%
  arrange(total_count)

# 3. Factor main_genre by total_count (lowest on top, highest at bottom)
genre_year_volume <- genre_year_volume %>%
  mutate(main_genre = factor(main_genre, levels = genre_totals$main_genre))

# 4. Plot heatmap
ggplot(genre_year_volume, aes(x = release_year, y = main_genre, fill = count)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue", na.value = "grey90") +
  labs(
    title = "Number of Ratings by Genre and Release Year",
    x = "Release Year",
    y = "Main Genre (sorted by rating volume, lowest on top)",
    fill = "Rating Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

#The heatmap displays the number of ratings for each movie genre by release year, with darker blue indicating a higher volume of ratings.

#Genres such as Drama and Comedy received a significant number of ratings during the 1990s and early 2000s. In contrast, genres like Western or Film-Noir have fewer ratings overall, primarily concentrated among older films.

##############################################################
## 2.2.10 Are there trends in genre preferences over time?
##############################################################

# Calculate the average grade by year and gender
genre_year_matrix <- edx %>%
  filter(!is.na(release_year), !is.na(main_genre)) %>%
  group_by(release_year, main_genre) %>%
  summarise(mean_rating = mean(rating), .groups = "drop")

# Create a heatmap: years on the x-axis, genders on the y-axis, color = average score

ggplot(genre_year_matrix, aes(x = release_year, y = main_genre, fill = mean_rating)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "darkred", high = "green", na.value = "grey90") +
  labs(title = "Average Movie Ratings by Year and Genre",
       x = "Release Year",
       y = "Main Genre",
       fill = "Avg Rating") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

#The graph illustrates how the average movie rating varies by genre and year. Genres such as Drama, Comedy, Adventure, and Action consistently feature a large number of movies each year, whereas other genres are less frequently represented. Most average ratings fall between 3 and 4, with limited variation observed over time. Certain genres, including Western and War, are more prevalent among older films and have become less common in recent years.

##############################################################
# 2.2.11 Is there a big disparity in average rating over time and in the top 4 genres ?
##############################################################

# Calculate average rating per release year
avg_year_rating <- edx %>%
  group_by(release_year) %>%
  summarise(avg_rating = mean(rating), .groups = "drop")

# Identify top 4 genres by number of ratings
top_genres <- edx %>%
  count(main_genre, sort = TRUE) %>%
  top_n(4, n) %>%
  pull(main_genre)

# Calculate average rating per year and genre
avg_year_genre <- edx %>%
  group_by(release_year, main_genre) %>%
  summarise(avg_rating = mean(rating), .groups = "drop") %>%
  filter(main_genre %in% top_genres)

# Create the plot
ggplot() +
  # 1. Points colored for the top movie genre (part 1)
  geom_point(
    data = avg_year_genre,
    aes(x = release_year, y = avg_rating, color = main_genre),
    size = 2, alpha = 0.6
  ) +
  # 2. Global trends smoothed (part 2)
  geom_smooth(
    data = avg_year_rating,
    aes(x = release_year, y = avg_rating),
    method = "loess", se = TRUE, color = "black", linewidth = 1.2
  ) +
  # 3. Points for the average yearly rating in black triangle
  geom_point(
    data = avg_year_rating,
    aes(x = release_year, y = avg_rating),
    color = "black", size = 2, shape = 17
  ) +
  labs(
    title = "Average Movie Rating by Year and Genre",
    x = "Year of Movie's Release",
    y = "Average Yearly Rating",
    color = "Top 4 Genre"
  ) +
  theme_minimal()

#The graph presents the average movie rating by year and genre. Ratings tend to be higher for films released between 1930 and 1960, while a gradual decline is observed for most genres after 1970.

#The overall trend, indicated by the black line, confirms this pattern. These changes may reflect shifts in user rating behavior or transformations in movie production over time.

##############################################################
# 2.2.12 How average movie ratings changed over time, by genre?
##############################################################


library(plotly)
library(dplyr)
library(RColorBrewer)
library(webshot2)
library(webshot)
library(htmlwidgets)


# Prepare data 
genre_year_summary <- edx %>%
  filter(!is.na(main_genre), !is.na(release_year)) %>%
  group_by(release_year, main_genre) %>%
  summarise(
    mean_rating = round(mean(rating), 2),
    count = n(),
    .groups = "drop"
  )

# Build a color palette with enough unique colors
n_genres <- length(unique(genre_year_summary$main_genre))
my_colors <- colorRampPalette(brewer.pal(8, "Paired"))(n_genres) # or try "Paired", "Dark2"

# Create the plotly chart
p <- plot_ly(
  data = genre_year_summary,
  x = ~release_year,
  y = ~mean_rating,
  type = "scatter",
  mode = "markers",
  color = ~main_genre,
  colors = my_colors,
  size = ~count,
  sizes = c(5, 50),
  marker = list(
    sizemode = "area",
    opacity = 0.7
  ),
  text = ~paste(
    "Genre:", main_genre,
    "<br>Year:", release_year,
    "<br>Average Rating:", mean_rating,
    "<br>Number of Ratings:", count
  ),
  hoverinfo = "text"
)

# Save the plotly as a page HTML
htmlwidgets::saveWidget(p, "plotly_temp.html")

# Take a webshot of plotly in PNG
webshot::webshot("plotly_temp.html", "plotly_plot.png", vwidth = 1200, vheight = 800)

# Integrate it into the pdf report
knitr::include_graphics("plotly_plot.png")

#This chart provides a combined view of both quality (average rating) and popularity (number of ratings) for each genre. For instance, certain genres may receive high ratings but have low rating volume, while others are widely rated yet receive lower average scores. Most movie genres maintain average ratings between 3 and 4 across the years.



##############################################################
# 3. Modeling: Baseline and Effects Models
##############################################################

## 3.1 RMSE Function
RMSE <- function(predicted_ratings, true_ratings) {
  sqrt(mean((predicted_ratings - true_ratings)^2))
}

## 3.2 Baseline Model: Global Average
mu <- mean(edx$rating)
global_avg_pred <- rep(mu, nrow(final_holdout_test))
rmse_global_avg <- RMSE(global_avg_pred, final_holdout_test$rating)
cat("RMSE for Global Average Model:", round(rmse_global_avg, 4))

## 3.3 Movie Effect Model
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))
# Predict on holdout
movie_pred <- final_holdout_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i)
rmse_movie_effect <- RMSE(movie_pred$pred, final_holdout_test$rating)
cat("RMSE for the Movie Effect Model:", round(rmse_movie_effect, 4))

## 3.4 Movie + User Effect Model
user_avgs <- edx %>%
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))
user_movie_pred <- final_holdout_test %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u)
rmse_user_movie_effect <- RMSE(user_movie_pred$pred, final_holdout_test$rating)
cat("RMSE for the Movie and User Effect Model:", round(rmse_user_movie_effect, 4))

## 3.5 Regularized Movie + User Effect Model

# Try a sequence of lambda values for regularization
lambdas <- seq(2, 10, 0.5)
rmse_results <- sapply(lambdas, function(lambda) {
  # Regularized movie effect
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu) / (n() + lambda), .groups = "drop")
  # Regularized user effect
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - mu - b_i) / (n() + lambda), .groups = "drop")
  # Predict
  pred <- final_holdout_test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # Calculate RMSE
  RMSE(pred, final_holdout_test$rating)
})
# Best lambda
best_lambda <- lambdas[which.min(rmse_results)]
best_rmse <- min(rmse_results)
cat("Best Lambda :", round(best_lambda, 4), "     ")
cat("Minimum RMSE :", round(best_rmse, 4))

# Plot the regularized model
cv_results <- data.frame(lambda = lambdas, RMSE = rmse_results)

ggplot(cv_results, aes(x = lambda, y = RMSE)) +
  geom_line(color = "darkblue", linewidth = 1) +
  geom_point(color = "red", size = 2)+
  geom_point(data = subset(cv_results, RMSE == min(RMSE)),
             aes(x = lambda, y = RMSE), color = "darkgreen", size = 4) +
  labs(
    title = "Cross-Validation: Regularization Parameter (lambda)",
    x = expression(lambda),
    y = "RMSE"
  ) +
  theme_minimal()

## 3.6. Summary of the results in the modelling phase

library(knitr)
library(kableExtra)

# Results table
results <- data.frame(
  Model = c(
    "Global Average Model",
    "Movie Effect Model",
    "Movie + User Effect Model",
    "Regularized Movie + User Effect Model"
  ),
  Description = c(
    "Predicts all ratings as the overall average",
    "Adjusts for movies rated higher or lower than average",
    "Accounts for user's rating style (strict vs. generous)",
    "Adds regularization to avoid overfitting (lambda = 5)"
  ),
  RMSE = c(1.0612, 0.9439, 0.8653, 0.8648)
)

# Formated table (HTML)
kable(results, "html", caption = "Summary of RMSE for Different Models") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  column_spec(1, bold = TRUE) %>%
  row_spec(0, background = "#f7cac9", color = "black") 


##############################################################
# 4. Final Model and Results
##############################################################

#Let's first define "lambda" and mu for "final_holdout_test" dataset

best_lambda <- 5
mu <- mean(edx$rating)

# Calculate regularized movie effect
movie_effect <- edx %>%
  group_by(movieId) %>%
  summarise(
    b_i = sum(rating - mu) / (n() + best_lambda),
    .groups = "drop"
  )

# Calculate regularized user effect
user_effect <- edx %>%
  left_join(movie_effect, by = "movieId") %>%
  group_by(userId) %>%
  summarise(
    b_u = sum(rating - mu - b_i) / (n() + best_lambda),
    .groups = "drop"
  )

# Predict ratings
final_predictions <- final_holdout_test %>%
  left_join(movie_effect, by = "movieId") %>%
  left_join(user_effect, by = "userId") %>%
  mutate(
    pred = mu + ifelse(is.na(b_i), 0, b_i) + ifelse(is.na(b_u), 0, b_u)
  )

# Calculate the RMSE 
RMSE <- function(pred, actual) {
  sqrt(mean((pred - actual)^2))
}
rmse_final <- RMSE(final_predictions$pred, final_predictions$rating)

#Check the values:
cat("nrow(final_holdout_test):", nrow(final_holdout_test), "\n")
cat("mu:", mu, "\n")
cat("Sum b_i:", sum(movie_effect$b_i, na.rm=TRUE), "\n")
cat("Sum b_u:", sum(user_effect$b_u, na.rm=TRUE), "\n")
cat("Head predictions:", paste(head(final_predictions$pred), collapse=", "), "\n")
cat("Head ratings:", paste(head(final_predictions$rating), collapse=", "), "\n")
cat(sprintf("Final RMSE on holdout test set: %.5f\n", rmse_final))
