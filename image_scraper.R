library(rvest)
library(tidyverse)

df <- read_csv("D:/Denver/Tools_DataScience/Project/df.csv") %>%
  dplyr::select(-X1)

scraplinks <- function(url){
  url_ <- read_html(url) %>% 
    html_nodes("a") %>% 
    html_attr('href')
  tibble(url = url_)
}

scrape_image <- function(search, num){
  url <- paste0("https://www.bing.com/images/search?q=",
                search,
                "&form=HDRSC3&first=1&scenario=ImageBasicHover")
  
  
  url1 <- scraplinks(url) %>% 
    filter(str_detect(url, "http"),
           str_detect(url, ".jpg")) %>%
    slice(num) %>%
    pull(url)
  
  download.file(url1, paste0("D:/Denver/Tools_DataScience/Project/GitHub/Colorism-Analysis/Images/", search, ".jpg"), mode = "wb")
}

try_it <- function(search){
  print(search)
  i <- 1
  while(i < 15){
    skip <- FALSE
    tryCatch(scrape_image(search, i), error = function(e) skip <- TRUE)
    if(skip){
      i <- i + 1
    } else {
      break
    }
  }
}

df %>%
  pull(Executive) %>%
  map(try_it)
