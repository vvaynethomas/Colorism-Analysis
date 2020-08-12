library(rvest)
library(tidyverse)

scrape_image <- function(search){
  url <- paste0("https://www.google.co.in/search?q=", search, "&source=lnms&tbm=isch")
  webpage <- html_session(url)
  link.titles <- webpage %>%
    html_nodes("img")
  
  img.url <- link.titles[2] %>%
    html_attr("src")
  
  download.file(img.url, paste0("D:/Denver/Tools_DataScience/Project/images/", search, ".jpg"), mode = "wb")
}

df <- read.csv("D:/Denver/Tools_DataScience/Project/df.csv")

df %>%
  pull(Executive) %>%
  map(scrape_image)


