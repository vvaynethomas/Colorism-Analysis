# Download one portrait per search query from Bing Images.
#
# Usage, from the repository root:
#   source("scripts/fetch_portraits.R")
#   fetch_portrait("Satya Nadella", dest_dir = "data/images")
#   roster <- readr::read_csv("data/executives.csv")
#   purrr::walk(utils::URLdecode(roster$Executive), fetch_portrait)
#
# Each query is saved as <dest_dir>/<query>.jpg, where spaces are encoded as %20, which
# is the naming colorism.batch_process() decodes back into a name. Check the downloaded
# files before analysis: the first .jpg result is not always the person searched for.
library(rvest)
library(tidyverse)

# All href targets on a page.
scrape_links <- function(url) {
  tibble(url = read_html(url) %>% html_nodes("a") %>% html_attr("href"))
}

# Save the `num`-th .jpg link on the Bing Images results page for `search`.
scrape_image <- function(search, num, dest_dir = "data/images") {
  query <- utils::URLencode(search)
  page <- paste0("https://www.bing.com/images/search?q=", query,
                 "&form=HDRSC3&first=1&scenario=ImageBasicHover")
  image_url <- scrape_links(page) %>%
    filter(str_detect(url, "http"), str_detect(url, ".jpg")) %>%
    slice(num) %>%
    pull(url)
  dir.create(dest_dir, showWarnings = FALSE, recursive = TRUE)
  download.file(image_url, file.path(dest_dir, paste0(query, ".jpg")), mode = "wb")
}

# Try successive results (up to `max_tries`) until one downloads.
fetch_portrait <- function(search, dest_dir = "data/images", max_tries = 14) {
  for (i in seq_len(max_tries)) {
    ok <- tryCatch({ scrape_image(search, i, dest_dir); TRUE }, error = function(e) FALSE)
    if (ok) return(invisible(TRUE))
  }
  message("no downloadable image for ", search)
  invisible(FALSE)
}
