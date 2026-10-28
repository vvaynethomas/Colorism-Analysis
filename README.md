# Colorism-Analysis

An exploratory Python and R project inspired by a question about colorism among business executives: how might portrait analysis contribute to exploring apparent skin tone?

The project follows portraits from collection to numerical summaries, combining web scraping, OpenCV face detection, center cropping, and distribution plots. Its hands-on focus is **turning images into descriptive pixel statistics** and exploring how those summaries vary across images.

**Start with [the batch analysis notebook](batch_colorism.ipynb)** for the main workflow, then explore [the image-processing functions](im_proc_functions/__init__.py) for the implementation.

## Explore the repository

| File | What to look for |
| --- | --- |
| [batch_colorism.ipynb](batch_colorism.ipynb) | Batch summaries and distribution plots of mean, median, and mode. |
| [im_proc_functions/__init__.py](im_proc_functions/__init__.py) | JPEG discovery, face detection, cropping, and statistics. |
| [testing_no_skin.ipynb](testing_no_skin.ipynb) | Single-image exploration of the detector and crop. |
| [image_scraper.ipynb](image_scraper.ipynb) | Wikipedia roster collection and exploratory plotting. |
| [image_scraper.R](image_scraper.R) | Portrait acquisition using Bing image search. |

## Analysis workflow

1. **Collect a roster.** The Python scraper reads the [Wikipedia list of chief executive officers](https://en.wikipedia.org/wiki/List_of_chief_executive_officers) into a table and can export `df.csv`. The separate R script searches for portraits and downloads JPEGs.
2. **Detect and crop.** OpenCV's frontal-face Haar cascade locates faces using a histogram-equalized grayscale copy. The module keeps the central 60% of a selected face box in each dimension, choosing randomly when multiple faces are detected.
3. **Summarize pixels.** The original crop is converted to grayscale, and its nonzero values are summarized by mean, minimum, maximum, median, and mode.
4. **Explore distributions.** The batch notebook filters missing results, displays descriptive summaries and distribution plots, and can export the results as `ceo_df.csv`.

The core stack is **pandas, NumPy, OpenCV, SciPy, Matplotlib, and seaborn**, with **rvest and tidyverse** for R acquisition.

## Exploring locally

Browse the notebooks on GitHub, or use Jupyter with the Python libraries above to experiment locally. Collection and exploratory cells also use requests, Beautiful Soup, scikit-image, and scikit-learn. Allow for compatibility adjustments to older library calls; dependency versions are unpinned.

Supply your own appropriately permitted portrait inputs in `Images/` and adapt the R script's absolute input/output paths if using acquisition. Portraits and the derived export are not bundled; later demonstration cells also reference local portraits. Review cells before running them: acquisition accesses external sites, and exports include names and face-image arrays, so handle those outputs with appropriate privacy safeguards.

## Interpreting results

The summaries describe grayscale intensity in the selected crops, which varies with lighting, image processing, and non-skin pixels. They are useful for exploring the image-processing workflow, rather than as validated skin-tone measurements or evidence of discrimination.
