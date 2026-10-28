# Colorism-Analysis

An exploratory Python and R project motivated by a question about colorism among business executives: could portrait analysis help investigate differences in apparent skin tone?

The implemented work explores a narrower technical question: **how can face detection, cropping, and grayscale statistics summarize portrait images?** It connects web-table collection, image acquisition, OpenCV processing, and distribution plots. It is a useful starting point for exploring that workflow and its measurement limits, not a completed study of colorism.

**Start with [the batch analysis notebook](batch_colorism.ipynb)** for the main workflow, then read [the image-processing functions](im_proc_functions/__init__.py) to see exactly what is measured. You can browse the source without downloading images or running the notebooks.

## Explore the repository

| File | What to look for |
| --- | --- |
| [batch_colorism.ipynb](batch_colorism.ipynb) | Batch processing, missing-value filtering, descriptive summaries, and distribution plots of mean, median, and mode. |
| [im_proc_functions/__init__.py](im_proc_functions/__init__.py) | JPEG discovery, face detection, center cropping, grayscale statistics, and assembly of the results table. |
| [testing_no_skin.ipynb](testing_no_skin.ipynb) | A single-image exploration of detection, cropping, and the same statistics. |
| [image_scraper.ipynb](image_scraper.ipynb) | Collection of an executive roster from Wikipedia, plus exploratory image processing and plotting. |
| [image_scraper.R](image_scraper.R) | An image-acquisition script using Bing search results and local file paths. |

## Analysis workflow

1. **Collect a roster.** The Python scraper reads the [Wikipedia list of chief executive officers](https://en.wikipedia.org/wiki/List_of_chief_executive_officers) into a table and can export `df.csv`. The separate R script searches for portraits and downloads JPEGs.
2. **Detect and crop.** The processing module reads `.jpg` files from a local directory, rejects images smaller than 120 pixels in either dimension, and uses OpenCV's frontal-face Haar cascade on a histogram-equalized grayscale copy. It selects a detected face at random when there are multiple detections and keeps the central 60% of the face box in each dimension.
3. **Summarize pixels.** The selected crop is converted from its original BGR values to grayscale. The code calculates mean, minimum, maximum, median, and mode over nonzero pixel values. Detection failures and crops whose mean or mode falls outside 30-230 yield missing statistics.
4. **Explore distributions.** The batch notebook drops missing rows, summarizes the retained values, plots distributions, and can export `ceo_df.csv`. The returned table also includes names and face-image arrays; it is not an anonymous aggregate.

The main tools are **pandas, NumPy, OpenCV, SciPy, Matplotlib, and seaborn**, with **rvest and tidyverse** for the R acquisition script. Additional exploratory cells use Beautiful Soup, requests, scikit-image, and scikit-learn.

## Reading and reproduction

**This is a source-first exploration, not a ready-to-run pipeline.** Portrait inputs and the derived face-level export are not included, and there is no pinned dependency environment or verified clean-run procedure.

For local experimentation, review the notebook cells before running them: acquisition cells make network requests, and export cells write files. Supply separately sourced images with appropriate permissions and privacy safeguards; do not assume public search results are cleared for reuse or redistribution.

The batch notebook expects an `Images/` directory, while later demonstration cells reference additional local portraits. The R script uses machine-specific absolute paths that must be adapted; the Python roster export does not automatically connect to those paths. Source-image URLs and acquisition dates are not recorded, so the original inputs cannot be reconstructed from this repository alone. Older library calls may also need adaptation for a current environment.

## What the measurements can and cannot tell us

The code demonstrates a route from image files to descriptive pixel statistics. It does **not** establish a validated skin-tone measure or a finding about colorism.

- A center crop is not skin segmentation: hair, background, and other non-skin pixels can affect the statistics. Lighting, exposure, pose, compression, and camera processing also change grayscale intensity. Histogram equalization is used for detection, not to calibrate the measured crop.
- The workflow has no validated sampling frame, comparison group, detector-quality assessment, or justification for its intensity thresholds. Random face selection can change retained images and summaries between runs.
- Pixel intensity does not identify race or ethnicity, and the project does not measure discrimination or social outcomes. Claims about colorism would require a defined research question, validated measurements, an appropriate study design, and relevant outcome data.

The central takeaway is methodological: producing a numeric image summary is different from establishing what that number means.
