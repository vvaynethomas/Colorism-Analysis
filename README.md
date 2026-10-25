# Colorism-Analysis

This repository is an exploratory image-processing project whose stated aim is to examine colorism using images of business executives. The notebooks and code do not define a testable hypothesis or measure discrimination, representation, or social outcomes. What they currently calculate is a set of grayscale pixel-intensity statistics for some detected face crops. Those statistics are not measurements of a person's identity or a validated measure of skin tone, and they cannot establish whether colorism occurred.

## Workflow and files

The checked-in material reflects a partially disconnected workflow:

1. `image_scraper.ipynb` reads the current Wikipedia page titled [List of chief executive officers](https://en.wikipedia.org/wiki/List_of_chief_executive_officers) into a table and can write it as `df.csv`. The notebook does not record a retrieval date or a frozen source copy.
2. `image_scraper.R` reads a separate CSV from a machine-specific `D:/Denver/.../df.csv` path, searches Bing Images for each listed executive, and downloads a JPEG into `Images/`. The script does not save the selected image's source URL, search date, or rights information. The checked-in `im_proc_functions/df.csv` is not at the path the script expects.
3. `im_proc_functions/__init__.py` enumerates `.jpg` files, detects faces with OpenCV's Haar cascade, crops the center 60% of each detected face box, and calculates grayscale statistics. Images smaller than 120 pixels in either dimension are rejected. If multiple faces are detected, the function chooses one at random. It returns missing values when detection fails or when the calculated mean or mode falls outside the code's 30–230 range.
4. `batch_colorism.ipynb` runs the image processing over `Images/`, drops missing results, displays distributions, and writes `ceo_df.csv`. `testing_no_skin.ipynb` is an exploratory single-image test of the same detector and statistics. `image_scraper.ipynb` also contains a plotting workflow over the image files.

The statistics are mean, minimum, maximum, median, and mode of nonzero grayscale values in the crop. The face detector uses a grayscale, histogram-equalized copy of the image; the reported statistics are calculated separately from the original crop after BGR-to-grayscale conversion. The crop is not a skin-only segmentation: it can include hair, background, and other non-skin pixels. No color calibration, lighting normalization, or validated skin-tone annotation is implemented.

## Data provenance and reproduction

Portraits and derived face-level data are intentionally excluded from the current repository tree because image sources, reuse permissions, and privacy expectations were not recorded. This includes the scraped `Images/` collection, the three root-level reference portraits, `ceo_df.csv` (which contains names and serialized face pixels), and saved notebook outputs that displayed portraits or face crops. The notebooks' code remains as a record of the prior exploratory workflow; it requires separately sourced, appropriately licensed local inputs to run. Generated image folders and derived exports are ignored to reduce accidental re-addition.

Previously saved notebook outputs reported different retained sample counts across runs. Those outputs and the face-level CSV have been removed from the current tree, were not independently reproduced, and should not be treated as validated results. The function's random choice when an image contains multiple detected faces is one source of run-to-run variation.

An end-to-end clean rerun has not been established. The R script depends on an external, machine-specific CSV path, and the repository has no dependency or environment lock file. The image acquisition URLs, dates, and source files used for prior outputs are not recorded, so the original inputs cannot be reconstructed from the repository alone. The project also declares a `SkinDetector` Git submodule in `.gitmodules`; its role in the notebook workflow is not documented.

## Interpretation, limitations, and responsible use

- Search-result images can vary in lighting, exposure, white balance, camera, image processing, resolution, compression, pose, and background. These factors can change pixel intensity independently of a person's skin.
- Face detection and center cropping are imperfect; they do not guarantee that the crop contains only a face or skin. No manual quality review, detector validation, measurement-error analysis, or justification for the hard-coded intensity thresholds is recorded.
- The source roster and image-selection process do not define a sampling frame, inclusion criteria, comparison group, or representative population. The available files should not be treated as a complete or unbiased sample of executives.
- A grayscale intensity statistic is not a person's race, ethnicity, or identity. The project does not infer those sensitive traits, and it does not implement text classification or skin-tone classification.
- Colorism concerns social treatment and outcomes. Image pixel values alone do not measure colorism or demonstrate discrimination, causation, or any group's characteristics.
- Image permissions, licensing, privacy expectations, and suitability for redistribution have not been resolved. The current tree excludes the portraits and face-level export rather than asserting that they were cleared for reuse. No new people have been scraped for this documentation.

## Lessons and proposal for future work

The current workflow is useful as a record of an exploratory coding exercise, but its data lineage, sampling decisions, and measurements are not sufficient for demographic or social conclusions. A future study should first define a narrow, ethical, testable research question and a justified sampling frame; document permitted image sources, acquisition dates, and reuse terms; and assess whether appropriately calibrated, validated measurements are possible under controlled or comparable imaging conditions. Any analysis of colorism would also need relevant social outcomes and a defensible design, not just image intensity. These are proposals for separately reviewed future work, not results or validation claimed by this repository.

**History note:** this change excludes the assets from the current tree, but does not rewrite earlier Git commits. Earlier repository history, existing clones, forks, or cached copies may still contain the files. Removal from history would require a separate, coordinated repository-owner decision.
