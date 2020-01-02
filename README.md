# Replication files for Chen, Glaeser, and Wessel (2020)

NBER Working Paper link: https://www.nber.org/papers/w26587

Thanks to Yanchen Jiang (github: @jeffjiang1204) for research assistance.

## Dependencies

- All code is written on Python 3.7 (Anaconda distribution) and R 3.5.1
- Non-standard python packages:
    - cenpy (https://cenpy-devs.github.io/cenpy/index.html)
    - pyjanitor (https://pyjanitor.readthedocs.io/)
    - geopandas (http://geopandas.org/)
    - rpy2 (https://rpy2.github.io/doc/v3.0.x/html/index.html)
    - Jupytext (https://github.com/mwouts/jupytext)
- R packages:
    - car (https://cran.r-project.org/web/packages/car/index.html)
    - plm (https://cran.r-project.org/web/packages/plm/plm.pdf)
    - drdid (https://pedrohcgs.github.io/DRDID/)
        - Doubly-robust difference-in-differences
    - did (https://bcallaway11.github.io/did/)
        - Callaway--Sant'Anna difference-in-differences

## Data
The only datasets that are not automatically retrieved are
- ZIP-tract crosswalk (https://www.huduser.gov/portal/datasets/usps_crosswalk.html)

- ZCTA-level population from ACS


These data are included in `data.zip`.

## Usage

All code can be run via
```bash
sh master.sh
```
at top of the directory, assuming that all dependencies are installed properly.

The file `settings.json` controls parameters the analysis depend on. The main text uses 

```json
{"START_YEAR" : 2014,
"LIC_ONLY" : true,
"OVERWRITE": true}
```

Toggling `LIC_ONLY` to `false` generates tables in the appendix.



Alternatively, we may run 

```bash
if [ -d "data/" ]
then
    echo "Data already unzipped."
else
    echo "Unzipping data."
    unzip data.zip
    rm -rf __MACOSX # Kill auxiliary file from unzipping
fi

for i in *py;do
  jupytext --to ipynb $i
done
```
to unzip the data and convert .py files to notebooks, and run the Jupyter notebooks manually.

Expected output:

```
Unzipping data.
Archive:  data.zip
   creating: data/
  inflating: data/ZIP_TRACT_032017.xlsx
   creating: __MACOSX/
   creating: __MACOSX/data/
  inflating: __MACOSX/data/._ZIP_TRACT_032017.xlsx
   creating: data/ACS_16_5YR_DP05/
  inflating: data/ACS_16_5YR_DP05/aff_download_readme_ann.txt
   creating: __MACOSX/data/ACS_16_5YR_DP05/
  inflating: __MACOSX/data/ACS_16_5YR_DP05/._aff_download_readme_ann.txt
  inflating: data/ACS_16_5YR_DP05/ACS_16_5YR_DP05.txt
  inflating: __MACOSX/data/ACS_16_5YR_DP05/._ACS_16_5YR_DP05.txt
  inflating: data/ACS_16_5YR_DP05/ACS_16_5YR_DP05_metadata.csv
  inflating: __MACOSX/data/ACS_16_5YR_DP05/._ACS_16_5YR_DP05_metadata.csv
  inflating: data/ACS_16_5YR_DP05/ACS_16_5YR_DP05_with_ann.csv
  inflating: __MACOSX/data/ACS_16_5YR_DP05/._ACS_16_5YR_DP05_with_ann.csv
  inflating: __MACOSX/data/._ACS_16_5YR_DP05
  inflating: __MACOSX/._data
[jupytext] Reading 01-zillow-design-and-spatial-match.py
[jupytext] Writing 01-zillow-design-and-spatial-match.ipynb
[jupytext] Sync timestamp of '01-zillow-design-and-spatial-match.py'
[jupytext] Reading 02-aggregating-to-zips.py
[jupytext] Writing 02-aggregating-to-zips.ipynb
[jupytext] Sync timestamp of '02-aggregating-to-zips.py'
[jupytext] Reading 03-supplemental-tables.py
[jupytext] Writing 03-supplemental-tables.ipynb
[jupytext] Sync timestamp of '03-supplemental-tables.py'
[NbConvertApp] Converting notebook 01-zillow-design-and-spatial-match.ipynb to notebook
[NbConvertApp] Executing notebook with kernel: python3
[NbConvertApp] Writing 167743 bytes to 01-zillow-design-and-spatial-match.nbconvert.ipynb
[NbConvertApp] Converting notebook 02-aggregating-to-zips.ipynb to notebook
[NbConvertApp] Executing notebook with kernel: python3
[NbConvertApp] WARNING | Timeout waiting for IOPub output
[NbConvertApp] Writing 37932 bytes to 02-aggregating-to-zips.nbconvert.ipynb
[NbConvertApp] Converting notebook 03-supplemental-tables.ipynb to notebook
[NbConvertApp] Executing notebook with kernel: python3
[NbConvertApp] Writing 47007 bytes to 03-supplemental-tables.nbconvert.ipynb
```

