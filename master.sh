if [ -d "data/" ]
then
    echo "Data already unzipped."
else
    echo "Unzipping data."
    unzip data.zip
    rm -rf __MACOSX
fi

for i in *py;do
  jupytext --to ipynb $i
done

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 01-zillow-design-and-spatial-match.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 02-aggregating-to-zips.ipynb
jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute 03-supplemental-tables.ipynb
