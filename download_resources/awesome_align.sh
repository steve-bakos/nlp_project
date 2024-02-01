set -e

MODEL_DIR=$1

mkdir -p $MODEL_DIR

cd tools

if [ ! -d awesome-align ]; then
    git clone https://github.com/neulab/awesome-align.git
fi

cd ..

if [ ! -d $MODEL_DIR/model_without_co ]; then
    # Use gdown CLI tool to download the file
    gdown --id 1IcQx6t5qtv4bdcGjjVCwXnRkpr67eisJ -O $MODEL_DIR/awesome_without_co.zip --quiet
    
    # Check if the file is an HTML file (which means the download failed)
    if file --mime-type "$MODEL_DIR/awesome_without_co.zip" | grep -q html; then
        echo "The downloaded file appears to be an HTML page, not a zip file."
        echo "Please try downloading the file manually using this URL:"
        echo "https://drive.google.com/uc?export=download&id=1IcQx6t5qtv4bdcGjjVCwXnRkpr67eisJ"
        # Cleanup and exit
        rm "$MODEL_DIR/awesome_without_co.zip"
        exit 1
    fi
    
    # Attempt to unzip the downloaded file
    unzip -d $MODEL_DIR $MODEL_DIR/awesome_without_co.zip
fi
