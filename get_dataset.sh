mkdir -p data
cd data

mkdir -p Vimeo
cd Vimeo

# echo "Here is some instructions" > vimeo.txt
wget -c http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
unzip vimeo_septuplet.zip
rm vimeo_septuplet.zip

cd ..
mkdir -p UVG
cd UVG

# echo "Here is some instructions" > uvg.txt

# 定義檔案列表
files=(
    Beauty_1920x1080_120fps_420_8bit_YUV
    HoneyBee_1920x1080_120fps_420_8bit_YUV
    ReadySetGo_1920x1080_120fps_420_8bit_YUV
    YachtRide_1920x1080_120fps_420_8bit_YUV
    Bosphorus_1920x1080_120fps_420_8bit_YUV
    Jockey_1920x1080_120fps_420_8bit_YUV
    ShakeNDry_1920x1080_120fps_420_8bit_YUV
)

# 下載所有文件
for file in "${files[@]}"; do
    wget -c "https://ultravideo.fi/video/${file}_RAW.7z"
done

wait

python3 pre_processing.py --un7z

wait

rm *.7z
