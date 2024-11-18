import os
import shutil
import py7zr
import argparse

videos_name = ['Beauty_1920x1080_120fps_420_8bit_YUV.yuv', 
              'HoneyBee_1920x1080_120fps_420_8bit_YUV.yuv', 
              'ReadySteadyGo_1920x1080_120fps_420_8bit_YUV.yuv',  
              'YachtRide_1920x1080_120fps_420_8bit_YUV.yuv', 
              'Bosphorus_1920x1080_120fps_420_8bit_YUV.yuv',  
              'Jockey_1920x1080_120fps_420_8bit_YUV.yuv', 
              'ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv']

videos_crop_name = ['Beauty_1920x1024_120fps_420_8bit_YUV.yuv', 
              'HoneyBee_1920x1024_120fps_420_8bit_YUV.yuv', 
              'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV.yuv',  
              'YachtRide_1920x1024_120fps_420_8bit_YUV.yuv', 
              'Bosphorus_1920x1024_120fps_420_8bit_YUV.yuv',  
              'Jockey_1920x1024_120fps_420_8bit_YUV.yuv', 
              'ShakeNDry_1920x1024_120fps_420_8bit_YUV.yuv']

iframe_settings = ['H265L20', 'H265L23', 'H265L26', 'H265L29']
short = ['Beauty', 'HoneyBee', 'ReadySteadyGo', 'YachtRide', 'Bosphorus', 'Jockey', 'ShakeNDry']


def un7z():
    seven_zip_files = []

    for root, _, files in os.walk('./'):
        for file in files:
            if file.endswith(".7z"):
                full_path = os.path.join(root, file)
                seven_zip_files.append(full_path)

    print(seven_zip_files)
    for file in seven_zip_files:
        with py7zr.SevenZipFile(file=file, mode='r') as archive:
            archive.extractall(path='videos')

def crop():
    videos_path = './videos'
    videos_crop_path = './videos_crop'

    for i in range(len(videos_crop_name)):
        os.system(f'ffmpeg -pix_fmt yuv420p  -s 1920x1080 -i {os.path.join(videos_path, videos_name[i])} -vf crop=1920:1024:0:0 {os.path.join(videos_crop_path, videos_crop_name[i])}')   

def lossless_frame():
    for i in range(len(videos_crop_name)):
        saveroot = f'images/{short[i]}'
        savepath = f'images/{short[i]}/im%03d.png'

        if not os.path.exists(saveroot):
            os.makedirs(saveroot)

        print(f'ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i videos_crop/{videos_crop_name[i]} {savepath}')
        os.system(f'ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i videos_crop/{videos_crop_name[i]} {savepath}')

def move():
    videos_crop_path = './videos_crop'
    for i in range(len(videos_crop_name)):        
        for iframe_setting in iframe_settings:
            source_dict = os.path.join(videos_crop_path, videos_crop_name[i].replace('.yuv', ''), iframe_setting)
            target_dict = os.path.join('./images', short[i])

            if not os.path.exists(source_dict):
                continue
            if not os.path.exists(target_dict):
                os.makedirs(target_dict)
            try:
                shutil.move(source_dict, target_dict)
                print(f'Success: {source_dict} to {target_dict}')
            except Exception as e:
                print(f"Error: {e}")    

def arg_parse():
    parser = argparse.ArgumentParser(description="preprocessing")
    
    # ------------------------------------------------------------------------------
    parser.add_argument('--un7z', action="store_true", help='un7z')
    parser.add_argument('--crop', action="store_true", help='crop: 1920x1080 > 1920x1024')
    parser.add_argument('--lossless_frame', action="store_true", help='Create I frame')
    parser.add_argument('--move', action="store_true", help='Move Folder')
    # ------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    if args.un7z:
        un7z()
    if args.crop:
        crop()
    if args.lossless_frame:
        lossless_frame()
    if args.move:
        move()
    
    
