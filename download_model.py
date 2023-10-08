from pathlib import Path
import requests


HUBERT_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'
RMVPE_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'
PRETRAINED_V1_DOWNLOAD_LINK = 'https://huggingface.co/Salmizu/Pretrained/resolve/main/'
PRETRAINED_V2_DOWNLOAD_LINK = 'https://huggingface.co/Salmizu/Pretrained_v2/resolve/main/'
SPLITTING_DOWNLOAD_LINK = 'https://huggingface.co/Salmizu/Songcover/resolve/main/'
SPLITTING2_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'

BASE_DIR = Path(__file__).resolve().parent.parent
hubert_models_dir = BASE_DIR / 'Songcover' /'assets' / 'hubert'
rmvpe_models_dir = BASE_DIR / 'Songcover' /'assets' / 'rmvpe'
pretrained_v1_models_dir = BASE_DIR / 'Songcover' / 'assets' / 'pretrained'
pretrained_v2_models_dir = BASE_DIR / 'Songcover' / 'assets' / 'pretrained_v2'
splitting_models_dir = BASE_DIR / 'Songcover' / 'assets' / 'uvr5_weights'
splitting2_models_dir = BASE_DIR / 'Songcover' / 'assets' / 'uvr5_weights'

def dl_model(link, model_name, dir_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == '__main__':
    hubert_models_names = ['hubert_base.pt',]
    for model in hubert_models_names:
        print(f'Downloading {model}...')
        dl_model(HUBERT_DOWNLOAD_LINK, model, hubert_models_dir)

    rmvpe_models_names = ['rmvpe.pt']
    for model in rmvpe_models_names:
        print(f'Downloading {model}...')
        dl_model(RMVPE_DOWNLOAD_LINK, model, rmvpe_models_dir)

    pretrained_v1_models_names = ['D32k.pth', 'D40k.pth', 'D48k.pth', 'G32k.pth', 'G40k.pth', 'G48k.pth', 'f0D32k.pth', 'f0D40k.pth', 'f0D48k.pth', 'f0G32k.pth', 'f0G40k.pth', 'f0G48k.pth']
    print(f'Downloading pretrained models for training...')
    for model in pretrained_v1_models_names:
       # print(f'Downloading {model}...')
        dl_model(PRETRAINED_V1_DOWNLOAD_LINK, model, pretrained_v1_models_dir)

    pretrained_v2_models_names = ['D32k.pth', 'D40k.pth', 'D48k.pth', 'G32k.pth', 'G40k.pth', 'G48k.pth', 'f0D32k.pth', 'f0D40k.pth', 'f0D48k.pth', 'f0G32k.pth', 'f0G40k.pth', 'f0G48k.pth']
    for model in pretrained_v2_models_names:
       # print(f'Downloading {model}...')
        dl_model(PRETRAINED_V2_DOWNLOAD_LINK, model, pretrained_v2_models_dir)

    splitting_models_names = ['HP2-人声vocals+非人声instrumentals.pth', 'HP5-主旋律人声vocals+其他instrumentals.pth', 'HP2_all_vocals.pth', 'HP3_all_vocals.pth', 'HP5_only_main_vocal.pth', 'VR-DeEchoAggressive.pth', 'VR-DeEchoDeReverb.pth', 'VR-DeEchoNormal.pth']
    print(f'Downloading pretrained model for splitting...')                    
    for model in splitting_models_names:
        dl_model(SPLITTING_DOWNLOAD_LINK, model, splitting_models_dir)

    splitting2_models_names = ['5_HP-Karaoke-UVR.pth']
    #print(f'Downloading pretrained model for splitting...')                    
    for model in splitting2_models_names:
        dl_model(SPLITTING2_DOWNLOAD_LINK, model, splitting2_models_dir)
                              
    print('All models downloaded!')
