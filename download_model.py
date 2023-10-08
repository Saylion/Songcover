from pathlib import Path
import requests


HUBERT_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'
RMVPE_DOWNLOAD_LINK = 'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/'
PRETRAINED_V1_DOWNLOAD_LINK = 'https://huggingface.co/Salmizu/Pretrained/resolve/main/'
PRETRAINED_V2_DOWNLOAD_LINK = 'https://huggingface.co/Salmizu/Pretrained_v2/resolve/main/'

BASE_DIR = Path(__file__).resolve().parent.parent
hubert_models_dir = BASE_DIR / 'assets' / 'hubert'
rmvpe_models_dir = BASE_DIR / 'assets' / 'rmvpe'
pretrained_v1_models_dir = BASE_DIR / 'assets' / 'pretrained'
pretrained_v2_models_dir = BASE_DUR / 'assets' / 'pretrained_v2'

def dl_model(link, model_name, dir_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == '__main__':
    pretrained_v2_model_names = ['D32k.pth', 'D40k.pth', 'D48k.pth', 'G32k.pth', 'G40k.pth', 'G48k.pth', 'f0D32k.pth', 'f0D40k.pth', 'f0D48k.pth', 'f0G32k.pth', 'f0G40k.pth', 'f0G48k.pth']
    print(f'Downloading pretrained models...')
    for model in pretrained_v2_model_names:
       # print(f'Downloading {model}...')
        dl_model(PRETRAINED_V1_DOWNLOAD_LINK, model, pretrained_v2_models_dir)

    hubert_model_names = ['hubert_base.pt',]
    for model in hubert_model_names:
        print(f'Downloading {model}...')
        dl_model(HUBERT_DOWNLOAD_LINK, model, hubert_models_dir)

    rmvpe_models_names = ['rmvpe.pt']
    for model in rmvpe_model_names:
        print(f'Downloading {model}...')
        dl_model(RMVPE_DOWNLOAD_LINK, model, rmvpe_models_dir)

    pretrained_v1_model_names = ['D32k.pth', 'D40k.pth', 'D48k.pth', 'G32k.pth', 'G40k.pth', 'G48k.pth', 'f0D32k.pth', 'f0D40k.pth', 'f0D48k.pth', 'f0G32k.pth', 'f0G40k.pth', 'f0G48k.pth']
    print(f'Downloading pretrained models...')
    for model in pretrained_v1_model_names:
       # print(f'Downloading {model}...')
        dl_model(PRETRAINED_V1_DOWNLOAD_LINK, model, pretrained_v1_models_dir)
    
    print('All models downloaded!')
