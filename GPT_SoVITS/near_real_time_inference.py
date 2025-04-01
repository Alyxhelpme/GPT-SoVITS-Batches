import os
import sys
import random
import torch
import warnings
import numpy as np
import sounddevice as sd
sd.default.blocksize = 4096  # Adjust buffer size
sd.default.latency = 0.2
from characterai import aiocai
import numpy as np
from scipy.io.wavfile import write
import asyncio
import threading
import logging
logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger("TTS")

# sys.path.append('./GPT_SoVITS')
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# if module_path not in sys.path:
sys.path.append(module_path)

# Check if 'cnhubert_base_path' exists, otherwise set a default and issue a warning
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "./pretrained_models/chinese-hubert-base"
)

if "cnhubert_base_path" not in os.environ:
    warnings.warn(
        "Environment variable 'cnhubert_base_path' not found. Using default path: "
        + cnhubert_base_path
    )

# cnhubert_base_path = os.environ.setdefault(
#     "cnhubert_base_path", "./pretrained_models/chinese-hubert-base"
# )

# Check if 'bert_path' exists, otherwise set a default and issue a warning
bert_path = os.environ.get(
    "bert_path", "./pretrained_models/chinese-roberta-wwm-ext-large"
)

if "bert_path" not in os.environ:
    warnings.warn(
        "Environment variable 'bert_path' not found. Using default path: " + bert_path
    )
# bert_path = os.environ.setdefault(
#     "bert_path", "./GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
# )

# from text import symbols2 as symbols_v2
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import get_method
# from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


cut_method = {
    i18n("不切"):"cut0",
    i18n("4sentences"): "cut1",
    i18n("50characters"): "cut2",
    i18n("Chinese"): "cut3",
    i18n("English"): "cut4",
    i18n("Punctuations"): "cut5",
}



def play_audio():
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            audio_queue.task_done()
            break
        try:
            sampling_rate, audio_chunk = chunk
            # sd.sleep(200)
            sd.default.samplerate = sampling_rate
            # sd.default.channels = 1 if len(audio_chunk.shape) == 1 else audio_chunk.shape[1]
            sd.play(audio_chunk, sampling_rate)
            sd.wait()
            audio_queue.task_done()

        except Exception as e:
            logger.debug(f"Error during audio playback: {e}")
            audio_queue.task_done()
            continue
        


def shorten_if_longer(ref_audio_path: str):
    audio = AudioSegment.from_wav(ref_audio_path)

    if len(audio)>10000: #if its longer than 10 seconds
        logger.debug("Audio is longer than 10 seconds, we're trimming it!")
        audio = audio[:10000] #no matter what's inside of it, we cut it to 10 seconds

        audio.export(f"./GPT_SoVITS/audio_sources/{os.path.basename(ref_audio_path)}",format="wav")
        return f"./GPT_SoVITS/audio_sources/{os.path.basename(ref_audio_path)}"
    if len(audio)<3000: #Audios shorter than 3 secs will return error
        raise ValueError("Audio duration must be between 3 and 10 seconds.")
    else:
        logger.debug("File length is good")
        return ref_audio_path
    
#### Inference invocation ####


def fast_inference(tts_pipeline,text,top_k,top_p,temperature,text_split_method,split_bucket,fragment_interval,parallel_infer,repetition_penalty, batch_size=20, speed_factor=1, prompt_text="", ref_audio_path = None):

    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
    inputs={
        "text": text,
        "text_lang": "en",
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": [],
        "prompt_text": prompt_text,
        "prompt_lang": "en",
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":True,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
    }
    # for item, actual_seed in tts_pipeline.run(inputs):
    #     logger.debug("Generated Item:", item)
    #     logger.debug("Seed Used:", actual_seed)
    #     yield item, actual_seed


    for chunk in tts_pipeline.run(inputs):
        # sampling_rate, audio_chunk = chunk
        yield chunk
        # audio_queue.put(chunk)
        # threading.Thread(target=play_audio, args=(audio_chunk, sampling_rate)).start()
        # sd.play(audio_chunk, sampling_rate)
        # sd.wait()
    # audio_queue.put(None)

    # sd.stop()


def initialize_default():
    #### Initialization of the model ####
    is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

    gpt_model_path = "GPT_SoVITS/pretrained_models/pretrained_sonic/Sonic-SoVITS1-e10.ckpt"
    sovits_model_path = "GPT_SoVITS/pretrained_models/pretrained_sonic/Sonic-SoVITS1_e8_s8736.pth"
    ref_audio_path = shorten_if_longer("I:/Mi unidad/SonicAI/rogerSonicTTSdataset/wavs/audio630.wav")
    version = "v2"

    ##Initialize model config
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = "cuda"
    tts_config.is_half = is_half
    tts_config.version = version #or #v1
    tts_config.t2s_weights_path = gpt_model_path
    tts_config.vits_weights_path = sovits_model_path

    logger.debug(tts_config)

    tts_pipeline = TTS(tts_config)
    gpt_path = tts_config.t2s_weights_path
    sovits_path = tts_config.vits_weights_path
    version = tts_config.version
    tts_pipeline.set_ref_audio(ref_audio_path) ## Set the default audio


    #### Default values ####
    text_language = "en"
    top_k = 5 #min 1 max 100
    top_p = 1 #min 0 max 1
    temperature = 1
    text_split_method = "English"
    batch_size = 20
    speed_factor = 1 #min 0.6 max 1.65
    split_bucket = True #Bool -> Data Bucketing (reduces some computation when using parallel inference)
    parallel_infer = True
    fragment_interval = 0.3 #Dont know the importance of this value yet Segment Interval (Seconds) float
    keep_random = True #True THIS PERMITS THAT THE VOICE OUTPUT IS RANDOM, ALLOWING VARIETY
    repetition_penalty = 1.35 #dont ask me
    prompt_text = "At Sonic Stadium asks, Dear Eggman and Shadow, We're thinking about rebranding from The Sonic Stadium but can't decide on anything. Can we ask for your input? You know what, I'll take this. I mean..."

def initialize_model(gpt_model_path:str,sovits_model_path:str,ref_audio_path:str,version="v2",languages="en"):
    is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

    gpt_model_path = gpt_model_path
    sovits_model_path=sovits_model_path
    ref_audio_path= shorten_if_longer(ref_audio_path),
    version="v2"

    ##Initialize model config
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = "cuda"
    tts_config.is_half = is_half
    tts_config.version = version #or #v1
    tts_config.t2s_weights_path = gpt_model_path
    tts_config.vits_weights_path = sovits_model_path
    tts_config.languages=languages

    logger.info(tts_config)
    tts_pipeline = TTS(tts_config)
    tts_config.t2s_weights_path
    tts_config.vits_weights_path
    version = tts_config.version
    tts_pipeline.set_ref_audio(ref_audio_path) ## Set the default audio

    return tts_pipeline



######### CHAR_AI API #########

TOKEN= os.getenv("CAI_TOKEN","")

async def main():
    char = input('CHAR ID: ')

    client = aiocai.Client(TOKEN)

    me = await client.get_me()

    async with await client.connect() as chat:
        new, answer = await chat.new_chat(
            char, me.id
        )

        fast_inference(answer.text,top_k,top_p,temperature,text_split_method,split_bucket,fragment_interval,parallel_infer,repetition_penalty)

        logger.info(f'{answer.name}: {answer.text}')

        
        while True:
            text = input('YOU: ')

            message = await chat.send_message(
                char, new.chat_id, text
            )

            fast_inference(message.text,top_k,top_p,temperature,text_split_method,split_bucket,fragment_interval,parallel_infer,repetition_penalty)

            logger.info(f'{message.name}: {message.text}')





if __name__ == "__main__":
    import queue
    audio_queue = queue.Queue()
# Initialize the queue
    try:
        asyncio.run(main())
        # while True:
        #     text = input("Sonic's speech: ")
        #Start thread
        # play_thread = threading.Thread(target=play_audio, daemon=True)
        # play_thread.start()
        #     fast_inference(text,top_k,top_p,temperature,text_split_method,split_bucket,fragment_interval,parallel_infer,repetition_penalty,ref_audio_path=ref_audio_path)
    except KeyboardInterrupt:
        logger.info("\nExiting...")
    finally:
        audio_queue.put(None)
        # Signal threads to stop and wait for them
        logger.info("Cleanup completed.")
        
