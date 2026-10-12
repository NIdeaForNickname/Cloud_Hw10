import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import azure.cognitiveservices.speech as speechsdk

VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "https://*****.cognitiveservices.azure.com/")
VISION_KEY = os.environ.get("VISION_KEY", "*****")

SPEECH_KEY = os.environ.get("SPEECH_KEY", "*****")
SPEECH_REGION = os.environ.get("SPEECH_REGION", "*****")


def analyzeImage(image_path: str):
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY)
    )

    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.TAGS]
    )

    if result.tags:
        print("Tags:")
        for tag in result.tags.list:
            print(f"    {tag.name};  Confidence: {tag.confidence:.2f}")
    else:
        print("No tags")


def textRecognition(image_path: str):
    client = ImageAnalysisClient(
        endpoint=VISION_ENDPOINT,
        credential=AzureKeyCredential(VISION_KEY)
    )

    with open(image_path, "rb") as f:
        image_data = f.read()

    result = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    if result.read:
        print("Найденный текст:")
        for line in result.read.blocks[0].lines:
            print(f"  {line.text}")
    else:
        print("Текст на изображении не найден.")


def speechRecognition(audio_path: str):
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    result = speech_recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"Recognized text: {result.text}")
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Fail: {cancellation_details.reason}")


if __name__ == "__main__":
    sample_image = "/home/bigbob/Downloads/library.png"
    sample_speech = "/home/bigbob/Downloads/test01_20s.wav"

    if os.path.exists(sample_image):
        analyzeImage(sample_image)
        textRecognition(sample_image)
    else:
        print(f"Файл {sample_image} не найден. Пропущены шаги анализа изображений.")

    speechRecognition(sample_speech)
