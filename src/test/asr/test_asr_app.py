import requests

def test_asr_upload():
    audio_path = "/mnt/e/Para-Minds-SpeechXI/src/test/asr/target.wav"
    files = {"audio": open(audio_path, "rb")}
    data = {"initial_prompt": "Start transcription"}

    resp = requests.post("http://127.0.0.1:8000/", files=files, data=data)
    print(resp.status_code, resp.text)

if __name__ == "__main__":
    test_asr_upload()
