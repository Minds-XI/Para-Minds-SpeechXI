from client.app import audio_sender_uc,filter_speech_uc,ingest_audio_uc


def main():
    try:
        while True:
            try:
                frame = ingest_audio_uc.ingest()
            except ValueError as e:
                print(e)
                continue
            vad_response = filter_speech_uc.filter(frame=frame)
            audio_sender_uc.send(frames=vad_response)
        
    except KeyboardInterrupt:
        print("\n[client] mic stopped by user")
    except Exception as e:
        print(f"[client] error: {e}")
    finally:
        ingest_audio_uc.close()


if __name__ == "__main__":
    main()