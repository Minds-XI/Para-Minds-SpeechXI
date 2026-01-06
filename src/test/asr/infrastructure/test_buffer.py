import asyncio
from asr.domain.entities import ASRResponse
from asr.infrastructure.whisper.entities import HypothesisBuffer  # adjust import to match your path


def test_hypothesis_buffer_basic():
    """
    Test the buffer with a realistic ASR stream simulation.
    """

    buf = HypothesisBuffer()

    # --- Step 1: first partial recognition ---
    chunk_1 = [
        ASRResponse(start=0.0, end=0.3, word="Hello"),
        ASRResponse(start=0.3, end=0.6, word="world"),
    ]

    buf.insert(chunk_1, offset=0)
    print("After first insert:", [x.word for x in buf.new])

    committed = buf.flush()
    print("Committed after flush 1:", [x.word for x in committed])
    assert committed is None

    # --- Step 2: new partial overlapping recognition (with repetition) ---
    chunk_2 = [
        ASRResponse(start=0.0, end=0.3, word="Hello"),
        ASRResponse(start=0.3, end=0.6, word="world"),
        ASRResponse(start=0.6, end=0.9, word="from"),
        ASRResponse(start=0.9, end=1.2, word="ParaMIND"),
    ]

    buf.insert(chunk_2, offset=0)
    print("After second insert:", [x.word for x in buf.new])

    committed = buf.flush()
    print("Committed after flush 2:", [x.word for x in committed])
    print("All committed:", [x.word for x in buf.commited])
    assert committed == ["Hello","world"]
    # --- Step 3: third update, duplicate prevention ---
    chunk_3 = [
        ASRResponse(start=0.6, end=0.9, word="from"),
        ASRResponse(start=0.9, end=1.2, word="ParaMIND"),
        ASRResponse(start=1.2, end=1.5, word="team"),
    ]
    buf.insert(chunk_3, offset=0)
    committed = buf.flush()
    print("Committed after flush 3:", [x.word for x in committed])
    print("Final committed:", [x.word for x in buf.commited])
    assert committed == ["Hello","world","from","ParaMIND"]

    # --- Step 4: pop old commits ---
    buf.pop_commited(time=0.8)
    print("After popping old commits:", [x.word for x in buf.commited])

    # --- Step 5: complete remaining buffer ---
    print("Buffer completion:", [x.word for x in buf.complete()])


if __name__ == "__main__":
    test_hypothesis_buffer_basic()
