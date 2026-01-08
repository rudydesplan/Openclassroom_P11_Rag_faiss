import asyncio
import orjson
from pathlib import Path
from progress_manager import ProgressManager
from async_rest_batch_client import AsyncRestBatchClient  # <— NEW REST CLIENT

INPUT_FILE = Path("documents_for_faiss.jsonl")
OUTPUT_FILE = Path("synthetic_eval_dataset.jsonl")

PARALLEL_WORKERS = 1   # only one batch request at a time
BATCH_SIZE = 32        # keep this max
COOLDOWN = 12          # allows 5 RPM

pm = ProgressManager("progress.txt")
client = AsyncRestBatchClient()  # <— USE REST BATCH CLIENT


async def worker(queue, f_out_lock):
    """Worker that consumes batches from the queue."""
    while True:
        item = await queue.get()
        if item is None:      # Shutdown signal
            queue.task_done()
            return

        uids, texts = item
        results = await client.batch_generate(texts)   # <— REST batch

        async with f_out_lock:
            with OUTPUT_FILE.open("ab") as f_out:
                for uid, text, result in zip(uids, texts, results):

                    # Case 1: No result
                    if not result:
                        print(f"❌ UID={uid} failed — marking done (avoid infinite loop)")
                        pm.mark_done(uid)
                        continue

                    # Case 2: Missing JSON keys
                    if "query" not in result or "expected_answer" not in result:
                        print(f"⚠ Missing keys for UID={uid}")
                        pm.mark_done(uid)
                        continue

                    # Case 3: Valid result
                    record = {
                        "uid": uid,
                        "text": text,
                        "query": result["query"],
                        "expected_answer": result["expected_answer"],
                    }

                    f_out.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                    f_out.flush()
                    pm.mark_done(uid)

        await asyncio.sleep(COOLDOWN)
        queue.task_done()


async def main():
    queue = asyncio.Queue()
    f_out_lock = asyncio.Lock()

    # Start workers
    workers = [
        asyncio.create_task(worker(queue, f_out_lock))
        for _ in range(PARALLEL_WORKERS)
    ]

    batch_uids = []
    batch_texts = []

    # Read input & push batches to queue
    with INPUT_FILE.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            item = orjson.loads(line)
            uid = item.get("uid")
            text = item.get("text", "")

            if pm.is_done(uid):
                continue

            batch_uids.append(uid)
            batch_texts.append(text)

            if len(batch_uids) >= BATCH_SIZE:
                await queue.put((batch_uids.copy(), batch_texts.copy()))
                batch_uids.clear()
                batch_texts.clear()

        # Submit leftover batch
        if batch_uids:
            await queue.put((batch_uids.copy(), batch_texts.copy()))

    # Stop workers
    for _ in workers:
        await queue.put(None)

    await queue.join()

    for w in workers:
        await w


if __name__ == "__main__":
    asyncio.run(main())
