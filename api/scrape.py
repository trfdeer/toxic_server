import asyncio
import csv
import sys
from typing import *

import requests

instances_response = requests.get(
    "https://api.invidious.io/instances.json?sort_by=health")
instances = [y["uri"] for y in list(
    map(lambda x: x[1], instances_response.json())) if y["api"] is True]

instance = instances[0]


def get_comments(instance: str, videoId: str, continuation: str = "", currentCount=0, maxCount=500):
    if (currentCount >= maxCount):
        return []

    firstPage = False
    req_url = f"{instance}/api/v1/comments/{videoId}"

    if len(continuation.strip()) > 0:
        req_url += f"?continuation={continuation}"
    else:
        firstPage = True

    resp = requests.get(req_url).json()
    currentCommentCount = len(resp["comments"])
    if firstPage:
        print(f"[{videoId}] Total {resp['commentCount']} comments.")

    yield list(map(lambda x: (x["commentId"], r"{}".format(x["content"])), resp["comments"]))

    try:
        continuation = resp["continuation"]
        yield from get_comments(instance, videoId, continuation, maxCount=maxCount, currentCount=currentCount+currentCommentCount)
    except KeyError:
        print(f"[{videoId}] Completed.")
        return list(map(lambda x: (x["commentId"], x["content"]), resp["comments"]))


def download_to_csv(videoId: str, commentLimit: int):
    filename = f"{videoId}.csv"
    with open(filename, "w") as f:
        f.write("id, comment_text\n")

    for comments in get_comments(instance, videoId, maxCount=commentLimit):
        with open(filename, "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(comments)
            print(f"[{videoId}] Wrote {len(comments)} comments.")


async def main():
    commentLimit = int(sys.argv[1])
    videoIds = sys.argv[2:]
    print(videoIds)

    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(
            None, download_to_csv, videoId, commentLimit
        )
        for videoId in videoIds
    ]

    await asyncio.gather(*futures)

if __name__ == "__main__":
    asyncio.run(main())
