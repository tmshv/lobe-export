from argparse import ArgumentParser, BooleanOptionalAction
from multiprocessing import cpu_count, Pool
from tqdm import tqdm
from PIL import Image
from typing import Callable, List, Optional
from pathlib import Path
import pandas as pd
import imagehash
import sqlite3
import os


def read_db(path: Path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    cursor = con.cursor()
    query = """
        select
            i.example_id,
            i.hash,
            json_extract(i.metadata, '$.filename') as filename,
            l.label,
            l.modified as date,
            m.accuracy
        from example_images i
        join example_labels l on i.example_id = l.example_id
        join example_metrics m on i.example_id = m.example_id
        join examples e on i.example_id = e.example_id
        order by l.modified
        ;
    """
    items = []
    for row in cursor.execute(query):
        items.append({
            "example_id": row["example_id"],
            "hash": row["hash"],
            "filename": row["filename"],
            "label": row["label"],
            "date": row["date"],
            "accuracy": row["accuracy"],
        })
    return pd.DataFrame(items)


def get_image_phash(item_img: Image) -> str:
    ph = imagehash.phash(item_img)
    return str(ph)    


def get_img(path: Path):
    if not path.is_file():
        return None
    try:
        img = Image.open(path)
        return img
    except:
        return None


def run_unpack(args):
    return run(*args)


def run(photo_id: str, base_path: Path):
    img = get_img(base_path / photo_id)
    if not img:
        return None
    return get_image_phash(img)


def calc_phashes(base_path: Path, names: List[str], workers: int):
    with Pool(workers) as pool:
        params = [(name, blob_path) for name in names]
        return list(tqdm(pool.imap(run_unpack, params, chunksize=5), total=len(names)))


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--project",
        required=True,
        dest="lobe_project",
        type=str,
        help="lobe project id (name of folder actually)",
    )
    parser.add_argument(
        "--workers",
        required=False,
        dest="workers",
        type=int,
        help="amount of workers",
    )
    parser.add_argument(
        "--phash",
        required=False,
        default=False,
        dest="phash",
        action=BooleanOptionalAction,
        type=int,
        help="add phash column to csv file",
    )
    parser.add_argument(
        "--cache",
        required=False,
        dest="cache_folder",
        type=str,
        help="path to cache folder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    project = args.lobe_project
    base_path = Path('~/Library/Application Support/Lobe/projects').expanduser()
    db_path = base_path / project / 'db.sqlite'
    blob_path = base_path / project / 'data/blobs'

    df = read_db(db_path)
    names = list(df['hash'])
    print(f"Found {len(names)} items")

    try:
        if args.phash:
            n = args.workers if args.workers else cpu_count()
            print(f'Using {n} workers to calculate phash')
            phash = calc_phashes(blob_path, names, workers=n)
            df['phash'] = phash
        df.to_csv(f'lobe-{project}_export.csv', index=False)
    except KeyboardInterrupt:
        exit()

