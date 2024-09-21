import os
from pathlib import Path


def init_system():
    """ """
    onix_dir = Path.home() / '.onix'
    if not onix_dir.exists():
        os.makedirs(onix_dir)

    log_dir = onix_dir / 'logs'
    if not log_dir.exists():
        os.makedirs(log_dir)

    db_path = onix_dir / 'onix.db'
    if not db_path.exists():
        from onix.db import create_db
        create_db(db_path)


def create_db(db_path: Path):
    """ """
    # Create the sqlite database
    import sqlite3
    conn = sqlite3.connect(db_path)
