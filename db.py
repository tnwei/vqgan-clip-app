import sqlite3

# Excluding "image_prompt BLOB" for now
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER NOT NULL UNIQUE,
    timestamp TEXT NOT NULL,
    num_steps INTEGER NOT NULL,
    text_prompt TEXT,
    output BLOB NOT NULL,
    animation BLOB NOT NULL,
    PRIMARY KEY(id AUTOINCREMENT)
);
"""


def create_table(conn):

    try:
        c = conn.cursor()
        c.execute(CREATE_TABLE_SQL)
        c.close()
        conn.commit()

    except sqlite3.Error as e:
        print(e)


def write_to_db(conn, timestamp, num_steps, text_prompt, output, animation):
    """
    Example:

    ```
    import datetime
    with open("ui.jpeg", "rb") as f:
        imblob = f.read()

    with open("temp.mp4", "rb") as f:
        animblob = f.read()

    write_to_db(conn, datetime.datetime.now(), 100, 'what a wonderful world', imblob, animblob)

    """

    try:
        c = conn.cursor()
        app_sql = f"""
        INSERT INTO runs (timestamp, num_steps, text_prompt, output, animation)
        VALUES (?, ?, ?, ?, ?);
        """

        res = c.execute(
            app_sql,
            [
                timestamp,
                num_steps,
                text_prompt,
                sqlite3.Binary(output),
                sqlite3.Binary(animation),
            ],
        )
        print(res.fetchall())
        c.close()
        conn.commit()

    except sqlite3.Error as e:
        print(e)
