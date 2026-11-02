import json
from pathlib import Path

from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, insert, select, update


class Store:
    def __init__(self, url):
        if url.startswith("sqlite:///"):
            path = url.removeprefix("sqlite:///")
            if path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, pool_pre_ping=True)
        self.meta = MetaData()
        self.runs = Table(
            "experiments",
            self.meta,
            Column("id", String(36), primary_key=True),
            Column("body", Text, nullable=False),
        )
        self.reviews = Table(
            "reviews",
            self.meta,
            Column("id", String(80), primary_key=True),
            Column("body", Text, nullable=False),
        )
        self.meta.create_all(self.engine)

    def put(self, body):
        with self.engine.begin() as connection:
            exists = connection.execute(select(self.runs.c.id).where(self.runs.c.id == body["id"])).first()
            if exists:
                connection.execute(
                    update(self.runs).where(self.runs.c.id == body["id"]).values(body=json.dumps(body))
                )
            else:
                connection.execute(insert(self.runs).values(id=body["id"], body=json.dumps(body)))

    def get(self, run_id):
        with self.engine.connect() as connection:
            row = connection.execute(select(self.runs.c.body).where(self.runs.c.id == run_id)).first()
            return json.loads(row[0]) if row else None

    def list(self):
        with self.engine.connect() as connection:
            bodies = [json.loads(row[0]) for row in connection.execute(select(self.runs.c.body))]
        return sorted(bodies, key=lambda b: b.get("created_at", ""), reverse=True)[:100]

    def review(self, run_id, body):
        key = f"{run_id}:{body['case_id']}"
        with self.engine.begin() as connection:
            if connection.execute(select(self.reviews.c.id).where(self.reviews.c.id == key)).first():
                connection.execute(
                    update(self.reviews).where(self.reviews.c.id == key).values(body=json.dumps(body))
                )
            else:
                connection.execute(insert(self.reviews).values(id=key, body=json.dumps(body)))

    def get_reviews(self, run_id):
        with self.engine.connect() as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    select(self.reviews.c.body).where(self.reviews.c.id.like(run_id + ":%"))
                )
            ]
