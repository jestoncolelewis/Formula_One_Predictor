import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "db.sqlite3"))
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class RaceResult(Base):
    __tablename__ = "race_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    raceId = Column(Integer, nullable=False)
    grid = Column(Float)
    position = Column(Float)
    year = Column(Integer)
    date = Column(String)
    time = Column(String)
    circuitRef = Column(String)
    driverRef = Column(String)
    constructorRef = Column(String)
    circuit_code = Column(Integer)
    driver_code = Column(Integer)
    constructor_code = Column(Integer)
    pos_delta = Column(Float)
    grid_rolling = Column(Float)
    position_rolling = Column(Float)
    pos_delta_rolling = Column(Float)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trained_at = Column(DateTime)
    accuracy = Column(Float)
    num_samples = Column(Integer)
    notes = Column(String)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
